"""
Ensambla el sistema completo de EDOs MPD para Schwarzschild-MOG y lo
compila a funciones numericas (numpy) via lambdify.

Sistema dinamico (8 variables): y = (t, r, theta, phi, p_t, p_r, p_th, p_ph)
Las 6 componentes de S^{munu} se reconstruyen algebraicamente en cada paso
usando las formulas VERIFICADAS (con la correccion a S^{tphi} que
la contracción correcta fija el orden de p_theta y p_phi implementado aquí).

Ecuaciones:
  dx^mu/dtau = u^mu               (u de la relacion TDSSC, ec. 3-5, normalizada)
  dp_mu/dtau = Gamma^lam_{mu nu} u^nu p_lam - (1/2) R_{mu nu ab} u^nu S^{ab}

Nota sobre la ley de fuerza covariante: la ec. 3-2 es para Dp^mu/Dtau
(contravariante). En componentes covariantes:
  dp_mu/dtau = Gamma^lam_{mu nu} u^nu p_lam - (1/2) R_{mu nu rho sig} u^nu S^{rho sig}
"""
import sympy as sp
import pickle
import time

t0 = time.time()

with open('riemann_symbolic.pkl', 'rb') as fh:
    data = pickle.load(fh)
Gamma_g = data['Gamma']
Riem_low_g = data['Riem_low']
f_sym = data['f']
g_sym = data['g']
ginv_sym = data['ginv']
coords = data['coords']
t, r, th, ph = coords

alpha, M, m, J = sp.symbols('alpha M m J', real=True)
pt, pr, pth, pph = sp.symbols('p_t p_r p_theta p_phi', real=True)
p_low = [pt, pr, pth, pph]
n = 4

# ---------------------------------------------------------------------
# Sustituir f(r) de MOG en toda la geometria
# ---------------------------------------------------------------------
f_mog = 1 - 2*M*(1+alpha)/r + alpha*(1+alpha)*M**2/r**2

def sub_f(expr):
    if expr == 0:
        return sp.S.Zero
    return sp.simplify(expr.replace(f_sym, f_mog).doit())

print("Sustituyendo f_MOG en Christoffel...")
Gamma = [[[sub_f(Gamma_g[a][b][c]) for c in range(n)] for b in range(n)] for a in range(n)]
print("Sustituyendo f_MOG en Riemann...", flush=True)
Riem_low = [[[[sub_f(Riem_low_g[a][b][c][d]) for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]
g = g_sym.replace(f_sym, f_mog)
ginv = ginv_sym.replace(f_sym, f_mog)
print(f"  geometria lista ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------
# Tensor de spin S^{munu} (juego VERIFICADO, con correccion en S^tphi)
# ---------------------------------------------------------------------
S_phr = (pph - J*sp.sin(th)**2)/(r*sp.sin(th)**2)
S_thph = J*sp.cos(th)/(r**2*sp.sin(th))
S_rth = -pth/r
S_tr = -(pth**2*sp.sin(th)**2 + pph**2 - J*pph*sp.sin(th)**2)/(pt*r*sp.sin(th)**2)
S_tth = (pr*pth*r*sp.sin(th) + J*pph*sp.cos(th))/(pt*r**2*sp.sin(th))
# Componente S^{tphi} de la condición de Tulczyjew-Dixon:
S_tph = (-J*pr*r - J*pth/sp.tan(th) + pph*pr*r/sp.sin(th)**2)/(pt*r**2)

Sup = sp.zeros(4, 4)
Sup[0, 1] = S_tr;   Sup[1, 0] = -S_tr
Sup[0, 2] = S_tth;  Sup[2, 0] = -S_tth
Sup[0, 3] = S_tph;  Sup[3, 0] = -S_tph
Sup[1, 2] = S_rth;  Sup[2, 1] = -S_rth
Sup[3, 1] = S_phr;  Sup[1, 3] = -S_phr
Sup[2, 3] = S_thph; Sup[3, 2] = -S_thph

# ---------------------------------------------------------------------
# u^mu segun TDSSC (ec. 3-5):
#   u^mu propto p^mu + 2 S^{mu nu} p^lam R_{nu lam rho sig} S^{rho sig} / Y
#   Y = 4 m_bar^2 + R_{abcd} S^{ab} S^{cd},   m_bar^2 = -p.p
# Luego dt/dtau etc se obtienen normalizando: u.u = -1
# ---------------------------------------------------------------------
print("Construyendo u^mu (esto contrae Riemann con S dos veces)...", flush=True)

p_up = [sum(ginv[a, b]*p_low[b] for b in range(n)) for a in range(n)]
m2bar = -sum(p_low[a]*p_up[a] for a in range(n))   # = m^2 dinamico

# R_{nu lam rho sig} S^{rho sig} (contraccion sobre rho,sig)
RS = [[sp.S.Zero]*n for _ in range(n)]
for nu in range(n):
    for lam in range(n):
        s = sp.S.Zero
        for rho in range(n):
            for sig in range(n):
                if Riem_low[nu][lam][rho][sig] != 0 and Sup[rho, sig] != 0:
                    s += Riem_low[nu][lam][rho][sig]*Sup[rho, sig]
        RS[nu][lam] = s

# Y = 4 m2bar + R_{abcd} S^{ab} S^{cd}
RSS = sp.S.Zero
for a in range(n):
    for b in range(n):
        if Sup[a, b] != 0 and RS[a][b] != 0:
            RSS += Sup[a, b]*RS[a][b]
Y = 4*m2bar + RSS

# w^mu = p^mu + 2 S^{mu nu} (p^lam R_{nu lam rho sig} S^{rho sig}) / Y
pRS = [sum(p_up[lam]*RS[nu][lam] for lam in range(n)) for nu in range(n)]
w_up = []
for mu in range(n):
    corr = sum(Sup[mu, nu]*pRS[nu] for nu in range(n))
    w_up.append(p_up[mu] + 2*corr/Y)

# normalizar: u^mu = w^mu / sqrt(-w.w)
w2 = sum(g[a, b]*w_up[a]*w_up[b] for a in range(n) for b in range(n))
norm = sp.sqrt(-w2)
u_up = [w/norm for w in w_up]

print(f"  u^mu construido ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------
# Ley de fuerza covariante:
# dp_mu/dtau = Gamma^lam_{mu nu} u^nu p_lam - (1/2) R_{mu nu rho sig} u^nu S^{rho sig}
# Nota: R_{mu nu rho sig} u^nu S^{rho sig} = RS_transpuesto... cuidado con el orden
# de indices: RS[nu][lam] = R_{nu lam rho sig} S^{rho sig}; necesitamos
# R_{mu nu rho sig} u^nu S^{rho sig} = sum_nu RS[mu][nu] u^nu
# ---------------------------------------------------------------------
print("Construyendo dp_mu/dtau...", flush=True)
dp = []
for mu in range(n):
    geo = sum(Gamma[lam][mu][nu]*u_up[nu]*p_low[lam] for lam in range(n) for nu in range(n))
    spin_force = sp.Rational(1, 2)*sum(RS[mu][nu]*u_up[nu] for nu in range(n))
    dp.append(geo - spin_force)

print(f"  sistema completo ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------
# Compilar a codigo fuente Python (portable, sin dependencias de pickle)
# ---------------------------------------------------------------------
print("Generando codigo fuente numerico...", flush=True)
from sympy.printing.pycode import pycode
from sympy.simplify.cse_main import cse

all_exprs = list(u_up) + dp + [m2bar, Y, w2, None]  # placeholder

# S^2 (magnitud de spin) para monitoreo de conservacion
Slow = sp.zeros(4, 4)
for a in range(n):
    for b in range(n):
        Slow[a, b] = sum(g[a, c]*g[b, d]*Sup[c, d] for c in range(n) for d in range(n))
S2_expr = sp.Rational(1, 2)*sum(Slow[a, b]*Sup[a, b] for a in range(n) for b in range(n))
all_exprs[-1] = S2_expr

replacements, reduced = cse(all_exprs, optimizations='basic')

lines = []
lines.append('"""')
lines.append("Sistema MPD compilado para Schwarzschild-MOG (generado automaticamente")
lines.append("por build_mpd_system.py -- NO editar a mano).")
lines.append("mpd_all(r, theta, p_t, p_r, p_theta, p_phi, J, alpha, M, m) ->")
lines.append("  (u^t, u^r, u^th, u^ph, dp_t, dp_r, dp_th, dp_ph, m2bar, Y, w2, S2)")
lines.append('"""')
lines.append("from numpy import sin, cos, tan, sqrt, pi")
lines.append("")
lines.append("def mpd_all(r, theta, p_t, p_r, p_theta, p_phi, J, alpha, M, m):")
for sym, expr in replacements:
    lines.append(f"    {sym} = {pycode(expr)}")
ret_names = []
for i, expr in enumerate(reduced):
    name = f"out{i}"
    lines.append(f"    {name} = {pycode(expr)}")
    ret_names.append(name)
lines.append(f"    return ({', '.join(ret_names)})")

with open('equations_of_motion.py', 'w') as fh:
    fh.write("\n".join(lines))

print(f"LISTO ({time.time()-t0:.1f}s). Generado mpd_rhs.py "
      f"({len(replacements)} subexpresiones comunes)")
