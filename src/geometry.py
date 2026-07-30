"""
Derivacion simbolica (sympy) de:
  - simbolos de Christoffel Gamma^a_bc
  - tensor de Riemann R^a_bcd  y su forma totalmente covariante R_abcd
para la metrica de Schwarzschild-MOG:
  ds^2 = -f(r) dt^2 + f(r)^{-1} dr^2 + r^2 dtheta^2 + r^2 sin^2(theta) dphi^2

Se deja f(r) como funcion generica simbolica durante la derivacion, y solo
al final se sustituye la forma especifica de MOG. Esto permite verificar
el resultado contra Schwarzschild (f=1-2M/r) de forma inmediata.
"""
import sympy as sp

t, r, th, ph, M, alpha = sp.symbols('t r theta phi M alpha', real=True)
coords = [t, r, th, ph]
n = 4

f = sp.Function('f')(r)  # generico por ahora

g = sp.diag(-f, 1/f, r**2, r**2*sp.sin(th)**2)
ginv = g.inv()

print("Metrica g_ab:")
sp.pprint(g)

# ---------------------------------------------------------------------
# Simbolos de Christoffel: Gamma^a_bc = 1/2 g^ad (d_b g_dc + d_c g_db - d_d g_bc)
# ---------------------------------------------------------------------
Gamma = [[[0]*n for _ in range(n)] for _ in range(n)]
for a in range(n):
    for b in range(n):
        for c in range(n):
            s = 0
            for d in range(n):
                s += ginv[a, d]*(sp.diff(g[d, c], coords[b])
                                 + sp.diff(g[d, b], coords[c])
                                 - sp.diff(g[b, c], coords[d]))
            Gamma[a][b][c] = sp.simplify(s/2)

print("\nSimbolos de Christoffel no nulos (Gamma^a_bc):")
names = ['t', 'r', 'theta', 'phi']
nonzero_christoffel = {}
for a in range(n):
    for b in range(n):
        for c in range(n):
            if Gamma[a][b][c] != 0:
                key = f"Gamma^{names[a]}_{names[b]}{names[c]}"
                if (c, b) != (b, c) and b > c:
                    continue  # simetria b<->c, mostrar solo una vez
                nonzero_christoffel[key] = Gamma[a][b][c]
                print(f"  {key} = {Gamma[a][b][c]}")

# ---------------------------------------------------------------------
# Tensor de Riemann: R^a_bcd = d_c Gamma^a_db - d_d Gamma^a_cb
#                              + Gamma^a_ce Gamma^e_db - Gamma^a_de Gamma^e_cb
# ---------------------------------------------------------------------
Riem_up = [[[[0]*n for _ in range(n)] for _ in range(n)] for _ in range(n)]
for a in range(n):
    for b in range(n):
        for c in range(n):
            for d in range(n):
                term = sp.diff(Gamma[a][d][b], coords[c]) - sp.diff(Gamma[a][c][b], coords[d])
                for e in range(n):
                    term += Gamma[a][c][e]*Gamma[e][d][b] - Gamma[a][d][e]*Gamma[e][c][b]
                Riem_up[a][b][c][d] = sp.simplify(term)

# forma totalmente covariante R_abcd = g_ae R^e_bcd
Riem_low = [[[[0]*n for _ in range(n)] for _ in range(n)] for _ in range(n)]
for a in range(n):
    for b in range(n):
        for c in range(n):
            for d in range(n):
                s = 0
                for e in range(n):
                    s += g[a, e]*Riem_up[e][b][c][d]
                Riem_low[a][b][c][d] = sp.simplify(s)

print("\nComponentes no nulas de R_abcd (hasta simetria estandar):")
nonzero_riemann = {}
for a in range(n):
    for b in range(n):
        for c in range(n):
            for d in range(n):
                val = Riem_low[a][b][c][d]
                if val != 0:
                    key = (a, b, c, d)
                    nonzero_riemann[key] = val

# Filtrar representantes independientes usando las simetrias:
# R_abcd = -R_bacd = -R_abdc = R_cdab
seen = set()
independent = {}
for (a, b, c, d), val in nonzero_riemann.items():
    canon_candidates = [(a, b, c, d), (b, a, c, d), (a, b, d, c), (b, a, d, c),
                         (c, d, a, b), (d, c, a, b), (c, d, b, a), (d, c, b, a)]
    if any(cc in seen for cc in canon_candidates):
        continue
    seen.add((a, b, c, d))
    independent[(a, b, c, d)] = val

for (a, b, c, d), val in independent.items():
    print(f"  R_{names[a]}{names[b]}{names[c]}{names[d]} = {val}")

# ---------------------------------------------------------------------
# Guardar resultados simbolicos (con f generico) para uso posterior
# ---------------------------------------------------------------------
import pickle
with open('riemann_symbolic.pkl', 'wb') as fh:
    pickle.dump({
        'Gamma': Gamma, 'Riem_up': Riem_up, 'Riem_low': Riem_low,
        'coords': coords, 'f': f, 'g': g, 'ginv': ginv,
    }, fh)

print("\nGuardado riemann_symbolic.pkl")
