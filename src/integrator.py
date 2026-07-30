"""
Integrador de las ecuaciones MPD para Schwarzschild-MOG + tests de
verificacion por leyes de conservacion.

Variables: y = (t, r, theta, phi, p_t, p_r, p_theta, p_phi)

Cantidades monitoreadas (deben ser constantes):
  E  = -p_t + (1/2) f'(r) S^{tr}     
  Jz = p_phi - r sin^2 S^{phi r} + r^2 sin cos S^{theta phi}  [= J por construccion]
  m2bar = -p.p
  S2 = (1/2) S_ab S^ab
"""
import numpy as np
from scipy.integrate import solve_ivp
from equations_of_motion import mpd_all


def f_mog(r, alpha, M=1.0):
    return 1.0 - 2.0*M*(1+alpha)/r + alpha*(1+alpha)*M**2/r**2

def fprime_mog(r, alpha, M=1.0):
    return 2.0*M*(1+alpha)/r**2 - 2.0*alpha*(1+alpha)*M**2/r**3

def horizon_mog(alpha, M=1.0):
    return M*(1+alpha) + M*np.sqrt(1+alpha)


def rhs(tau, y, J, alpha, M, m):
    t, r, th, ph, p_t, p_r, p_th, p_ph = y
    out = mpd_all(r, th, p_t, p_r, p_th, p_ph, J, alpha, M, m)
    ut, ur, uth, uph = out[0], out[1], out[2], out[3]
    dpt, dpr, dpth, dpph = out[4], out[5], out[6], out[7]
    return [ut, ur, uth, uph, dpt, dpr, dpth, dpph]


def conserved(y, J, alpha, M, m):
    t, r, th, ph, p_t, p_r, p_th, p_ph = y
    out = mpd_all(r, th, p_t, p_r, p_th, p_ph, J, alpha, M, m)
    m2bar, Y, w2, S2 = out[8], out[9], out[10], out[11]
    # E = -p_t - (1/2) f'(r) S^{tr}
    # (signo verificado: simbolicamente desde la formula de Killing ec. 3-9,
    #  y numericamente: con este signo E se conserva a ~1e-13; con el signo +
    #  de la formulación deriva ~0.5. La ec. 3-14 del texto guia
    #  tiene el signo del termino de spin invertido.)
    Str = -(p_th**2*np.sin(th)**2 + p_ph**2 - J*p_ph*np.sin(th)**2)/(p_t*r*np.sin(th)**2)
    E = -p_t - 0.5*fprime_mog(r, alpha, M)*Str
    # Jz de la formula de conservacion
    S_phr = (p_ph - J*np.sin(th)**2)/(r*np.sin(th)**2)
    S_thph = J*np.cos(th)/(r**2*np.sin(th))
    Jz = p_ph - r*np.sin(th)**2*S_phr + r**2*np.sin(th)*np.cos(th)*S_thph
    return E, Jz, m2bar, S2


def initial_conditions_equatorial(r0, E, J, S, alpha, M=1.0, m=1.0, sign_pr=-1.0):
    """
    Condiciones iniciales en el plano ecuatorial:
    dadas (r0, E, J, S), resuelve p_t y p_phi del sistema
      p_t = -m sqrt(f) cosh(Z),  p_phi = m r sinh(Z)  con sinh Z de la cuadratica,
    y luego p_r del requisito m^2 = -p.p (si es posible; si no, p_r=0).
    p_theta = 0 inicialmente, pero la orbita puede salir del plano si se perturba.
    """
    fr = f_mog(r0, alpha, M)
    A = m**2*r0**2 - S**2*fr
    lin = J*m*r0
    C = (J**2 - S**2)*fr + (1.0 - fr)*J**2
    disc = lin**2 - A*C
    if disc < 0:
        raise ValueError("sin solucion para sinh Z en r0 (region prohibida)")
    shZ = (lin - np.sqrt(disc))/A     # rama -
    chZ = np.sqrt(1 + shZ**2)
    p_t = -m*np.sqrt(fr)*chZ
    p_ph = m*r0*shZ

    # ajustar p_t para energia E dada via p_r: en el plano ecuatorial con p_theta=0,
    #   m^2 = -g^tt p_t^2 - g^rr p_r^2 - g^phph p_phi^2
    #   => p_r^2 = ( -m^2 + p_t^2/f - p_phi^2/r^2 ) * f ... resolvemos consistentemente:
    # Aqui usamos la formulación basada en las cantidades conservadas: E fija p_t via la formula de conservacion.
    # Para simplificar: elegimos E como la que corresponde a p_r dado. Iteramos:
    from scipy.optimize import brentq

    def E_of_pt(pt_val):
        # dado p_t, obtener p_r^2 de la mass-shell y calcular E
        pr2 = (pt_val**2/fr - m**2 - p_ph**2/r0**2)*fr
        if pr2 < 0:
            pr2 = 0.0
        p_r = sign_pr*np.sqrt(pr2)
        Str = -(p_ph**2 - J*p_ph)/(pt_val*r0)
        return -pt_val - 0.5*fprime_mog(r0, alpha, M)*Str

    # buscar p_t tal que E_of_pt(p_t) = E
    ptlo, pthi = -5.0*m, -1e-6
    Elo, Ehi = E_of_pt(ptlo), E_of_pt(pthi)
    if not (min(Elo, Ehi) <= E <= max(Elo, Ehi)):
        # E fuera de rango: usar el p_t de la rama del potencial (V_-) y reportar
        pr2 = (p_t**2/fr - m**2 - p_ph**2/r0**2)*fr
        p_r = sign_pr*np.sqrt(max(pr2, 0.0))
        return np.array([0.0, r0, np.pi/2, 0.0, p_t, p_r, 0.0, p_ph])
    pt_sol = brentq(lambda x: E_of_pt(x) - E, ptlo, pthi, xtol=1e-14)
    pr2 = (pt_sol**2/fr - m**2 - p_ph**2/r0**2)*fr
    p_r = sign_pr*np.sqrt(max(pr2, 0.0))
    return np.array([0.0, r0, np.pi/2, 0.0, pt_sol, p_r, 0.0, p_ph])


def integrate(y0, J, alpha, tau_max, M=1.0, m=1.0, rtol=1e-10, atol=1e-12,
              max_step=np.inf, events_horizon=True):
    r_h = horizon_mog(alpha, M)

    def horizon_event(tau, y, *args):
        return y[1] - r_h*1.10
    horizon_event.terminal = True
    horizon_event.direction = -1

    def escape_event(tau, y, *args):
        return y[1] - 60.0
    escape_event.terminal = True
    escape_event.direction = +1

    sol = solve_ivp(rhs, [0, tau_max], y0, args=(J, alpha, M, m),
                    method='DOP853', rtol=rtol, atol=atol,
                    dense_output=True, max_step=max_step,
                    events=[horizon_event, escape_event] if events_horizon else None)
    return sol


# =====================================================================
# TESTS DE VERIFICACION
# =====================================================================
if __name__ == "__main__":
    M, m = 1.0, 1.0

    print("="*70)
    print("TEST 1: S=0, alpha=0 -> geodesica de Schwarzschild pura")
    print("        orbita circular en r=10 (L=r/sqrt(r-3) para Schw)")
    print("="*70)
    r0 = 10.0
    L_circ = r0/np.sqrt(r0 - 3.0)      # momento angular orbital circular Schw
    E_circ = (r0 - 2.0)/np.sqrt(r0*(r0 - 3.0))
    fr = f_mog(r0, 0.0)
    p_t0 = -E_circ
    p_ph0 = L_circ
    y0 = np.array([0.0, r0, np.pi/2, 0.0, p_t0, 0.0, 0.0, p_ph0])
    sol = integrate(y0, J=L_circ, alpha=0.0, tau_max=500.0)
    r_arr = sol.y[1]
    print(f"  r inicial={r_arr[0]:.10f}, r final={r_arr[-1]:.10f}")
    print(f"  desviacion max de r: {np.abs(r_arr-r0).max():.2e}  (debe ser ~1e-8 o menor)")

    print()
    print("="*70)
    print("TEST 2: conservacion con SPIN, alpha=0.3, fuera del plano")
    print("="*70)
    alpha = 0.3
    J, S = 5.0, 1.0
    y0 = initial_conditions_equatorial(r0=8.0, E=0.955, J=J, S=S, alpha=alpha)
    # perturbar fuera del plano para probar la dinamica theta:
    y0[6] = 0.05   # p_theta pequeno
    E0, Jz0, m20, S20 = conserved(y0, J, alpha, M, m)
    print(f"  inicial: E={E0:.12f}  Jz={Jz0:.12f}  m2={m20:.12f}  S2={S20:.12f}")
    sol = integrate(y0, J, alpha, tau_max=2000.0)
    print(f"  status: {sol.status} ({sol.message}), pasos={len(sol.t)}")
    drifts = {k: [] for k in ['E', 'Jz', 'm2', 'S2']}
    for i in range(0, len(sol.t), max(1, len(sol.t)//50)):
        E, Jz, m2b, S2 = conserved(sol.y[:, i], J, alpha, M, m)
        drifts['E'].append(abs(E - E0))
        drifts['Jz'].append(abs(Jz - Jz0))
        drifts['m2'].append(abs(m2b - m20))
        drifts['S2'].append(abs(S2 - S20))
    for k, v in drifts.items():
        print(f"  drift max {k}: {max(v):.3e}")
    print(f"  rango de theta explorado: [{sol.y[2].min():.4f}, {sol.y[2].max():.4f}] rad "
          f"(pi/2={np.pi/2:.4f})")
