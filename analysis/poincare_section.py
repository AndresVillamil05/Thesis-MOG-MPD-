"""
Secciones de Poincare para la particula con spin en Schwarzschild-MOG.

Seccion: theta = pi/2, cruces con d(theta)/dtau > 0. Se grafica (r, p_r).

Para que la seccion sea valida, TODAS las trayectorias deben compartir
exactamente las mismas constantes (E, J, S, m). Dado (r0, p_theta0),
resolvemos (p_t, p_r, p_phi) del sistema no lineal:
    E(p)  = E_objetivo      [E = -p_t - (1/2) f' S^{tr}]
    m2(p) = 1               [mass shell]
    S2(p) = S_objetivo^2    [magnitud del spin]
"""
import numpy as np
from scipy.optimize import fsolve
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from integrator import f_mog, fprime_mog, horizon_mog, rhs
from scipy.integrate import solve_ivp


def constraints_eq(p, r0, pth0, J, S, alpha, M=1.0, m=1.0):
    """Sistema E, m2, S2 en theta=pi/2 (sin=1, cos=0)."""
    p_t, p_r, p_ph = p
    fr = f_mog(r0, alpha, M)
    frp = fprime_mog(r0, alpha, M)
    # S^{tr} en theta=pi/2:
    Str = -(pth0**2 + p_ph**2 - J*p_ph)/(p_t*r0)
    E = -p_t - 0.5*frp*Str
    m2 = p_t**2/fr - fr*p_r**2 - pth0**2/r0**2 - p_ph**2/r0**2
    # S^2 en theta=pi/2 (derivada simbolicamente, verificada):
    S2 = (pth0**2/fr + (J - p_ph)**2/fr
          - fr*p_r**2*pth0**2/p_t**2
          - (p_ph**2 - J*p_ph + pth0**2)**2/(p_t**2*r0**2)
          - fr*p_r**2*(J - p_ph)**2/p_t**2)
    return E, m2, S2


def build_ic(r0, pth0, E_target, J, S, alpha, M=1.0, m=1.0,
             sign_pr=+1.0, guess=None):
    """Resuelve (p_t,p_r,p_phi) para las constantes objetivo. Devuelve y0 o None.
    Intenta una grilla de puntos de arranque para el solver no lineal."""
    def eqs(p):
        E, m2, S2 = constraints_eq(p, r0, pth0, J, S, alpha, M, m)
        return [E - E_target, m2 - m**2, S2 - S**2]

    guesses = []
    if guess is not None:
        guesses.append(guess)
    for pr_g in [0.05, 0.2, 0.5, 0.0]:
        for pph_g in [J - S, J - S/2, J, J - 1.5*S]:
            for pt_g in [-E_target, -1.05*E_target, -0.9*E_target]:
                guesses.append([pt_g, sign_pr*pr_g, pph_g])

    for g in guesses:
        sol, info, ier, msg = fsolve(eqs, g, full_output=True, xtol=1e-13)
        if ier != 1:
            continue
        p_t, p_r, p_ph = sol
        if p_t >= 0:
            continue
        res = eqs([p_t, p_r, p_ph])
        if max(abs(x) for x in res) > 1e-9:
            continue
        if sign_pr*p_r < 0:
            p_r2 = -p_r  # la otra rama de p_r (simetria del mass-shell)
            if max(abs(x) for x in eqs([p_t, p_r2, p_ph])) < 1e-9:
                p_r = p_r2
        return np.array([0.0, r0, np.pi/2, 0.0, p_t, p_r, pth0, p_ph])
    return None


def poincare_section(y0, J, alpha, tau_max, M=1.0, m=1.0,
                     rtol=1e-10, atol=1e-12, max_crossings=2000):
    """Integra y devuelve los cruces (r, p_r) por theta=pi/2 con dtheta/dtau>0."""
    r_h = horizon_mog(alpha, M)

    def crossing(tau, y, *args):
        return y[2] - np.pi/2
    crossing.terminal = False
    crossing.direction = +1

    def horizon_event(tau, y, *args):
        return y[1] - r_h*1.10
    horizon_event.terminal = True
    horizon_event.direction = -1

    def escape_event(tau, y, *args):
        return y[1] - 80.0
    escape_event.terminal = True
    escape_event.direction = +1

    sol = solve_ivp(rhs, [0, tau_max], y0, args=(J, alpha, M, m),
                    method='DOP853', rtol=rtol, atol=atol,
                    events=[crossing, horizon_event, escape_event])
    ye = sol.y_events[0]
    if len(ye) == 0:
        return np.empty((0, 2)), sol.status
    pts = np.column_stack([ye[:max_crossings, 1], ye[:max_crossings, 5]])
    return pts, sol.status


if __name__ == "__main__":
    # ---- prueba con UNA trayectoria: verificar constantes y cruces ----
    from integrator import conserved
    M = m = 1.0
    alpha, J, S = 0.3, 5.0, 1.0
    E_target = 0.945

    y0 = build_ic(r0=7.0, pth0=0.4, E_target=E_target, J=J, S=S, alpha=alpha)
    print("y0 =", y0)
    if y0 is not None:
        E0, Jz0, m20, S20 = conserved(y0, J, alpha, M, m)
        print(f"constantes logradas: E={E0:.12f} (objetivo {E_target})  "
              f"Jz={Jz0:.10f}  m2={m20:.10f}  S2={S20:.10f} (objetivo {S**2})")
        pts, status = poincare_section(y0, J, alpha, tau_max=3000.0)
        print(f"cruces registrados: {len(pts)}  status={status}")
        if len(pts):
            print("primeros cruces (r, p_r):")
            print(pts[:5])
