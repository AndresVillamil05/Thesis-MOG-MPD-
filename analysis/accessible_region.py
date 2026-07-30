"""
Mapeo sistematico de la region accesible de la seccion de Poincare.

En la seccion (theta=pi/2), dadas las constantes (E, J, S, m) y un punto
(r, p_r), las incognitas (p_t, q=p_theta^2, p_phi) quedan determinadas por
las tres ligaduras E, m^2, S^2 (todas dependen de p_theta solo via q).
Punto accesible <=> existe solucion con q >= 0. Frontera: q = 0.
"""
import numpy as np
from scipy.optimize import fsolve
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from integrator import f_mog, fprime_mog


def section_constraints(u, r, pr, J, S, alpha, M=1.0, m=1.0):
    p_t, q, p_ph = u
    fr = f_mog(r, alpha, M)
    frp = fprime_mog(r, alpha, M)
    Str = -(q + p_ph**2 - J*p_ph)/(p_t*r)
    E = -p_t - 0.5*frp*Str
    m2 = p_t**2/fr - fr*pr**2 - q/r**2 - p_ph**2/r**2
    S2 = (q/fr + (J - p_ph)**2/fr
          - fr*pr**2*q/p_t**2
          - (p_ph**2 - J*p_ph + q)**2/(p_t**2*r**2)
          - fr*pr**2*(J - p_ph)**2/p_t**2)
    return E, m2, S2


def section_solve(r, pr, E_t, J, S, alpha, M=1.0, m=1.0):
    """Devuelve (p_t, q, p_phi) o None. q puede ser negativo (inaccesible)."""
    def eqs(u):
        E, m2, S2 = section_constraints(u, r, pr, J, S, alpha, M, m)
        return [E - E_t, m2 - m**2, S2 - S**2]

    for q_g in [0.05, 0.5, 0.0, 1.5]:
        for pph_g in [J - S, J - S/2, J - 1.5*S]:
            sol, info, ier, msg = fsolve(eqs, [-E_t, q_g, pph_g],
                                          full_output=True, xtol=1e-12)
            if ier != 1:
                continue
            p_t, q, p_ph = sol
            if p_t >= 0:
                continue
            if max(abs(x) for x in eqs(sol)) > 1e-8:
                continue
            return p_t, q, p_ph
    return None


def map_accessible(E_t, J, S, alpha, r_range, pr_range, nr=110, npr=81):
    rs = np.linspace(*r_range, nr)
    prs = np.linspace(*pr_range, npr)
    Q = np.full((npr, nr), np.nan)
    for i, pr in enumerate(prs):
        for j, r in enumerate(rs):
            out = section_solve(r, pr, E_t, J, S, alpha)
            if out is not None:
                Q[i, j] = out[1]   # q = p_theta^2
    return rs, prs, Q


if __name__ == "__main__":
    import time
    t0 = time.time()
    ALPHA, J, S = 0.3, 5.0, 1.0
    E_t = 0.9505
    rs, prs, Q = map_accessible(E_t, J, S, ALPHA,
                                 r_range=(4.2, 20.0), pr_range=(-0.30, 0.30))
    np.savez("section_map.npz", rs=rs, prs=prs, Q=Q, E=E_t)
    acc = np.sum(Q >= 0)
    print(f"mapa listo ({time.time()-t0:.0f}s): {acc} puntos accesibles "
          f"de {Q.size} ({100*acc/Q.size:.1f}%)")
    # frontera interior en p_r=0:
    mid = np.argmin(np.abs(prs))
    row = Q[mid]
    inside = np.where(row >= 0)[0]
    if len(inside):
        print(f"en p_r=0: region accesible r en [{rs[inside[0]]:.3f}, {rs[inside[-1]]:.3f}]")
        print(f"q maximo en p_r=0: {np.nanmax(row):.4f} en r={rs[np.nanargmax(row)]:.3f}")
