"""Condiciones iniciales en theta0 arbitrario con p_r=0, resolviendo
(p_t, p_theta, p_phi) de las ligaduras (E, m^2, S^2)."""
import numpy as np
from scipy.optimize import fsolve
from integrator import f_mog, fprime_mog

def constraints_general(u, r, th, J, S, alpha, M=1.0, m=1.0, pr=0.0):
    p_t, p_th, p_ph = u
    fr = f_mog(r, alpha, M)
    frp = fprime_mog(r, alpha, M)
    s, c = np.sin(th), np.cos(th)
    Str = -(p_th**2*s**2 + p_ph**2 - J*p_ph*s**2)/(p_t*r*s**2)
    E = -p_t - 0.5*frp*Str
    m2 = p_t**2/fr - fr*pr**2 - p_th**2/r**2 - p_ph**2/(r**2*s**2)
    # S^2 general (derivada y verificada simbolicamente en la sesion MPD)
    S2 = (J**2*c**2 + p_th**2/fr + (J*s**2 - p_ph)**2/(fr*s**2)
          - (J*p_ph*c + pr*p_th*r*s)**2*fr/(p_t**2*r**2*s**2)
          - (-J*p_ph*s**2 + p_ph**2 + p_th**2*s**2)**2/(p_t**2*r**2*s**4)
          - (J*pr*r + J*p_th*c/s - p_ph*pr*r/s**2)**2*fr*s**2/(p_t**2*r**2))
    return E, m2, S2

def build_ic_general(r0, th0, E_t, J, S, alpha, M=1.0, m=1.0, pr=0.0):
    def eqs(u):
        E, m2, S2 = constraints_general(u, r0, th0, J, S, alpha, M, m, pr)
        return [E - E_t, m2 - m**2, S2 - S**2]
    for pth_g in [0.3, 0.8, 0.05, 1.5, -0.3]:
        for pph_g in [J - S, J - S/2, J - 1.5*S]:
            sol, info, ier, msg = fsolve(eqs, [-E_t, pth_g, pph_g],
                                          full_output=True, xtol=1e-13)
            if ier != 1:
                continue
            p_t, p_th, p_ph = sol
            if p_t >= 0 or max(abs(x) for x in eqs(sol)) > 1e-9:
                continue
            return np.array([0.0, r0, th0, 0.0, p_t, pr, p_th, p_ph])
    return None

if __name__ == "__main__":
    from integrator import conserved
    # condición inicial de referencia: alpha=0, J=3.959, S=1, theta0=1.5
    for r0, E in [(4.0, 0.953), (3.5, 0.961), (5.0, 0.942)]:
        y0 = build_ic_general(r0, 1.5, E, J=3.959, S=1.0, alpha=0.0)
        if y0 is None:
            print(f"r0={r0} E={E}: sin solucion")
            continue
        Ec, Jz, m2, S2 = conserved(y0, 3.959, 0.0, 1.0, 1.0)
        print(f"r0={r0} E={E}: OK  ->  E={Ec:.6f} Jz={Jz:.6f} m2={m2:.6f} S2={S2:.6f}")
