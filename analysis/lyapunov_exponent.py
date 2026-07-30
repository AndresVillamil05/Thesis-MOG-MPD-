"""
Exponente maximo de Lyapunov (metodo de Benettin) y FLI para las
ecuaciones MPD en Schwarzschild-MOG.

Evoluciona el vector de desviacion w junto al estado y:
    dw/dtau = J(y) . w   aproximado por diferencias finitas direccionales
Renormaliza w cada delta_tau y acumula log de los factores de estiramiento:
    lambda(tau) = (1/tau) * sum log s_k      (exponente de Lyapunov)
    FLI(tau)    = log10 ||w|| acumulado      (crece ~log tau si regular,
                                              ~lineal en tau si caotico)
"""
import numpy as np
from scipy.integrate import solve_ivp
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from integrator import rhs, horizon_mog


def rhs_aug(tau, Y, J, alpha, M, m, delta=1e-7):
    y = Y[:8]
    w = Y[8:]
    f0 = np.array(rhs(tau, y, J, alpha, M, m))
    nw = np.linalg.norm(w)
    if nw == 0:
        return np.concatenate([f0, np.zeros(8)])
    u = w / nw
    f1 = np.array(rhs(tau, y + delta*u, J, alpha, M, m))
    dw = (f1 - f0) / delta * nw
    return np.concatenate([f0, dw])


def lyapunov(y0, J, alpha, tau_max, M=1.0, m=1.0, chunk=50.0,
             rtol=1e-8, atol=1e-10, seed=0):
    """Devuelve (taus, lambdas, flis): historia de lambda(tau) y FLI(tau)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=8)
    w /= np.linalg.norm(w)
    r_h = horizon_mog(alpha, M)

    Y = np.concatenate([y0, w])
    tau = 0.0
    log_sum = 0.0
    taus, lambdas, flis = [], [], []
    n_chunks = int(tau_max / chunk)

    def horizon_event(t_, Y_, *args):
        return Y_[1] - r_h*1.15
    horizon_event.terminal = True
    horizon_event.direction = -1

    def escape_event(t_, Y_, *args):
        return Y_[1] - 80.0
    escape_event.terminal = True
    escape_event.direction = +1

    for k in range(n_chunks):
        if not np.all(np.isfinite(Y)):
            break
        sol = solve_ivp(rhs_aug, [tau, tau + chunk], Y,
                        args=(J, alpha, M, m), method='DOP853',
                        rtol=rtol, atol=atol,
                        events=[horizon_event, escape_event])
        if sol.status != 0 or sol.y[1, -1] < r_h*1.16 or sol.y[1, -1] > 79:
            break
        Y = sol.y[:, -1]
        tau = sol.t[-1]
        w = Y[8:]
        s = np.linalg.norm(w)
        log_sum += np.log(s)
        Y[8:] = w / s          # renormalizar
        taus.append(tau)
        lambdas.append(log_sum / tau)
        flis.append(log_sum / np.log(10))
    return np.array(taus), np.array(lambdas), np.array(flis)


if __name__ == "__main__":
    # VALIDACION: geodesica circular Schwarzschild (regular) -> lambda ~ 1/tau
    import time
    t0 = time.time()
    r0 = 10.0
    L = r0/np.sqrt(r0 - 3.0)
    E = (r0 - 2.0)/np.sqrt(r0*(r0 - 3.0))
    y0 = np.array([0.0, r0, np.pi/2, 0.0, -E, 0.0, 0.0, L])
    taus, lams, flis = lyapunov(y0, J=L, alpha=0.0, tau_max=5000.0)
    print(f"validacion geodesica circular ({time.time()-t0:.0f}s):")
    for i in [len(taus)//10, len(taus)//3, len(taus)-1]:
        print(f"  tau={taus[i]:7.0f}  lambda={lams[i]:.3e}  "
              f"lambda*tau={lams[i]*taus[i]:.2f}  FLI={flis[i]:.2f}")
    print("  (regular: lambda*tau debe mantenerse ~constante, lambda->0)")
