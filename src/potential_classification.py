"""
Busqueda y clasificacion de puntos criticos de V_eff(r,theta) para
Schwarzschild-MOG, replicando la clasificacion tipo A/B/C/D de
Suzuki-Maeda / Giri et al. (2022).
"""
import numpy as np
from scipy.optimize import fsolve
from effective_potential import Veff, horizon_mog, isco_schw_mog


def grad_hess(func, r, theta, h=1e-4, **kw):
    """Gradiente y Hessiana 2D de func(r,theta) por diferencias finitas centradas."""
    f00 = func(r, theta, **kw)
    fr1 = func(r+h, theta, **kw); fr2 = func(r-h, theta, **kw)
    ft1 = func(r, theta+h, **kw); ft2 = func(r, theta-h, **kw)
    Vr = (fr1-fr2)/(2*h)
    Vt = (ft1-ft2)/(2*h)
    Vrr = (fr1 - 2*f00 + fr2)/h**2
    Vtt = (ft1 - 2*f00 + ft2)/h**2
    frt1 = func(r+h, theta+h, **kw); frt2 = func(r+h, theta-h, **kw)
    frt3 = func(r-h, theta+h, **kw); frt4 = func(r-h, theta-h, **kw)
    Vrt = (frt1-frt2-frt3+frt4)/(4*h**2)
    return np.array([Vr, Vt]), np.array([[Vrr, Vrt], [Vrt, Vtt]])


def find_critical_points(J, S, alpha, r_h, r_max=15.0, branch=-1, n_seed=14):
    """Escanea una grilla de puntos semilla y converge a puntos criticos via fsolve.
    Devuelve lista de dicts {r, theta, V, kind} con kind en {'min','max','saddle'}.
    """
    def Vfun(r, theta):
        return Veff(np.atleast_1d(r), np.atleast_1d(theta), J, S, alpha, branch=branch)[0]

    def system(x):
        r, theta = x
        if r <= r_h*1.001 or r > 40 or theta <= 0.02 or theta >= np.pi-0.02:
            return [1e3*(r), 1e3*(theta)]
        g, _ = grad_hess(Vfun, r, theta)
        return g

    found = []
    r_seeds = np.linspace(r_h*1.05, r_max, n_seed)
    th_seeds = np.linspace(0.15*np.pi, 0.85*np.pi, 9)
    for rs in r_seeds:
        for ts in th_seeds:
            sol, info, ier, msg = fsolve(system, [rs, ts], full_output=True, xtol=1e-10)
            if ier != 1:
                continue
            r_c, th_c = sol
            if not (r_h*1.001 < r_c < r_max and 0.02 < th_c < np.pi-0.02):
                continue
            g, H = grad_hess(Vfun, r_c, th_c)
            if np.max(np.abs(g)) > 1e-6:
                continue
            # deduplicate
            dup = False
            for f in found:
                if abs(f['r']-r_c) < 1e-3 and abs(f['theta']-th_c) < 1e-3:
                    dup = True
                    break
            if dup:
                continue
            det = np.linalg.det(H)
            if det < -1e-10:
                kind = 'saddle'
            elif det > 1e-10 and H[0, 0] > 0:
                kind = 'min'
            elif det > 1e-10 and H[0, 0] < 0:
                kind = 'max'
            else:
                kind = 'degenerate'
            found.append({'r': r_c, 'theta': th_c, 'V': Vfun(r_c, th_c), 'kind': kind})
    return found


def classify_potential(J, S, alpha, branch=-1, verbose=False):
    r_h = horizon_mog(alpha)
    pts = find_critical_points(J, S, alpha, r_h, branch=branch)
    n_min = sum(1 for p in pts if p['kind'] == 'min')
    n_saddle = sum(1 for p in pts if p['kind'] == 'saddle')
    eq_saddle = any(p['kind'] == 'saddle' and abs(p['theta']-np.pi/2) < 0.05 for p in pts)
    off_saddle = any(p['kind'] == 'saddle' and abs(p['theta']-np.pi/2) > 0.05 for p in pts)
    eq_min = any(p['kind'] == 'min' and abs(p['theta']-np.pi/2) < 0.05 for p in pts)

    if verbose:
        for p in pts:
            print(f"  r={p['r']:.4f} theta={p['theta']:.4f} ({p['theta']*180/np.pi:.1f} deg) "
                  f"V={p['V']:.5f}  {p['kind']}")

    if eq_min and eq_saddle and not off_saddle:
        tipo = 'A'
    elif eq_min and off_saddle:
        tipo = 'B'
    elif (not eq_min) and (not off_saddle) and eq_saddle:
        tipo = 'D'
    elif n_min == 0 and n_saddle == 0:
        tipo = 'C'
    else:
        tipo = f'?({n_min}min,{n_saddle}sad,eqS={eq_saddle},offS={off_saddle})'
    return tipo, pts


if __name__ == "__main__":
    alpha = 0.3
    tests = [
        (4.4, 0.5),
        (4.4, 1.0),
        (4.4, 1.5),
        (3.9, 0.6),
        (3.9, 1.4),
    ]
    for J, S in tests:
        tipo, pts = classify_potential(J, S, alpha, verbose=False)
        print(f"alpha={alpha} J={J} S={S}  ->  Tipo {tipo}   (#puntos criticos={len(pts)})")
