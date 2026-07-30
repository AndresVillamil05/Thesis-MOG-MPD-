"""Secciones de Poincare en la configuracion de pozo somero J=4.75."""
import numpy as np, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from accessible_region import section_solve
from poincare_section import poincare_section

ALPHA, J, S = 0.3, 4.75, 1.0
TAU = 50000.0

RUNS = {
    "media":       {"E": 0.9260, "r_fan": [5.55, 5.8, 6.1, 6.5, 7.0, 7.5, 8.0, 8.5, 8.8]},
    "cerca_silla": {"E": 0.9278, "r_fan": [4.85, 5.0, 5.2, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 9.6]},
}

for tag, cfg in RUNS.items():
    E_t = cfg["E"]
    all_pts, labels = [], []
    t0 = time.time()
    for r0 in cfg["r_fan"]:
        out = section_solve(r0, 0.0, E_t, J, S, ALPHA)
        if out is None or out[1] < 0:
            print(f"[{tag}] r0={r0}: inaccesible"); continue
        p_t, q, p_ph = out
        y0 = np.array([0.0, r0, np.pi/2, 0.0, p_t, 0.0, np.sqrt(q), p_ph])
        pts, status = poincare_section(y0, J, ALPHA, tau_max=TAU, max_crossings=4000)
        all_pts.append(pts); labels.append((r0, status, len(pts)))
        destino = {0:'ok', 1:'cayo/escapo', -1:'detenida'}[status]
        print(f"[{tag}] r0={r0}: {len(pts)} cruces, {destino} ({time.time()-t0:.0f}s)")
    np.savez(f"poincare_J475_{tag}.npz", E=E_t,
             pts=np.array(all_pts, dtype=object),
             labels=np.array(labels, dtype=object), allow_pickle=True)
    print(f"[{tag}] guardado\n")
