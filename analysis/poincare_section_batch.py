"""
Computa las secciones de Poincare completas (multi-semilla) y las guarda
en .npz para graficar aparte. Dos paneles:
  (a) E lejos de la silla -> esperamos toros KAM (curvas cerradas)
  (b) E cerca de la silla -> esperamos dispersion caotica
Parametros del potencial Tipo B: alpha=0.3, J=5, S=1
(minimo V=0.9297 en r=8.82; silla V=0.9506 en r=4.37, fuera del plano)
"""
import numpy as np
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from poincare_section import build_ic, poincare_section
from integrator import conserved

M = m = 1.0
ALPHA, J, S = 0.3, 5.0, 1.0
TAU = 40000.0

PANELS = {
    "regular": {"E": 0.9370, "seeds": [(8.0, 0.10), (8.8, 0.15), (9.5, 0.20),
                                        (10.2, 0.10), (11.0, 0.05), (8.4, 0.25)]},
    "caotico": {"E": 0.9490, "seeds": [(6.0, 0.10), (7.0, 0.30), (8.0, 0.50),
                                        (9.0, 0.50), (12.0, 0.30), (15.0, 0.15)]},
}

results = {}
t0 = time.time()
for name, cfg in PANELS.items():
    E_t = cfg["E"]
    all_pts = []
    labels = []
    for (r0, pth0) in cfg["seeds"]:
        y0 = build_ic(r0=r0, pth0=pth0, E_target=E_t, J=J, S=S, alpha=ALPHA)
        if y0 is None:
            print(f"[{name}] semilla r0={r0}, pth0={pth0}: SIN solucion, se omite")
            continue
        E0, Jz0, m20, S20 = conserved(y0, J, ALPHA, M, m)
        assert abs(E0 - E_t) < 1e-8, f"E no coincide: {E0}"
        pts, status = poincare_section(y0, J, ALPHA, tau_max=TAU)
        all_pts.append(pts)
        labels.append((r0, pth0, status, len(pts)))
        print(f"[{name}] r0={r0}, pth0={pth0}: {len(pts)} cruces, status={status} "
              f"({time.time()-t0:.0f}s)")
    results[name] = {"E": E_t, "pts": all_pts, "labels": labels}

np.savez("poincare_data.npz",
         regular_E=results["regular"]["E"],
         caotico_E=results["caotico"]["E"],
         regular_pts=np.array(results["regular"]["pts"], dtype=object),
         caotico_pts=np.array(results["caotico"]["pts"], dtype=object),
         regular_labels=np.array(results["regular"]["labels"], dtype=object),
         caotico_labels=np.array(results["caotico"]["labels"], dtype=object),
         allow_pickle=True)
print(f"\nguardado poincare_data.npz ({time.time()-t0:.0f}s total)")
