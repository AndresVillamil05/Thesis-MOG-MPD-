"""
Orbitas Tipo B en proyeccion x-z -- version mejorada para legibilidad:
  - tau mas corto (2000 en vez de 6000): las hebras individuales se ven
    en lugar de una banda solida;
  - trayectoria coloreada por tiempo propio (colormap viridis);
  - lineas finas con transparencia leve.
Mismos parametros y condiciones iniciales que la figura original.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from integrator import (initial_conditions_equatorial, integrate, conserved,
                         horizon_mog)

plt.rcParams.update({"font.size": 10, "font.family": "serif",
                      "mathtext.fontset": "cm"})

M = m = 1.0
ALPHA = 0.3
J, S = 5.0, 1.0
r_h = horizon_mog(ALPHA)
R0S = [8.8, 7.0, 6.0, 5.0, 4.6]
TAU_MAX = 2000.0
CMAP_TRAJ = "viridis"

fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5))
axes = axes.flat
lc_ref = None

for i, r0 in enumerate(R0S):
    y0 = initial_conditions_equatorial(r0=r0, E=0.94, J=J, S=S, alpha=ALPHA,
                                        sign_pr=-1)
    y0[2] = 1.5
    E0, Jz0, m20, S20 = conserved(y0, J, ALPHA, M, m)
    sol = integrate(y0, J, ALPHA, tau_max=TAU_MAX, rtol=1e-10, atol=1e-12)
    r = sol.y[1]; th = sol.y[2]; ph = sol.y[3]
    x = r*np.sin(th)*np.cos(ph)
    z = r*np.cos(th)

    ax = axes[i]
    puntos = np.array([x, z]).T.reshape(-1, 1, 2)
    segs = np.concatenate([puntos[:-1], puntos[1:]], axis=1)
    lc = LineCollection(segs, cmap=CMAP_TRAJ,
                        norm=plt.Normalize(0, sol.t[-1]),
                        linewidths=0.45, alpha=0.85)
    lc.set_array(sol.t[:-1])
    ax.add_collection(lc)
    lc_ref = lc

    ax.plot(x[0], z[0], "o", color="red", ms=6, zorder=10)
    ax.plot(0, 0, "o", color="black", ms=7, zorder=9)   # posicion del BH

    x_lim = np.abs(x).max()*1.06
    z_lim = max(np.abs(z).max()*1.35, 0.9)
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-z_lim, z_lim)
    ax.set_xlabel(r"$x/M$"); ax.set_ylabel(r"$z/M$")
    ax.set_title(rf"$r_0={r0}$,  $E={E0:.4f}$", fontsize=10)

ax = axes[len(R0S)]
ax.axis("off")
ax.text(0.02, 0.95,
        rf"$\alpha={ALPHA}$,  $J={J}\,mM$,  $S={S}\,mM$" + "\n"
        rf"$\theta_0=1.5$,  $\tau\leq{TAU_MAX:g}$" + "\n\n"
        "Proyección $x$–$z$.\n"
        "Punto rojo: posición inicial.\n"
        "Punto negro: posición del agujero\n"
        "negro (tamaño no a escala).",
        transform=ax.transAxes, fontsize=10, va="top", family="serif")
cbar = fig.colorbar(lc_ref, ax=ax, orientation="horizontal",
                    fraction=0.08, pad=0.04, shrink=0.85)
cbar.set_label(r"tiempo propio $\tau$", fontsize=10)

fig.suptitle("Órbitas de partícula con espín — proyección $x$–$z$, "
             "potencial Tipo B", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("orbits_typeB_xz.png", dpi=200)
fig.savefig("orbits_typeB_xz.pdf")
print("guardado orbits_typeB_xz")
