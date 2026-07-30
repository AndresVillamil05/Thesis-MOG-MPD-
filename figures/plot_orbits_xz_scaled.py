"""
Orbitas Tipo B en proyeccion x-z CON EL AGUJERO NEGRO A ESCALA:
  - aspecto igual (el horizonte es un circulo verdadero de radio r_h);
  - oclusion correcta: los tramos de la trayectoria con y<0 (detras del
    agujero) se dibujan bajo el disco y atenuados; los tramos con y>0
    (delante) se dibujan encima;
  - trayectoria coloreada por tiempo propio;
  - un panel por fila (las orbitas son "panqueques" anchos y delgados).
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

fig, axes = plt.subplots(len(R0S), 1, figsize=(9.5, 13.5))
lc_ref = None

for ax, r0 in zip(axes, R0S):
    y0 = initial_conditions_equatorial(r0=r0, E=0.94, J=J, S=S, alpha=ALPHA,
                                        sign_pr=-1)
    y0[2] = 1.5
    E0, Jz0, m20, S20 = conserved(y0, J, ALPHA, M, m)
    sol = integrate(y0, J, ALPHA, tau_max=TAU_MAX, rtol=1e-10, atol=1e-12)
    r = sol.y[1]; th = sol.y[2]; ph = sol.y[3]
    x = r*np.sin(th)*np.cos(ph)
    yy = r*np.sin(th)*np.sin(ph)
    z = r*np.cos(th)

    puntos = np.array([x, z]).T.reshape(-1, 1, 2)
    segs = np.concatenate([puntos[:-1], puntos[1:]], axis=1)
    y_mid = 0.5*(yy[:-1] + yy[1:])
    t_seg = sol.t[:-1]
    norm = plt.Normalize(0, sol.t[-1])

    detras = y_mid < 0
    lc_b = LineCollection(segs[detras], cmap=CMAP_TRAJ, norm=norm,
                          linewidths=0.4, alpha=0.35, zorder=2)
    lc_b.set_array(t_seg[detras])
    ax.add_collection(lc_b)

    ax.add_patch(plt.Circle((0, 0), r_h, color="black", zorder=4))

    lc_f = LineCollection(segs[~detras], cmap=CMAP_TRAJ, norm=norm,
                          linewidths=0.5, alpha=0.95, zorder=6)
    lc_f.set_array(t_seg[~detras])
    ax.add_collection(lc_f)
    lc_ref = lc_f

    ax.plot(x[0], z[0], "o", color="red", ms=5, zorder=8)

    x_lim = np.abs(x).max()*1.05
    z_lim = max(r_h*1.18, np.abs(z).max()*1.25)
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-z_lim, z_lim)
    ax.set_aspect("equal")
    ax.set_ylabel(r"$z/M$")
    ax.text(0.985, 0.92, rf"$r_0={r0}$,  $E={E0:.4f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))

axes[-1].set_xlabel(r"$x/M$")
cbar = fig.colorbar(lc_ref, ax=axes, orientation="vertical",
                    fraction=0.025, pad=0.02)
cbar.set_label(r"tiempo propio $\tau$", fontsize=10)

fig.suptitle("Órbitas de partícula con espín — proyección $x$–$z$ con el "
             "horizonte a escala\n"
             rf"($\alpha={ALPHA}$, $J={J}\,mM$, $S={S}\,mM$, $\theta_0=1.5$; "
             "tramos atenuados: detrás del agujero)", fontsize=12)
fig.savefig("orbits_typeB_xz_scaled.png", dpi=200,
            bbox_inches="tight")
fig.savefig("orbits_typeB_xz_scaled.pdf",
            bbox_inches="tight")
print("guardado orbits_typeB_xz_scaled")
