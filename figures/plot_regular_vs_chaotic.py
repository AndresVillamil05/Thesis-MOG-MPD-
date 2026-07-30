"""
La orbita que faltaba: comparacion REGULAR vs CAOTICA.

Panel izquierdo: orbita regular (energia en el pozo, bajo la silla) -> tejido
   de toro ordenado, sobrevive toda la integracion.
Panel derecho: orbita CAOTICA transitoria (energia sobre la silla) -> rebota
   erraticamente en el pozo y termina cayendo al agujero en tau finito.
   Es la firma dinamica del caos detectada via Lyapunov (lambda ~ 0.029).

Proyeccion x-z, coloreada por tiempo propio.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from initial_conditions import build_ic_general
from integrator import integrate, conserved, horizon_mog

plt.rcParams.update({"font.size": 10.5, "font.family": "serif",
                      "mathtext.fontset": "cm"})

M = m = 1.0
CMAP = "plasma"

# ambas en Schwarzschild (J=3.959, S=1) — sistema de referencia en Schwarzschild
CASOS = [
    {"tag": "regular",  "alpha": 0.0, "J": 3.959, "S": 1.0,
     "r0": 6.2, "E": 0.9330, "tau": 4000.0,
     "titulo": "Órbita regular\n" r"($E=0.933<V_{\rm silla}$)"},
    {"tag": "caotica",  "alpha": 0.0, "J": 3.959, "S": 1.0,
     "r0": 4.0, "E": 0.9530, "tau": 4000.0,
     "titulo": "Órbita caótica transitoria\n" r"($E=0.953>V_{\rm silla}$)"},
]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
lc_ref = None

for ax, caso in zip(axes, CASOS):
    y0 = build_ic_general(caso["r0"], 1.5, caso["E"],
                          J=caso["J"], S=caso["S"], alpha=caso["alpha"])
    E0, Jz0, m20, S20 = conserved(y0, caso["J"], caso["alpha"], M, m)
    r_h = horizon_mog(caso["alpha"])
    sol = integrate(y0, caso["J"], caso["alpha"], tau_max=caso["tau"],
                    rtol=1e-10, atol=1e-12)
    r = sol.y[1]; th = sol.y[2]; ph = sol.y[3]
    x = r*np.sin(th)*np.cos(ph); z = r*np.cos(th)
    cayo = sol.t[-1] < caso["tau"]*0.99

    puntos = np.array([x, z]).T.reshape(-1, 1, 2)
    segs = np.concatenate([puntos[:-1], puntos[1:]], axis=1)
    lc = LineCollection(segs, cmap=CMAP, norm=plt.Normalize(0, sol.t[-1]),
                        linewidths=0.6, alpha=0.9)
    lc.set_array(sol.t[:-1])
    ax.add_collection(lc)
    lc_ref = lc

    ax.plot(x[0], z[0], "o", color="lime", ms=7, zorder=10,
            markeredgecolor="black", markeredgewidth=0.6)
    ax.add_patch(plt.Circle((0, 0), r_h, color="black", zorder=9))

    destino = (rf"cae al horizonte en $\tau={sol.t[-1]:.0f}\,M$" if cayo
               else rf"sobrevive $\tau={caso['tau']:.0f}\,M$")
    x_lim = np.abs(x).max()*1.08
    z_lim = max(np.abs(z).max()*1.3, r_h*1.2)
    ax.set_xlim(-x_lim, x_lim); ax.set_ylim(-z_lim, z_lim)
    ax.set_xlabel(r"$x/M$"); ax.set_ylabel(r"$z/M$")
    ax.set_title(caso["titulo"] + "\n" + destino, fontsize=10.5)

cbar = fig.colorbar(lc_ref, ax=axes, orientation="vertical",
                    fraction=0.025, pad=0.02)
cbar.set_label(r"tiempo propio $\tau$", fontsize=10)

fig.suptitle(r"Órbita regular vs. caótica en Schwarzschild ($J=3.959\,mM$, "
             r"$S=1\,mM$, $\theta_0=1.5$)", fontsize=12.5, y=1.02)
fig.savefig("regular_vs_chaotic.png", dpi=200,
            bbox_inches="tight")
fig.savefig("regular_vs_chaotic.pdf", bbox_inches="tight")
print("guardado regular_vs_chaotic")
