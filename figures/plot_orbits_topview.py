"""
Vista superior (proyeccion x-y) de las orbitas de la particula con spin:
la roseta de precesion del periastro, invisible en la proyeccion x-z.

Figura 1: los 5 radios del potencial Tipo B (J=5, S=1, alpha=0.3)
Figura 2: comparacion en alpha con condiciones iniciales identicas
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from integrator import (initial_conditions_equatorial, integrate, conserved,
                         horizon_mog)

plt.rcParams.update({"font.size": 10, "font.family": "serif",
                      "mathtext.fontset": "cm"})

M = m = 1.0

# ================= FIGURA 1: Tipo B, 5 radios, vista superior ==============
ALPHA = 0.3
J, S = 5.0, 1.0
r_h = horizon_mog(ALPHA)
R0S = [8.8, 7.0, 6.0, 5.0, 4.6]
TAU = 2000.0

fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
axes = axes.flat

for i, r0 in enumerate(R0S):
    y0 = initial_conditions_equatorial(r0=r0, E=0.94, J=J, S=S, alpha=ALPHA,
                                        sign_pr=-1)
    y0[2] = 1.5
    E0, Jz0, m20, S20 = conserved(y0, J, ALPHA, M, m)
    sol = integrate(y0, J, ALPHA, tau_max=TAU, rtol=1e-10, atol=1e-12)
    r = sol.y[1]; th = sol.y[2]; ph = sol.y[3]
    x = r*np.sin(th)*np.cos(ph)
    yy = r*np.sin(th)*np.sin(ph)

    ax = axes[i]
    ax.plot(x, yy, lw=0.45, color='#1a6b8a', alpha=0.9)
    ax.plot(x[0], yy[0], 'o', color='red', ms=5, zorder=10)
    ax.add_patch(plt.Circle((0, 0), r_h, color='black', zorder=9))
    ax.set_xlabel(r"$x/M$"); ax.set_ylabel(r"$y/M$")
    ax.set_title(rf"$r_0={r0}$,  $E={E0:.4f}$", fontsize=10)
    lim = max(np.abs(x).max(), np.abs(yy).max())*1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')

ax = axes[len(R0S)]
ax.axis('off')
ax.text(0.02, 0.9,
        rf"$\alpha={ALPHA}$, $J={J}\,mM$, $S={S}\,mM$" + "\n"
        rf"$\theta_0=1.5$,  $\tau\leq{TAU:g}$" + "\n\n"
        "Vista superior (proyección $x$–$y$):\n"
        "cada 'pétalo' es una oscilación radial;\n"
        "el avance angular entre pétalos\n"
        "es la precesión del periastro.",
        transform=ax.transAxes, fontsize=10, va='top', family='serif')

fig.suptitle(r"Órbitas de partícula con spin — vista superior, potencial Tipo B",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig_orbitas_tipoB_topview.png", dpi=200)
fig.savefig("fig_orbitas_tipoB_topview.pdf")
print("guardado fig_orbitas_tipoB_topview")

# ============ FIGURA 2: comparacion en alpha, vista superior ================
R0 = 6.0
TAU2 = 2500.0
ALPHAS = [0.0, 0.15, 0.3]

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
for ax, alpha in zip(axes2, ALPHAS):
    rh = horizon_mog(alpha)
    y0 = initial_conditions_equatorial(r0=R0, E=0.94, J=J, S=S, alpha=alpha,
                                        sign_pr=-1)
    y0[2] = 1.5
    E0, _, _, _ = conserved(y0, J, alpha, M, m)
    sol = integrate(y0, J, alpha, tau_max=TAU2, rtol=1e-10, atol=1e-12)
    r = sol.y[1]; th = sol.y[2]; ph = sol.y[3]
    x = r*np.sin(th)*np.cos(ph)
    yy = r*np.sin(th)*np.sin(ph)

    ax.plot(x, yy, lw=0.45, color='#1a6b8a', alpha=0.9)
    ax.plot(x[0], yy[0], 'o', color='red', ms=6, zorder=10)
    ax.add_patch(plt.Circle((0, 0), rh, color='black', zorder=9))
    label = "Schwarzschild" if alpha == 0 else "MOG"
    fate = {0: "ligada", 1: "escapa", -1: "detenida"}[sol.status]
    ax.set_title(rf"$\alpha={alpha}$ ({label}): $E={E0:.4f}$, {fate}",
                 fontsize=10.5)
    lim = max(np.abs(x).max(), np.abs(yy).max(), rh*2)*1.06
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel(r"$x/M$"); ax.set_ylabel(r"$y/M$")

fig2.suptitle(rf"Vista superior variando $\alpha$ — $J={J}\,mM$, $S={S}\,mM$, "
              rf"$r_0={R0}$, $\theta_0=1.5$ idénticos", fontsize=12.5)
fig2.tight_layout(rect=[0, 0, 1, 0.92])
fig2.savefig("orbits_vs_alpha_topview.png", dpi=200)
fig2.savefig("orbits_vs_alpha_topview.pdf")
print("guardado orbits_vs_alpha_topview")
