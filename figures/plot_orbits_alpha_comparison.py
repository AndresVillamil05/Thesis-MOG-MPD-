"""
La figura central de la tesis: misma particula (J, S, r0, theta0, E
relativo identicos) orbitando con alpha = 0 (Schwarzschild puro),
0.15 y 0.3 (MOG). Todo cambio entre paneles es atribuible unicamente
al parametro de gravedad modificada.
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
J, S = 5.0, 1.0
R0 = 6.0
TAU = 6000.0
ALPHAS = [0.0, 0.15, 0.3]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, alpha in zip(axes, ALPHAS):
    r_h = horizon_mog(alpha)
    y0 = initial_conditions_equatorial(r0=R0, E=0.94, J=J, S=S, alpha=alpha,
                                        sign_pr=-1)
    y0[2] = 1.5
    E0, Jz0, m20, S20 = conserved(y0, J, alpha, M, m)

    sol = integrate(y0, J, alpha, tau_max=TAU, rtol=1e-10, atol=1e-12)
    r = sol.y[1]; th = sol.y[2]; ph = sol.y[3]
    x = r*np.sin(th)*np.cos(ph)
    z = r*np.cos(th)

    fate = {0: "ligada", 1: "cayó/escapó", -1: "detenida"}[sol.status]

    ax.plot(x, z, lw=0.18, color='#1a6b8a', alpha=0.9)
    ax.plot(x[0], z[0], 'o', color='red', ms=6, zorder=10)
    circ = plt.Circle((0, 0), r_h, color='black', zorder=9)
    ax.add_patch(circ)
    lim = max(np.abs(x).max(), np.abs(z).max(), r_h*1.5)*1.06
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel(r"$x/M$"); ax.set_ylabel(r"$z/M$")
    label = "Schwarzschild" if alpha == 0 else "MOG"
    ax.set_title(rf"$\alpha={alpha}$ ({label}): $E={E0:.4f}$, $r_h={r_h:.2f}$"
                 + f"\n{fate}, " + rf"$\tau_f={sol.t[-1]:.0f}$, "
                 + rf"$r\in[{r.min():.1f},{r.max():.1f}]$", fontsize=9.5)

fig.suptitle(rf"Efecto del parámetro MOG sobre la órbita — $J={J}\,mM$, "
             rf"$S={S}\,mM$, $r_0={R0}$, $\theta_0=1.5$ idénticos", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("orbits_vs_alpha.png", dpi=200)
fig.savefig("orbits_vs_alpha.pdf")
print("guardado orbits_vs_alpha")
