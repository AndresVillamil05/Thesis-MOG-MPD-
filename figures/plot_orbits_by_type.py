"""
Orbitas en los cuatro tipos de potencial (A, B, C, D) -- validacion
dinamica de la clasificacion: cada tipo debe producir el comportamiento
orbital que la topologia del potencial predice.
  A: orbita ligada regular (toro ordenado)
  B: orbita ligada con oscilacion vertical grande (posible caos)
  C: caida directa al agujero (sin region ligada)
  D: desviacion progresiva del plano ecuatorial y caida (silla en theta)
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
ALPHA = 0.3
r_h = horizon_mog(ALPHA)

CASES = [
    {"tipo": "A", "J": 5.0, "S": 0.3, "r0": 9.0, "E": 0.955, "tau": 4000.0,
     "desc": "ligada regular"},
    {"tipo": "B", "J": 5.0, "S": 1.0, "r0": 5.0, "E": 0.94, "tau": 4000.0,
     "desc": "ligada, oscilación vertical"},
    {"tipo": "C", "J": 4.0, "S": 0.9, "r0": 7.0, "E": 0.95, "tau": 4000.0,
     "desc": "caída al agujero"},
    {"tipo": "D", "J": 4.2, "S": 1.8, "r0": 4.0, "E": 0.865, "tau": 4000.0,
     "desc": "escape del plano y caída"},
]

fig, axes = plt.subplots(2, 2, figsize=(11, 10))

for ax, case in zip(axes.flat, CASES):
    J, S, r0, E, tau = case["J"], case["S"], case["r0"], case["E"], case["tau"]
    y0 = initial_conditions_equatorial(r0=r0, E=E, J=J, S=S, alpha=ALPHA,
                                        sign_pr=-1)
    y0[2] = 1.5  # theta0 fuera del plano, 
    E0, Jz0, m20, S20 = conserved(y0, J, ALPHA, M, m)

    sol = integrate(y0, J, ALPHA, tau_max=tau, rtol=1e-10, atol=1e-12)
    r = sol.y[1]; th = sol.y[2]; ph = sol.y[3]
    x = r*np.sin(th)*np.cos(ph)
    z = r*np.cos(th)

    fate = {0: "τ completo", 1: "cayó al horizonte" if len(sol.t_events[0]) else "escapó",
            -1: "integración detenida"}[sol.status]

    ax.plot(x, z, lw=0.2, color='#1a6b8a', alpha=0.9)
    ax.plot(x[0], z[0], 'o', color='red', ms=6, zorder=10)
    if sol.status == 1 and len(sol.t_events[0]):
        ax.plot(x[-1], z[-1], 'x', color='darkorange', ms=9, mew=2.5, zorder=10)
    circ = plt.Circle((0, 0), r_h, color='black', zorder=9)
    ax.add_patch(circ)
    lim = max(np.abs(x).max(), np.abs(z).max(), r_h*2)*1.08
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel(r"$x/M$"); ax.set_ylabel(r"$z/M$")
    ax.set_title(rf"Tipo {case['tipo']}: $J={J}$, $S={S}$, $r_0={r0}$, "
                 rf"$E={E0:.4f}$" + f"\n{case['desc']} — {fate} "
                 rf"($\tau_f={sol.t[-1]:.0f}$)", fontsize=9.5)

fig.suptitle(rf"Órbitas en los cuatro tipos de potencial — Schwarzschild-MOG, "
             rf"$\alpha={ALPHA}$, $\theta_0=1.5$", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("orbits_by_type.png", dpi=200)
fig.savefig("orbits_by_type.pdf")
print("guardado orbits_by_type")
