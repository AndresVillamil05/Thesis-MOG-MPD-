"""
Perfil del potencial efectivo en función del espín:
Variacion de V_eff+/- con r en el plano ecuatorial de Schwarzschild-MOG,
para distintos valores del spin S, con J=4mM fijo.
Un panel por valor de alpha .
Lineas gruesas = V_eff+, lineas finas = V_eff-, punteada vertical = horizonte.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from effective_potential import Veff, horizon_mog

plt.rcParams.update({"font.size": 11, "font.family": "serif",
                      "mathtext.fontset": "cm"})

J = 4.0
S_VALUES = [0.2, 0.4, 0.6, 0.8, 1.2]           # valores de espín considerados
COLORS = ['#6a0dad', '#1f6fd6', '#12a5b0', '#e0a800', '#d62718']
ALPHAS = [0.0, 0.3, 0.8]                        # paneles: GR, fiducial, extremo
R_MAX = 30.0

fig, axes = plt.subplots(3, 1, figsize=(6.4, 12.5), sharex=True)

for ax, alpha in zip(axes, ALPHAS):
    r_h = horizon_mog(alpha)
    r = np.linspace(r_h*1.001, R_MAX, 1600)
    for S, color in zip(S_VALUES, COLORS):
        Vp = Veff(r, np.pi/2, J, S, alpha, branch=+1)
        Vm = Veff(r, np.pi/2, J, S, alpha, branch=-1)
        ax.plot(r, Vp, color=color, lw=2.4, label=rf"${S}\,mM$")
        ax.plot(r, Vm, color=color, lw=0.9)
    ax.axvline(r_h, color='gray', ls='--', lw=1.4)
    etiqueta = "Schwarzschild" if alpha == 0.0 else "MOG"
    ax.set_title(rf"$\alpha={alpha}$ ({etiqueta}),  $J=4\,mM$", fontsize=11.5)
    ax.set_ylabel(r"$V_{eff}\,(m)$")
    ax.set_xlim(0, R_MAX)
    ax.set_ylim(0.86, 1.10)
    ax.grid(alpha=0.2)
    leg = ax.legend(title=r"$S$", fontsize=9, loc='center right',
                    framealpha=0.95)
    leg.get_title().set_fontsize(10)

axes[-1].set_xlabel(r"$r\,(M)$")
fig.suptitle(r"$V_{eff\pm}$ en el plano ecuatorial variando el spin $S$"
             "\n" r"(línea gruesa: $V_{eff+}$, fina: $V_{eff-}$, punteada: horizonte)",
             fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig("potential_vs_spin.png", dpi=200)
fig.savefig("potential_vs_spin.pdf")
print("guardado potential_vs_spin.png/.pdf")
