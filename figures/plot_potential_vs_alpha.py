"""
Comparación del potencial efectivo MOG frente a Schwarzschild:
Comparacion de V_eff+/- para Schwarzschild-MOG (lineas solidas) contra
Schwarzschild puro alpha=0 (lineas discontinuas), para J=4mM, S=1mM,
mostrando 3 valores crecientes de alpha (el parámetro de desviación gravitacional.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from effective_potential import Veff, horizon_mog

plt.rcParams.update({"font.size": 11})

J, S = 4.0, 1.0
ALPHAS = [0.05, 0.3, 0.8]
R_MAX = 15.0

fig, axes = plt.subplots(3, 1, figsize=(6, 11), sharex=True)

r_h0 = horizon_mog(0.0)
r_schw = np.linspace(r_h0 * 1.02, R_MAX, 600)
Vp_schw = Veff(r_schw, np.pi / 2, J, S, 0.0, branch=+1)
Vm_schw = Veff(r_schw, np.pi / 2, J, S, 0.0, branch=-1)

for ax, alpha in zip(axes, ALPHAS):
    r_h = horizon_mog(alpha)
    r = np.linspace(r_h * 1.02, R_MAX, 600)
    Vp = Veff(r, np.pi / 2, J, S, alpha, branch=+1)
    Vm = Veff(r, np.pi / 2, J, S, alpha, branch=-1)

    ax.plot(r, Vp, color="tab:blue", lw=1.8, label=r"$V_{eff+}$ (MOG)")
    ax.plot(r, Vm, color="tab:red", lw=1.8, label=r"$V_{eff-}$ (MOG)")
    ax.plot(r_schw, Vp_schw, color="tab:blue", lw=1.4, ls="--",
             label=r"$V_{eff+}$ (Schwarzschild)")
    ax.plot(r_schw, Vm_schw, color="tab:red", lw=1.4, ls="--",
             label=r"$V_{eff-}$ (Schwarzschild)")

    ax.set_title(rf"$\alpha = {alpha}$,  $J=4\,mM$,  $S=1\,mM$", fontsize=11)
    ax.set_ylabel(r"$V_{eff}(m)$")
    ax.set_ylim(0.4, 1.15)
    ax.set_xlim(0, R_MAX)
    ax.grid(alpha=0.25)

axes[0].legend(loc="lower right", fontsize=8.5, framealpha=0.9)
axes[-1].set_xlabel(r"$r(M)$")
fig.suptitle("Potencial efectivo: Schwarzschild-MOG vs. Schwarzschild\n"
             r"", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("potential_mog_vs_schwarzschild.png", dpi=200)
fig.savefig("potential_mog_vs_schwarzschild.pdf")
print("guardado potential_mog_vs_schwarzschild.png/.pdf")
