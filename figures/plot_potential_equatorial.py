"""
Perfil del potencial efectivo en función de alpha:
Variacion de V_eff+/- con r en el plano ecuatorial, para distintos
valores de alpha (parametro MOG), con J=4mM, S=0.5mM fijos.
Lineas gruesas = V_eff+, lineas finas = V_eff-.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from effective_potential import Veff, horizon_mog

plt.rcParams.update({"font.size": 11})

J, S = 4.0, 0.5
ALPHAS = [0.0001, 0.05, 0.1, 0.3, 0.5, 0.8]
COLORS = plt.cm.plasma(np.linspace(0.05, 0.85, len(ALPHAS)))
R_MAX = 30.0

fig, ax = plt.subplots(figsize=(7.5, 6))

for alpha, color in zip(ALPHAS, COLORS):
    r_h = horizon_mog(alpha)
    r = np.linspace(r_h * 1.02, R_MAX, 800)
    Vp = Veff(r, np.pi / 2, J, S, alpha, branch=+1)
    Vm = Veff(r, np.pi / 2, J, S, alpha, branch=-1)

    ax.plot(r, Vp, color=color, lw=2.2, label=rf"$\alpha={alpha}$")
    ax.plot(r, Vm, color=color, lw=0.9)

ax.set_xlim(0, R_MAX)
ax.set_ylim(0.75, 1.05)
ax.set_xlabel(r"$r(M)$")
ax.set_ylabel(r"$V_{eff}(m)$")
ax.set_title(r"$V_{eff\pm}$ vs $r$ para distintos $\alpha$"
             "\n" r"$J=4\,mM$, $S=0.5\,mM$   "
             r"(grueso: $V_{eff+}$, fino: $V_{eff-}$)", fontsize=11)
ax.legend(title=r"$\alpha$", fontsize=9, ncol=2)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("potential_vs_alpha.png", dpi=200)
fig.savefig("potential_vs_alpha.pdf")
print("guardado potential_vs_alpha.png/.pdf")
