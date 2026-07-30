"""Grafica las secciones de Poincare desde poincare_data.npz."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10.5, "font.family": "serif",
                      "mathtext.fontset": "cm"})

d = np.load("poincare_data.npz", allow_pickle=True)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
colors = plt.cm.tab10.colors

for ax, key, title in [
        (axes[0], "regular", rf"(a) $E={float(d['regular_E']):.4f}$ (lejos de la silla)"),
        (axes[1], "caotico", rf"(b) $E={float(d['caotico_E']):.4f}$ (cerca de la silla, $V_{{silla}}=0.9506$)")]:
    pts_list = d[f"{key}_pts"]
    labels = d[f"{key}_labels"]
    for i, pts in enumerate(pts_list):
        pts = np.asarray(pts, dtype=float)
        if len(pts) == 0:
            continue
        r0, pth0, status, ncross = labels[i]
        ax.plot(pts[:, 0], pts[:, 1], '.', ms=2.2, color=colors[i % 10],
                label=rf"$r_0={r0}$, $p_{{\theta 0}}={pth0}$")
    ax.set_xlabel(r"$r/M$")
    ax.set_ylabel(r"$p_r/m$")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7.5, markerscale=3, framealpha=0.9)
    ax.grid(alpha=0.2)

fig.suptitle(r"Secciones de Poincaré ($\theta=\pi/2$, $\dot\theta>0$) — "
             r"Schwarzschild-MOG, $\alpha=0.3$, $J=5\,mM$, $S=1\,mM$",
             fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig_poincare.png", dpi=210)
fig.savefig("fig_poincare.pdf")
print("guardado fig_poincare")
