"""
Mapas de contorno del potencial efectivo V_eff(rho, z)
complemento en dos dimensiones.

Mismo tratamiento que la figura 3D:
  - ventana de niveles centrada en la estructura (minimos/sillas), con
    'extend' para que las regiones fuera de la ventana saturen limpiamente
    (el canal de caida hacia el horizonte se ve oscuro: eso es fisica);
  - puntos criticos marcados (silla roja, minimo negro, maximo blanco);
  - SEPARATRIZ: la curva de nivel que pasa por la silla, en linea negra
    discontinua gruesa -- delimita la region ligada del canal de caida;
  - horizonte como disco negro.

Reutiliza la fisica y el buscador de puntos criticos del módulo de clasificación del potencial
(debe estar en la misma carpeta).
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from plot_potential_classification import (f_mog, horizon_mog, Veff_rtheta,
                                      find_critical_points, classify_type)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "figure.titlesize": 14,
    "mathtext.fontset": "cm",
})

CMAP = "RdYlBu_r"

def panel_contorno(ax, J, S, alpha, letra, rmax, branch=-1):
    """Panel de contorno: rectangulo limpio en (rho, z), sin disco negro."""
    r_h = horizon_mog(alpha)
    pts = find_critical_points(J, S, alpha, r_h, r_max=rmax, branch=branch)
    tipo = classify_type(pts)

    # --- ventana de niveles centrada en la estructura ---
    Vc = [p["V"] for p in pts]
    if Vc:
        spread = max(max(Vc) - min(Vc), 0.012)
        vlo = min(Vc) - 0.9*spread
        vhi = max(Vc) + 0.8*spread
    else:  # tipo C
        V_out = Veff_rtheta(np.array([rmax]), np.array([np.pi/2]),
                             J, S, alpha, branch=branch)[0]
        vlo, vhi = V_out - 0.12, V_out + 0.02

    # --- dominio rectangular en (rho, z), en el plano meridional ---
    # rho_lo: pasar el acantilado (V ecuatorial < vlo) antes de la estructura
    rho_lo = r_h*1.05
    if pts:
        r_smin = min(p["r"] for p in pts)
        rr = np.linspace(rho_lo, r_smin*0.98, 400)
        Veq = Veff_rtheta(rr, np.full_like(rr, np.pi/2), J, S, alpha, branch=branch)
        under = np.where(np.isnan(Veq) | (Veq < vlo))[0]
        if len(under):
            rho_lo = rr[under[-1]] + 1e-3
        # rho_hi: recortar si V ecuatorial cruza el tope hacia afuera (tipo D)
        r_smax = max(p["r"] for p in pts)
        rr2 = np.linspace(r_smax*1.02, rmax, 300)
        V2 = Veff_rtheta(rr2, np.full_like(rr2, np.pi/2), J, S, alpha, branch=branch)
        over = np.where(V2 > vhi)[0]
        rho_hi = rr2[over[0]] if len(over) else rmax
        # semialtura en z: estructura + margen
        z_c = max(abs(p["r"]*np.cos(p["theta"])) for p in pts)
        z_half = max(1.9*z_c + 0.35, 0.6)
    else:
        rr = np.linspace(rho_lo, rmax*0.9, 400)
        Veq = Veff_rtheta(rr, np.full_like(rr, np.pi/2), J, S, alpha, branch=branch)
        under = np.where(np.isnan(Veq) | (Veq < vlo))[0]
        if len(under):
            rho_lo = rr[under[-1]] + 1e-3
        rho_hi = rmax
        z_half = 1.0

    res = 500
    rho = np.linspace(rho_lo, rho_hi, res)
    z = np.linspace(-z_half, z_half, res)
    RHO, Z = np.meshgrid(rho, z)
    Rg = np.sqrt(RHO**2 + Z**2)
    THg = np.arccos(np.clip(Z/Rg, -1, 1))
    V = Veff_rtheta(Rg, THg, J, S, alpha, branch=branch)

    niveles = np.linspace(vlo, vhi, 24)
    cf = ax.contourf(RHO, Z, V, levels=niveles, cmap=CMAP, extend="both")
    ax.contour(RHO, Z, V, levels=niveles, colors="k",
               linewidths=0.35, alpha=0.5)

    # separatriz por la(s) silla(s)
    saddles = sorted({round(p["V"], 6) for p in pts if p["kind"] == "saddle"})
    for vs in saddles:
        ax.contour(RHO, Z, V, levels=[vs], colors="k",
                   linewidths=2.0, linestyles="--")

    # puntos criticos
    for p in pts:
        rho_p = p["r"]*np.sin(p["theta"]); z_p = p["r"]*np.cos(p["theta"])
        color = {"saddle": "red", "min": "black", "max": "white"}.get(p["kind"], "gray")
        ax.plot(rho_p, z_p, "o", color=color, ms=8,
                markeredgecolor="white" if color != "white" else "black",
                markeredgewidth=1.0, zorder=8)

    ax.set_xlabel(r"$\rho/M$")
    ax.set_ylabel(r"$z/M$")
    ax.set_title(rf"({letra})  $\alpha={alpha}$, $J={J}$, $S={S}$  — Tipo {tipo}",
                 fontsize=10.5)
    return cf

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                              constrained_layout=True)
    alpha_comun = 0.3
    CASES = [
        {"letra": "a", "J": 5.0, "S": 0.3, "rmax": 22.0},
        {"letra": "b", "J": 5.0, "S": 1.0, "rmax": 16.0},
        {"letra": "c", "J": 4.0, "S": 0.9, "rmax": 11.0},
        {"letra": "d", "J": 4.2, "S": 1.8, "rmax": 7.0},
    ]
    for ax, case in zip(axes.flat, CASES):
        cf = panel_contorno(ax, case["J"], case["S"], alpha_comun,
                             case["letra"], case["rmax"])
        cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02,
                          format="%.3f",
                          ticks=matplotlib.ticker.MaxNLocator(5, prune="both"))
        cb.set_label(r"$V_{\rm eff}$", fontsize=10)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle(r"Mapas de contorno de $V_{\rm eff}$ — Schwarzschild-MOG, "
                 rf"$\alpha={alpha_comun}$ "
                 "(línea discontinua: separatriz por la silla)", fontsize=13.5)
    fig.savefig("potential_contours.png", dpi=220)
    fig.savefig("potential_contours.pdf")
    print("Guardado: potential_contours.png/.pdf")
    plt.show()
