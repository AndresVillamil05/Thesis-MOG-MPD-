"""
Orbitas Tipo B en el PLANO MERIDIONAL (rho, z) -- version mejorada:

  1. Se grafica (rho, z) = (r sin(theta), r cos(theta)) en lugar de (x, z):
     la proyeccion x-z dobla la trayectoria en +-x y amontona las hebras;
     el plano meridional muestra la geometria limpia del movimiento.
  2. La trayectoria se colorea por tiempo propio tau (colormap), de modo
     que las hebras individuales se distinguen y se ve la evolucion.
  3. Se superpone la CURVA DE VELOCIDAD CERO V_eff(rho,z) = E (linea negra):
     la frontera de la region permitida. La orbita "llena" exactamente esa
     region -- lo que antes parecia amontonamiento ahora es fisica visible.
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
from effective_potential import Veff

plt.rcParams.update({"font.size": 10, "font.family": "serif",
                      "mathtext.fontset": "cm"})

M = m = 1.0
ALPHA = 0.3
J, S = 5.0, 1.0
r_h = horizon_mog(ALPHA)
R0S = [8.8, 7.0, 6.0, 5.0, 4.6]
TAU_MAX = 4000.0
CMAP_TRAJ = "viridis"

def curva_velocidad_cero(ax, E, J_orb, S_orb, m_orb, rho_rng, z_rng):
    """Contorno V_eff(rho,z) = E con las cantidades conservadas REALES
    de la orbita (Jz y S medidos, no los nominales): asi la curva acota
    exactamente el movimiento."""
    rho = np.linspace(*rho_rng, 400)
    z = np.linspace(*z_rng, 400)
    RHO, Z = np.meshgrid(rho, z)
    Rg = np.sqrt(RHO**2 + Z**2)
    THg = np.arccos(np.clip(Z/Rg, -1, 1))
    V = Veff(Rg, THg, J_orb, S_orb, ALPHA, m=m_orb, branch=-1)
    ax.contour(RHO, Z, V, levels=[E], colors="k", linewidths=1.6, zorder=5)

fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5))
axes = axes.flat
lc_ref = None

for i, r0 in enumerate(R0S):
    y0 = initial_conditions_equatorial(r0=r0, E=0.94, J=J, S=S, alpha=ALPHA,
                                        sign_pr=-1)
    y0[2] = 1.5
    E0, Jz0, m20, S20 = conserved(y0, J, ALPHA, M, m)
    sol = integrate(y0, J, ALPHA, tau_max=TAU_MAX, rtol=1e-10, atol=1e-12)
    r = sol.y[1]; th = sol.y[2]
    rho = r*np.sin(th); z = r*np.cos(th)

    ax = axes[i]
    # trayectoria coloreada por tiempo propio
    puntos = np.array([rho, z]).T.reshape(-1, 1, 2)
    segs = np.concatenate([puntos[:-1], puntos[1:]], axis=1)
    lc = LineCollection(segs, cmap=CMAP_TRAJ,
                        norm=plt.Normalize(0, sol.t[-1]), linewidths=0.55,
                        alpha=0.9)
    lc.set_array(sol.t[:-1])
    ax.add_collection(lc)
    lc_ref = lc

    # curva de velocidad cero para la energia de ESTA orbita
    pad_r = 0.08*(rho.max() - rho.min() + 1)
    pad_z = 0.15*(np.abs(z).max() + 0.3)
    rho_rng = (max(r_h*1.02, rho.min() - pad_r), rho.max() + pad_r)
    z_rng = (-np.abs(z).max() - pad_z, np.abs(z).max() + pad_z)
    curva_velocidad_cero(ax, E0, Jz0, np.sqrt(S20), np.sqrt(m20),
                          rho_rng, z_rng)

    ax.plot(rho[0], z[0], "o", color="red", ms=6, zorder=10)
    ax.set_xlim(*rho_rng); ax.set_ylim(*z_rng)
    ax.set_xlabel(r"$\rho/M$"); ax.set_ylabel(r"$z/M$")
    ax.set_title(rf"$r_0={r0}$,  $E={E0:.4f}$", fontsize=10)

# panel de leyenda
ax = axes[len(R0S)]
ax.axis("off")
ax.text(0.02, 0.95,
        rf"$\alpha={ALPHA}$,  $J={J}\,mM$,  $S={S}\,mM$" + "\n"
        rf"$\theta_0=1.5$,  $\tau\leq{TAU_MAX:g}$" + "\n\n"
        "Plano meridional $(\\rho, z)$.\n"
        "Línea negra: curva de velocidad\n"
        "cero $V_{\\rm eff}=E$ (frontera de la\n"
        "región permitida).\n"
        "Punto rojo: posición inicial.\n"
        "(La curva usa los valores conservados\n"
        "reales de cada órbita: $J_z$, $S$ y $m$.)",
        transform=ax.transAxes, fontsize=10, va="top", family="serif")
cbar = fig.colorbar(lc_ref, ax=ax, orientation="horizontal",
                    fraction=0.08, pad=0.04, shrink=0.85)
cbar.set_label(r"tiempo propio $\tau$", fontsize=10)

fig.suptitle("Órbitas de partícula con espín en el plano meridional — "
             "potencial Tipo B", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("orbits_typeB_meridional.png", dpi=200)
fig.savefig("orbits_typeB_meridional.pdf")
print("guardado orbits_typeB_meridional")
