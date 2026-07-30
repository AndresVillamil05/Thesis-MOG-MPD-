"""
Clasificacion del potencial efectivo V_eff para particulas con spin
en Schwarzschild-MOG -- VERSION  (sin puntas, estilo profesional).

Que cambia respecto a la version anterior (la que se veia "con puntas"):

  1. BANDA ECUATORIAL AUTOMATICA: theta se restringe a una banda alrededor
     de pi/2 dimensionada para contener los puntos criticos + margen, y se
     ENCOGE automaticamente hasta que las paredes casi verticales del pozo
     en theta quedan fuera del dominio. Eran esas paredes (cortadas por la
     region invalida del discriminante) las que producian los dientes.
  2. RECORTE RADIAL INTERNO: r_min se sube hasta pasar el acantilado de V
     hacia el horizonte, que aplastaba la estructura (minimos/sillas) en
     una sabana plana ilegible.
  3. SUPERFICIE OPACA + ANTIALIASING (la semitransparencia produce
     artefactos de orden de dibujo en mplot3d) y SIN contornos montados
     sobre la superficie (fuente de moteado). El piso conserva el mapa
     de contornos.
  4. El colormap se normaliza al rango real del dominio recortado, asi
     toda la paleta se gasta en la estructura y no en el acantilado.

La fisica (sinh Z de la cuadratica, V_eff, busqueda de puntos criticos
via Hessiana) es identica a la version verificada.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.optimize import fsolve

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "mathtext.fontset": "cm",
})

M = 1.0
m = 1.0
CMAP = "turbo"

# =============================================================================
# FISICA (identica a la version verificada)
# =============================================================================
def f_mog(r, alpha, M=1.0):
    return 1.0 - 2.0*M*(1+alpha)/r + alpha*(1+alpha)*M**2/r**2

def fprime_mog(r, alpha, M=1.0):
    return 2.0*M*(1+alpha)/r**2 - 2.0*alpha*(1+alpha)*M**2/r**3

def horizon_mog(alpha, M=1.0):
    return M*(1+alpha) + M*np.sqrt(1+alpha)

def sinhZ(r, theta, J, S, alpha, m=1.0, M=1.0, branch=-1):
    fr = f_mog(r, alpha, M)
    A = m**2*r**2 - S**2*fr
    lin = J*m*r*np.sin(theta)
    C = (J**2 - S**2)*fr + (1.0 - fr)*J**2*np.sin(theta)**2
    disc = lin**2 - A*C
    disc = np.where(disc < 0, np.nan, disc)
    return (lin + branch*np.sqrt(disc)) / A

def Veff_rtheta(r, theta, J, S, alpha, m=1.0, M=1.0, branch=-1):
    fr = f_mog(r, alpha, M)
    frp = fprime_mog(r, alpha, M)
    sZ = sinhZ(r, theta, J, S, alpha, m, M, branch)
    cZ = np.sqrt(1.0 + sZ**2)
    term1 = np.sqrt(fr) * cZ
    term2 = (r*frp/2.0) * (sZ/(np.sqrt(fr)*cZ)) * (J*np.sin(theta)/(m*r) - sZ)
    return m*(term1 + term2)

# =============================================================================
# PUNTOS CRITICOS (Hessiana) -- igual que antes
# =============================================================================
def _grad_hess(func, r, theta, h=1e-4):
    f00 = func(r, theta)
    fr1, fr2 = func(r+h, theta), func(r-h, theta)
    ft1, ft2 = func(r, theta+h), func(r, theta-h)
    Vr = (fr1-fr2)/(2*h); Vt = (ft1-ft2)/(2*h)
    Vrr = (fr1-2*f00+fr2)/h**2; Vtt = (ft1-2*f00+ft2)/h**2
    frt1 = func(r+h, theta+h); frt2 = func(r+h, theta-h)
    frt3 = func(r-h, theta+h); frt4 = func(r-h, theta-h)
    Vrt = (frt1-frt2-frt3+frt4)/(4*h**2)
    return np.array([Vr, Vt]), np.array([[Vrr, Vrt], [Vrt, Vtt]])

def find_critical_points(J, S, alpha, r_h, r_max=15.0, branch=-1, n_seed=14):
    def Vfun(r, theta):
        return Veff_rtheta(np.atleast_1d(r), np.atleast_1d(theta),
                            J, S, alpha, branch=branch)[0]
    def system(x):
        r, theta = x
        if r <= r_h*1.001 or r > 40 or theta <= 0.02 or theta >= np.pi-0.02:
            return [1e3*r, 1e3*theta]
        g, _ = _grad_hess(Vfun, r, theta)
        return g
    found = []
    for rs in np.linspace(r_h*1.05, r_max, n_seed):
        for ts in np.linspace(0.15*np.pi, 0.85*np.pi, 9):
            sol, info, ier, msg = fsolve(system, [rs, ts], full_output=True, xtol=1e-10)
            if ier != 1:
                continue
            r_c, th_c = sol
            if not (r_h*1.001 < r_c < r_max and 0.02 < th_c < np.pi-0.02):
                continue
            g, H = _grad_hess(Vfun, r_c, th_c)
            if np.max(np.abs(g)) > 1e-6:
                continue
            if any(abs(f["r"]-r_c) < 1e-3 and abs(f["theta"]-th_c) < 1e-3 for f in found):
                continue
            det = np.linalg.det(H)
            if det < -1e-10:
                kind = "saddle"
            elif det > 1e-10 and H[0, 0] > 0:
                kind = "min"
            elif det > 1e-10 and H[0, 0] < 0:
                kind = "max"
            else:
                kind = "degenerate"
            found.append({"r": r_c, "theta": th_c, "V": Vfun(r_c, th_c), "kind": kind})
    return found

def classify_type(pts):
    n_min = sum(1 for p in pts if p["kind"] == "min")
    n_saddle = sum(1 for p in pts if p["kind"] == "saddle")
    eq_saddle = any(p["kind"] == "saddle" and abs(p["theta"]-np.pi/2) < 0.05 for p in pts)
    off_saddle = any(p["kind"] == "saddle" and abs(p["theta"]-np.pi/2) > 0.05 for p in pts)
    eq_min = any(p["kind"] == "min" and abs(p["theta"]-np.pi/2) < 0.05 for p in pts)
    if eq_min and eq_saddle and not off_saddle:
        return "A"
    if eq_min and off_saddle:
        return "B"
    if (not eq_min) and (not off_saddle) and eq_saddle:
        return "D"
    if n_min == 0 and n_saddle == 0:
        return "C"
    return "?"

# =============================================================================
# DOMINIO INTELIGENTE: banda ecuatorial auto-encogida + recorte del acantilado
# =============================================================================
def _display_domain(J, S, alpha, r_h, rmax, pts, branch=-1):
    """Devuelve (r_lo, th_half): el rectangulo (r,theta) que muestra la
    estructura sin paredes verticales ni acantilado ni region invalida."""
    Vc = [p["V"] for p in pts]
    if Vc:
        spread = max(max(Vc) - min(Vc), 0.012)
        vhi_t = max(Vc) + 0.55*spread
        vlo_t = min(Vc) - 0.9*spread
        th_struct = max(abs(p["theta"] - np.pi/2) for p in pts)
    else:                       # tipo C: sin estructura -> banda fija modesta
        vhi_t = vlo_t = None
        th_struct = 0.0

    # -- banda en theta: encoger hasta que max(V) <= vhi_t (paredes afuera) --
    th_half = min(th_struct + 0.10*np.pi, 0.25*np.pi) if Vc else 0.07*np.pi
    if vhi_t is not None:
        for cand in np.linspace(th_half, th_struct + 0.02*np.pi, 14):
            rr = np.linspace(r_h*1.6, rmax, 120)
            tt = np.linspace(np.pi/2 - cand, np.pi/2 + cand, 120)
            Rg, Tg = np.meshgrid(rr, tt)
            Vg = Veff_rtheta(Rg, Tg, J, S, alpha, branch=branch)
            if np.nanmax(Vg) <= vhi_t:
                th_half = cand
                break
        else:
            th_half = th_struct + 0.02*np.pi

    # -- r_lo: primero validez (sin NaN), luego pasar el acantilado --
    def sin_nan(r_lo):
        rr = np.linspace(r_lo, rmax, 300)
        tt = np.linspace(np.pi/2 - th_half, np.pi/2 + th_half, 300)
        Rg, Tg = np.meshgrid(rr, tt)
        return not np.isnan(Veff_rtheta(Rg, Tg, J, S, alpha, branch=branch)).any()

    r_lo = None
    for fac in [1.02, 1.08, 1.15, 1.25, 1.40, 1.60]:
        if sin_nan(r_h*fac):
            r_lo = r_h*fac
            break
    if r_lo is None:
        r_lo = r_h*1.6

    if vlo_t is None:
        Veq_out = Veff_rtheta(np.array([rmax]), np.array([np.pi/2]),
                               J, S, alpha, branch=branch)[0]
        vlo_t = Veq_out - 0.12
        r_scan_hi = rmax*0.9
    else:
        r_scan_hi = min(p["r"] for p in pts)*0.98
    if True:
        rr_scan = np.linspace(r_lo, r_scan_hi, 400)
        Vscan = Veff_rtheta(rr_scan, np.full_like(rr_scan, np.pi/2),
                             J, S, alpha, branch=branch)
        under = np.where(np.isnan(Vscan) | (Vscan < vlo_t))[0]
        if len(under):
            r_lo = max(r_lo, rr_scan[under[-1]] + 1e-3)
    return r_lo, th_half

# =============================================================================
# PANEL 
# =============================================================================
def disenar_panel(ax, J, S, alpha, letra_panel, rmax, branch=-1):
    r_h = horizon_mog(alpha)
    pts = find_critical_points(J, S, alpha, r_h, r_max=rmax, branch=branch)
    tipo = classify_type(pts)
    r_lo, th_half = _display_domain(J, S, alpha, r_h, rmax, pts, branch)

    res = 350

    # ---- semiancho adaptativo w(r): el mayor h con V(r, pi/2 +- h) <= vhi_t
    # (biseccion; V crece monotonamente al alejarse del ecuador cerca de la
    # estructura). El borde de la malla sigue la curva de nivel V = vhi_t. ----
    Vc_all = [p["V"] for p in pts]
    if Vc_all:
        spread_ = max(max(Vc_all) - min(Vc_all), 0.012)
        vhi_t = max(Vc_all) + 0.55*spread_
    else:
        vhi_t = None

    # recortar rmax donde V ecuatorial cruza el tope de la ventana hacia
    # afuera (si no, la banda colapsa en una lengueta degenerada)
    rmax_eff = rmax
    if vhi_t is not None and pts:
        r_struct_max = max(p["r"] for p in pts)
        rr_out = np.linspace(r_struct_max*1.02, rmax, 300)
        V_out = Veff_rtheta(rr_out, np.full_like(rr_out, np.pi/2),
                             J, S, alpha, branch=branch)
        over = np.where(V_out > vhi_t)[0]
        if len(over):
            rmax_eff = rr_out[over[0]]
    r = np.linspace(r_lo, rmax_eff, res)

    def semiancho(ri):
        if vhi_t is None:
            return th_half
        def V_at(h):
            v = Veff_rtheta(np.array([ri, ri]),
                             np.array([np.pi/2 - h, np.pi/2 + h]),
                             J, S, alpha, branch=branch)
            return np.nanmax(v) if not np.isnan(v).all() else np.inf
        if V_at(th_half) <= vhi_t:
            return th_half
        lo_h, hi_h = 0.0, th_half
        for _ in range(40):
            mid = 0.5*(lo_h + hi_h)
            if V_at(mid) <= vhi_t:
                lo_h = mid
            else:
                hi_h = mid
        return max(lo_h, 0.015)

    w = np.array([semiancho(ri) for ri in r])
    u = np.linspace(-1.0, 1.0, res)
    TH = np.pi/2 + np.outer(u, w)          # malla curvilinea theta(i,j)
    R = np.tile(r, (res, 1))
    V = Veff_rtheta(R, TH, J, S, alpha, branch=branch)
    RHO = R*np.sin(TH)
    Z = R*np.cos(TH)

    vlo, vhi = np.nanmin(V), np.nanmax(V)
    V = np.where(np.isnan(V), vhi, V)      # astillas residuales (raras) al tope
    zfloor = vlo - 0.45*(vhi - vlo)
    norm = mcolors.Normalize(vmin=vlo, vmax=vhi)

    rho_lo, rho_hi = RHO.min(), RHO.max()
    z_lo, z_hi = Z.min(), Z.max()
    z_pad = 0.10*(z_hi - z_lo) if z_hi > z_lo else 0.5
    rho_pad = 0.04*(rho_hi - rho_lo)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"]["linestyle"] = ":"
        axis._axinfo["grid"]["alpha"] = 0.35

    ax.plot_surface(RHO, Z, V, cmap=CMAP, norm=norm,
                     rcount=res, ccount=res, antialiased=False,
                     linewidth=0, alpha=1.0, shade=True)

    ax.contourf(RHO, Z, V, levels=np.linspace(vlo, vhi, 28), zdir="z",
                offset=zfloor, cmap=CMAP, norm=norm, alpha=0.95)
    ax.contour(RHO, Z, V, levels=np.linspace(vlo, vhi, 14), zdir="z",
               offset=zfloor, colors="k", linewidths=0.3, alpha=0.35)

    for p in pts:
        rho_p = p["r"]*np.sin(p["theta"]); z_p = p["r"]*np.cos(p["theta"])
        color = "red" if p["kind"] == "saddle" else "black"
        ax.plot([rho_p]*2, [z_p]*2, [zfloor, p["V"]],
                color="black", lw=1.0, alpha=0.7)
        ax.scatter([rho_p], [z_p], [p["V"]], color=color, s=42,
                   edgecolors="white", linewidths=0.7, depthshade=False)
        ax.scatter([rho_p], [z_p], [zfloor], color=color, s=15,
                   edgecolors="white", linewidths=0.5, depthshade=False)

    ax.set_xlabel(r"$\rho/M$", labelpad=6)
    ax.set_ylabel(r"$z/M$", labelpad=6)
    ax.set_zlabel(r"$V_{\rm eff}$", labelpad=5)
    ax.set_xlim(rho_lo - rho_pad, rho_hi + rho_pad)
    ax.set_ylim(z_lo - z_pad, z_hi + z_pad)
    ax.set_zlim(zfloor, vhi + 0.03*(vhi - vlo))
    ax.set_box_aspect((1.55, 1.0, 0.75))
    ax.set_title(rf"$\alpha={alpha}$, $J={J}$, $S={S}$  (Tipo {tipo})",
                 fontsize=10, pad=0)
    ax.text2D(0.5, -0.02, f"({letra_panel})", transform=ax.transAxes,
              fontsize=12, weight="bold", ha="center")
    ax.view_init(elev=22, azim=-62)
    ax.tick_params(labelsize=7.5)

# =============================================================================
# FIGURA COMPUESTA 2x2 (tipos A, B, C, D)
# =============================================================================
if __name__ == "__main__":
    fig = plt.figure(figsize=(12.5, 10), dpi=130)
    alpha_comun = 0.3
    CASES = [
        {"letra": "a", "J": 5.0, "S": 0.3, "rmax": 22.0},
        {"letra": "b", "J": 5.0, "S": 1.0, "rmax": 16.0},
        {"letra": "c", "J": 4.0, "S": 0.9, "rmax": 11.0},
        {"letra": "d", "J": 4.2, "S": 1.8, "rmax": 7.0},
    ]
    for i, case in enumerate(CASES):
        ax = fig.add_subplot(2, 2, i+1, projection="3d", computed_zorder=False)
        disenar_panel(ax, case["J"], case["S"], alpha_comun,
                      case["letra"], case["rmax"])
    fig.suptitle(r"Clasificación del potencial efectivo $V_{\rm eff}$ — "
                 rf"Schwarzschild-MOG, $\alpha={alpha_comun}$", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.12, top=0.93)
    fig.savefig("potential_classification.png", dpi=220)
    fig.savefig("potential_classification.pdf")
    print("Guardado: potential_classification.png/.pdf")
    plt.show()
