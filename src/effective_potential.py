"""
Dinamica de particulas con spin en Schwarzschild-MOG.
Implementa el potencial efectivo V_eff(r,theta;J,S,alpha) generalizando
Suzuki-Maeda (1998) / Giri et al. (2022, SQBH) a la metrica MOG.

Unidades geometricas: G_N = c = 1, M = 1 (masa del agujero), m = 1 (masa
de la particula de prueba). J y S quedan medidos en unidades de mM.
"""
import numpy as np

# ----------------------------------------------------------------------
# Metrica de Schwarzschild-MOG (ec. 2-22 de la tesis)
# ----------------------------------------------------------------------
def f_mog(r, alpha, M=1.0):
    return 1.0 - 2.0*M*(1+alpha)/r + alpha*(1+alpha)*M**2/r**2

def fprime_mog(r, alpha, M=1.0):
    return 2.0*M*(1+alpha)/r**2 - 2.0*alpha*(1+alpha)*M**2/r**3

def horizon_mog(alpha, M=1.0):
    return M*(1+alpha) + M*np.sqrt(1+alpha)

def isco_schw_mog(alpha, M=1.0):
    """
    ISCO sin spin (solución de Cardano).
    """
    Z = (alpha**3 + 8*alpha**2 + 15*alpha + 8
         + alpha*(alpha+1)*np.sqrt(alpha+5))**(1/3)
    return M*(Z + (alpha**2+5*alpha+4)/Z + 2*(alpha+1))


# ----------------------------------------------------------------------
# sinh(Z) y potencial efectivo (generalizacion no-ecuatorial)
# ----------------------------------------------------------------------
def sinhZ(r, theta, J, S, alpha, m=1.0, M=1.0, branch=-1):
    """
    Resuelve:
    (m^2 r^2 - S^2 f) sinh^2 Z - 2 J m r sin(theta) sinh Z
        + (J^2 - S^2) f + (1 - f) J^2 sin^2(theta) = 0
    branch = -1  -> rama V_- (la usada para el analisis de caos)
    branch = +1  -> rama V_+
    """
    fr = f_mog(r, alpha, M)
    A = m**2*r**2 - S**2*fr
    lin = J*m*r*np.sin(theta)
    C = (J**2 - S**2)*fr + (1.0 - fr)*J**2*np.sin(theta)**2
    disc = lin**2 - A*C
    disc = np.where(disc < 0, np.nan, disc)
    return (lin + branch*np.sqrt(disc)) / A


def Veff(r, theta, J, S, alpha, m=1.0, M=1.0, branch=-1):
    fr = f_mog(r, alpha, M)
    frp = fprime_mog(r, alpha, M)
    sZ = sinhZ(r, theta, J, S, alpha, m, M, branch)
    cZ = np.sqrt(1.0 + sZ**2)
    term1 = np.sqrt(fr) * cZ
    term2 = (r*frp/2.0) * (sZ/(np.sqrt(fr)*cZ)) * (J*np.sin(theta)/(m*r) - sZ)
    return m*(term1 + term2)


# ----------------------------------------------------------------------
# Comprobacion: limite Schwarzschild (alpha=0) reproduce Suzuki-Maeda
# ----------------------------------------------------------------------
if __name__ == "__main__":
    r, theta = 6.0, np.pi/2
    J, S, alpha = 3.959, 1.0, 0.0
    print("f(6,0) =", f_mog(r, 0.0), " (esperado 1-2/6=0.6667)")
    print("V_eff(r=6,theta=pi/2; J=3.959,S=1,alpha=0) =",
          Veff(r, theta, J, S, alpha))
    print("Horizonte alpha=0:", horizon_mog(0.0), " (esperado 2.0)")
    print("Horizonte alpha=0.6:", horizon_mog(0.6))
    print("ISCO(alpha=0):", isco_schw_mog(0.0), " (esperado 6.0)")
    print("ISCO(alpha=0.6):", isco_schw_mog(0.6))
