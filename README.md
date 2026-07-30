# Dinámica de partículas con espín en Schwarzschild-MOG

Código numérico desarrollado para el trabajo de grado **"Estudio del movimiento de
partículas con espín alrededor de un agujero negro Schwarzschild-MOG: implementación
numérica y análisis del caos"** (Programa de Pregrado en Física, Observatorio
Astronómico Nacional, Universidad Nacional de Colombia).

El proyecto integra numéricamente las ecuaciones de **Mathisson–Papapetrou–Dixon
(MPD)** bajo la **condición suplementaria de Tulczyjew–Dixon** para una partícula de
prueba con espín en la geometría de Schwarzschild derivada de la teoría de Gravedad
Escalar-Tensorial-Vectorial (STVG/MOG) de Moffat, y analiza la dinámica resultante
mediante el potencial efectivo, secciones de Poincaré y exponentes de Lyapunov.

---

## Estructura del repositorio

```
src/         Núcleo físico: geometría, sistema de ecuaciones e integrador
analysis/    Herramientas de análisis dinámico (Poincaré, Lyapunov)
figures/     Scripts que generan las figuras del documento
```

### `src/` — núcleo físico

| Archivo | Descripción |
|---|---|
| `geometry.py` | Cálculo simbólico (SymPy) de los símbolos de Christoffel y el tensor de Riemann de la métrica MOG. Se valida contra el escalar de Kretschmann de Schwarzschild. |
| `generate_equations.py` | Despeja algebraicamente el tensor de espín y la cuadrivelocidad, y genera el lado derecho del sistema (`equations_of_motion.py`) con eliminación de subexpresiones comunes. |
| `equations_of_motion.py` | Lado derecho del sistema de 8 EDOs (generado automáticamente). |
| `integrator.py` | Integrador (DOP853, paso adaptativo) con detección de eventos y cálculo de las cantidades conservadas. |
| `initial_conditions.py` | Construcción de condiciones iniciales fijando las constantes exactas (E, J_z, m, S). |
| `effective_potential.py` | Potencial efectivo V_eff(r, θ) para partícula con espín. |
| `potential_classification.py` | Clasificación del potencial (tipos A–D) mediante análisis de puntos críticos. |

### `analysis/` — herramientas dinámicas

| Archivo | Descripción |
|---|---|
| `poincare_section.py` | Secciones de Poincaré (superficie θ = π/2). |
| `poincare_section_batch.py` | Cálculo de secciones para múltiples condiciones iniciales. |
| `lyapunov_exponent.py` | Exponente de Lyapunov máximo (método de Benettin). |
| `accessible_region.py` | Mapa de la región accesible de la superficie de sección. |
| `resonance_scan.py` | Barrido fino de parámetros en la cadena de resonancia. |

### `figures/` — generación de figuras

| Archivo | Figura que genera |
|---|---|
| `plot_potential_classification.py` | Clasificación 3D del potencial efectivo (tipos A–D). |
| `plot_potential_contours.py` | Mapas de contorno del potencial en el plano meridional. |
| `plot_potential_vs_alpha.py` | Perfil del potencial para distintos valores de α. |
| `plot_potential_equatorial.py` | Potencial efectivo ecuatorial. |
| `plot_potential_vs_spin.py` | Perfil del potencial para distintos valores de espín. |
| `plot_orbits_by_type.py` | Órbitas representativas de cada topología. |
| `plot_orbits_alpha_comparison.py` | Comparación de órbitas variando α. |
| `plot_orbits_xz.py` | Órbitas en proyección x–z. |
| `plot_orbits_xz_scaled.py` | Órbitas en x–z con el horizonte a escala. |
| `plot_orbits_meridional.py` | Órbitas en el plano meridional (ρ, z). |
| `plot_orbits_topview.py` | Vista superior de las órbitas. |
| `plot_regular_vs_chaotic.py` | Comparación entre órbita regular y transitoria. |
| `plot_poincare_sections.py` | Secciones de Poincaré. |

---

## Requisitos

- Python 3.9+
- NumPy, SciPy, SymPy, Matplotlib

```bash
pip install -r requirements.txt
```

---

## Uso

Los scripts de `figures/` y `analysis/` importan el núcleo desde `src/` mediante
rutas relativas, por lo que pueden ejecutarse directamente:

```bash
# 1. Generar el sistema de ecuaciones (produce equations_of_motion.py)
python src/generate_equations.py

# 2. Integrar una órbita y verificar la conservación de las constantes
python src/integrator.py

# 3. Reproducir una figura, p. ej. la clasificación del potencial
python figures/plot_potential_classification.py
```

---

## Notas sobre la implementación

- **Unidades geometrizadas** G = c = 1, con M = m = 1. El momento angular J y el
  espín S se expresan en unidades de mM.
- **Control de calidad:** a lo largo de cada trayectoria se monitorean las cuatro
  cantidades conservadas (E, J_z, m², S²), que se preservan típicamente a ~10⁻¹³,
  y se verifica que la línea de mundo permanece temporal (u·u = −1).
- La condición de Tulczyjew–Dixon determina las componentes temporales del tensor
  de espín; las espaciales se fijan orientando los ejes con J_x = J_y = 0, J_z = J.

---

## Referencias principales

- J. W. Moffat, *Scalar-tensor-vector gravity theory*, JCAP **2006**(03), 004 (2006).
- S. Suzuki & K. Maeda, *Chaos in Schwarzschild spacetime: the motion of a spinning
  particle*, Phys. Rev. D **55**, 4848 (1997).
- S. Giri, P. Sheoran, H. Nandan & S. Shaymatov, *Chaotic motion and Periastron
  precession of spinning test particles moving in the vicinage of a Schwarzschild
  black hole surrounded by a quintessence matter field*, Eur. Phys. J. Plus
  **138**, 245 (2023).

---

## Autor

Andrés Julián Villamil Barros
Director: Eduard Alexis Larragaña
Universidad Nacional de Colombia — Observatorio Astronómico Nacional
