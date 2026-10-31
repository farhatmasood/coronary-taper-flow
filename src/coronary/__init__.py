"""Reduced-order haemodynamics of the coronary arterial tree.

Modules
-------
model             one-call builder: geometry, outflow closure, calibration
experiments       experiments on the reference tree, as plain functions
geometry          segment geometry and tree topology
wall              elastic wall law, wave speed, compliance
rheology          Newtonian and Carreau blood viscosity
stenosis          Young and Tsai focal stenosis pressure drop
coronary_bc       morphometric outflow closure and microvascular outlets
waveforms         aortic, ventricular, intramyocardial driving pressures
steady_solver     nonlinear network solve for mean flow and pressure
pulsatile_solver  linear transmission-line solve for waveforms
ffr               fractional flow reserve
wss               wall shear stress and oscillatory indices
patient           patient trees from CT angiography lumen masks (optional extras)
"""

from . import (
    constants,
    coronary_bc,
    experiments,
    ffr,
    geometry,
    model,
    pulsatile_solver,
    rheology,
    steady_solver,
    stenosis,
    wall,
    waveforms,
    wss,
)

__version__ = "1.0.0"

__all__ = [
    "constants", "geometry", "wall", "rheology", "stenosis", "coronary_bc",
    "waveforms", "steady_solver", "pulsatile_solver", "ffr", "wss",
    "model", "experiments", "__version__",
]
