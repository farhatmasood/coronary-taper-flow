"""Wall shear stress (WSS) and related haemodynamic indices.

For the cross-sectionally averaged (Poiseuille) description the wall shear
stress in a vessel of lumen radius r carrying flow Q is

    tau_w = 4 * mu * Q / (pi * r^3)                                     (1)

From a pulsatile WSS history tau_w(t) over one cycle the time-averaged wall
shear stress (TAWSS) and oscillatory shear index (OSI) are

    TAWSS = (1/T) * integral |tau_w| dt                                 (2)
    OSI   = 0.5 * (1 - |integral tau_w dt| / integral |tau_w| dt)       (3)

Low TAWSS and high OSI mark regions prone to atherosclerosis.
"""

from __future__ import annotations

import numpy as np

from . import constants as C
from . import rheology
from ._compat import trapezoid


def wall_shear_stress(flow: float | np.ndarray, radius: float,
                      rheology_model: str = "newtonian") -> float | np.ndarray:
    """Wall shear stress (Pa) from flow and lumen radius, eq. (1)."""
    radius = max(radius, 1e-6)
    if rheology_model == "newtonian":
        mu = C.MU_BLOOD
    else:
        shear = 4.0 * np.abs(flow) / (np.pi * radius**3)
        mu = rheology.carreau_viscosity(shear)
    return 4.0 * mu * np.asarray(flow, dtype=float) / (np.pi * radius**3)


def tawss(tau_history: np.ndarray, time: np.ndarray) -> float:
    """Time-averaged wall shear stress over one cycle, eq. (2)."""
    period = time[-1] - time[0]
    return float(trapezoid(np.abs(tau_history), time) / period)


def osi(tau_history: np.ndarray, time: np.ndarray) -> float:
    """Oscillatory shear index, eq. (3); 0 = unidirectional, 0.5 = fully oscillatory."""
    integral_signed = abs(trapezoid(tau_history, time))
    integral_abs = trapezoid(np.abs(tau_history), time)
    if integral_abs < 1e-20:
        return 0.0
    return float(0.5 * (1.0 - integral_signed / integral_abs))
