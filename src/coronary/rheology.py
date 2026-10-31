"""Blood rheology: Newtonian and shear-thinning (Carreau) viscosity models.

For the reduced (1D / 0D) description the relevant quantity is an apparent
viscosity representative of a vessel cross-section. For Poiseuille-type flow
the characteristic wall shear rate is

    gamma_dot = 4 * |Q| / (pi * r^3)                                   (1)

and a Carreau apparent viscosity is evaluated at that shear rate. This keeps
the non-Newtonian effect consistent with the resistance terms used elsewhere.
"""

from __future__ import annotations

import numpy as np

from . import constants as C


def newtonian_viscosity() -> float:
    """Constant dynamic viscosity (Pa.s)."""
    return C.MU_BLOOD


def carreau_viscosity(shear_rate: np.ndarray | float) -> np.ndarray | float:
    """Carreau shear-thinning apparent viscosity (Pa.s).

    mu(g) = mu_inf + (mu_0 - mu_inf) * [1 + (lambda*g)^2] ^ ((n-1)/2)
    """
    g = np.asarray(shear_rate, dtype=float)
    bracket = 1.0 + (C.CARREAU_LAMBDA * g) ** 2
    mu = C.CARREAU_MU_INF + (C.CARREAU_MU_0 - C.CARREAU_MU_INF) * bracket ** (
        (C.CARREAU_N - 1.0) / 2.0)
    return mu


def wall_shear_rate(flow: float, radius: float) -> float:
    """Characteristic wall shear rate (1/s) for Poiseuille flow, eq. (1)."""
    radius = max(radius, 1e-6)
    return 4.0 * abs(flow) / (np.pi * radius**3)


def effective_viscosity(flow: float, radius: float, model: str = "newtonian") -> float:
    """Apparent viscosity for a vessel carrying ``flow`` at lumen ``radius``.

    Parameters
    ----------
    flow   : volumetric flow rate (m^3/s)
    radius : representative lumen radius (m)
    model  : 'newtonian' or 'carreau'
    """
    if model == "newtonian":
        return C.MU_BLOOD
    if model == "carreau":
        return float(carreau_viscosity(wall_shear_rate(flow, radius)))
    raise ValueError(f"unknown rheology model: {model!r}")
