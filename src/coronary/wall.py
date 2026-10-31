"""Elastic-wall mechanics: Olufsen stiffness law and the pressure-area tube law.

Wall stiffness follows the empirical Olufsen law

    Eh/r0 = k1 * exp(k2 * r0) + k3                                     (1)

with E the Young's modulus, h the wall thickness and r0 the unstressed lumen
radius. The transmural pressure-area relation (tube law) is

    p - p0 = f(r0) * (1 - sqrt(A0/A)),   f(r0) = (4/3) * (Eh/r0)        (2)

from which the small-signal wave speed and compliance follow analytically.
"""

from __future__ import annotations

import numpy as np

from . import constants as C


def eh_over_r0(r0: np.ndarray | float) -> np.ndarray | float:
    """Wall stiffness group Eh/r0 (Pa) from the Olufsen law, eq. (1)."""
    r0 = np.asarray(r0, dtype=float)
    return C.WALL_K1 * np.exp(C.WALL_K2 * r0) + C.WALL_K3


def stiffness_f(r0: np.ndarray | float, stiffness_scale: float = 1.0) -> np.ndarray | float:
    """Tube-law stiffness coefficient f(r0) = (4/3) Eh/r0 (Pa), eq. (2).

    ``stiffness_scale`` multiplies Eh/r0 and is used by the rigid-wall ablation
    (large scale -> incompliant wall).
    """
    return (4.0 / 3.0) * stiffness_scale * eh_over_r0(r0)


def wave_speed_c0(r0: np.ndarray | float, stiffness_scale: float = 1.0) -> np.ndarray | float:
    """Small-signal (Moens-Korteweg-type) wave speed at reference area (m/s).

    c0^2 = f(r0) / (2 rho)  -- obtained from c^2 = (A/rho) dp/dA at A = A0.
    """
    f = stiffness_f(r0, stiffness_scale)
    return np.sqrt(f / (2.0 * C.RHO_BLOOD))


def compliance_per_length(r0: np.ndarray | float,
                          stiffness_scale: float = 1.0) -> np.ndarray | float:
    """Vessel compliance per unit length dA/dp at reference area (m^2/Pa).

    C0 = 2 A0 / f(r0).
    """
    a0 = np.pi * np.asarray(r0, dtype=float) ** 2
    return 2.0 * a0 / stiffness_f(r0, stiffness_scale)


def pressure_from_area(area: np.ndarray | float,
                       area_ref: np.ndarray | float,
                       r0: np.ndarray | float,
                       p_ref: float = 0.0,
                       stiffness_scale: float = 1.0) -> np.ndarray | float:
    """Transmural pressure from lumen area via the tube law, eq. (2)."""
    f = stiffness_f(r0, stiffness_scale)
    area = np.maximum(np.asarray(area, dtype=float), 1e-12)
    return p_ref + f * (1.0 - np.sqrt(area_ref / area))


def area_from_pressure(pressure: np.ndarray | float,
                       area_ref: np.ndarray | float,
                       r0: np.ndarray | float,
                       p_ref: float = 0.0,
                       stiffness_scale: float = 1.0) -> np.ndarray | float:
    """Lumen area from transmural pressure -- analytic inverse of eq. (2)."""
    f = stiffness_f(r0, stiffness_scale)
    denom = 1.0 - (np.asarray(pressure, dtype=float) - p_ref) / f
    # denom -> 0 as pressure -> f (wall fully distended); clip for safety.
    denom = np.maximum(denom, 1e-3)
    return area_ref / denom**2
