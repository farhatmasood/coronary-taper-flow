"""Focal stenosis pressure-drop model (Young & Tsai type).

A stenosis is described by a single discrete element whose trans-lesional
pressure drop is the sum of a viscous (Poiseuille) term linear in flow and a
flow-separation term quadratic in flow:

    dP = R_v * Q  +  S_e * Q * |Q|                                     (1)

    R_v = 8 * pi * mu * L_s / A_s^2                  (Poiseuille, throat)  (2)
    S_e = K_t * (rho / 2) * (1/A_s - 1/A_0)^2        (Borda-Carnot type)  (3)

A_0 is the nominal (healthy) lumen area, A_s the minimum (throat) area and
L_s the lesion length. The separation term is the sudden-expansion loss with
the empirical Young & Tsai coefficient K_t = 1.52, which accounts for the
convergent section and turbulence in addition to the ideal Borda-Carnot loss.

A stenosis of diameter-reduction DS in [0, 1) has

    A_s = A_0 * (1 - DS)^2 ,   r_s = r_0 * (1 - DS).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from . import rheology

# Young & Tsai empirical separation-loss coefficient.
K_T = 1.52


@dataclass
class Stenosis:
    """A focal stenosis placed on a named segment.

    Parameters
    ----------
    segment_id      : id of the segment carrying the lesion
    diameter_stenosis : fractional diameter reduction DS in [0, 1)
    length          : lesion length L_s (m); default 10 mm focal lesion
    position        : normalised location along the segment in [0, 1]
    """

    segment_id: str
    diameter_stenosis: float
    length: float = 10.0e-3
    position: float = 0.5
    include_separation: bool = True     # set False for the viscous-only ablation

    def __post_init__(self) -> None:
        if not 0.0 <= self.diameter_stenosis < 1.0:
            raise ValueError("diameter_stenosis must be in [0, 1)")

    @property
    def area_stenosis_ratio(self) -> float:
        """A_s / A_0."""
        return (1.0 - self.diameter_stenosis) ** 2

    @property
    def area_stenosis_pct(self) -> float:
        """Cross-sectional area reduction (%) -- the angiographic 'area stenosis'."""
        return 100.0 * (1.0 - self.area_stenosis_ratio)


def stenosis_resistances(stenosis: Stenosis, nominal_radius: float,
                         flow: float = 0.0, rheology_model: str = "newtonian"
                         ) -> tuple[float, float]:
    """Return (R_v, S_e) of eq. (2)-(3) for a lesion on a vessel of radius r_0.

    For the non-Newtonian model the viscous coefficient uses the apparent
    viscosity evaluated at the throat (high-shear) radius.
    """
    a0 = np.pi * nominal_radius**2
    r_s = nominal_radius * (1.0 - stenosis.diameter_stenosis)
    a_s = np.pi * r_s**2

    mu = rheology.effective_viscosity(flow, r_s, model=rheology_model)

    r_v = 8.0 * np.pi * mu * stenosis.length / a_s**2
    s_e = K_T * 0.5 * C.RHO_BLOOD * (1.0 / a_s - 1.0 / a0) ** 2
    if not stenosis.include_separation:
        s_e = 0.0
    return r_v, s_e


def pressure_drop(stenosis: Stenosis, nominal_radius: float, flow: float,
                  rheology_model: str = "newtonian") -> float:
    """Trans-lesional pressure drop dP (Pa) for a given flow, eq. (1)."""
    r_v, s_e = stenosis_resistances(stenosis, nominal_radius, flow, rheology_model)
    return r_v * flow + s_e * flow * abs(flow)


def effective_resistance(stenosis: Stenosis, nominal_radius: float, flow: float,
                         rheology_model: str = "newtonian") -> float:
    """Secant resistance dP/Q (Pa.s/m^3) of the lesion at the operating flow."""
    q = flow if abs(flow) > 1e-12 else 1e-12
    return pressure_drop(stenosis, nominal_radius, q, rheology_model) / q
