"""Fractional flow reserve (FFR) from the steady hyperaemic solve.

FFR is the ratio of maximal achievable myocardial flow with a stenosis to the
maximal flow that would exist without it. Under maximal hyperaemia the
microvascular resistance is minimal and constant, so flow scales with pressure
and FFR reduces to a pressure ratio measured across the lesion:

    FFR        = P_d / P_a                        (clinical convention)
    FFR_corr   = (P_d - P_v) / (P_a - P_v)         (venous-corrected)

P_a is mean aortic pressure, P_d the mean pressure immediately distal to the
lesion, P_v the coronary venous pressure. The clinical ischaemia threshold is
FFR <= 0.80.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from . import stenosis as stenosis_mod
from .coronary_bc import CoronaryOutlet
from .geometry import CoronaryTree
from .steady_solver import solve_steady

FFR_ISCHEMIA_THRESHOLD = 0.80


@dataclass
class FFRResult:
    segment_id: str
    diameter_stenosis: float
    area_stenosis_pct: float
    ffr: float                       # P_d / P_a
    ffr_corrected: float             # (P_d - P_v)/(P_a - P_v)
    p_aorta_mmhg: float
    p_distal_mmhg: float
    flow_rest_ml_min: float          # lesion flow at rest
    flow_hyper_ml_min: float         # lesion flow at hyperaemia
    flow_reserve: float              # hyperaemic / resting lesion flow
    ischemic: bool


def compute_ffr(tree: CoronaryTree, outlets: dict[str, CoronaryOutlet],
                lesion: stenosis_mod.Stenosis,
                rheology_model: str = "newtonian") -> FFRResult:
    """Compute FFR for a single lesion."""
    rest = solve_steady(tree, outlets, [lesion], hyperemic=False,
                        rheology_model=rheology_model)
    hyper = solve_steady(tree, outlets, [lesion], hyperemic=True,
                         rheology_model=rheology_model)

    p_a = hyper.p_inlet
    p_d = hyper.p_post_stenosis[lesion.segment_id]
    ffr = p_d / p_a
    ffr_corr = (p_d - C.P_VENOUS) / (p_a - C.P_VENOUS)

    q_rest = rest.flow[lesion.segment_id]
    q_hyper = hyper.flow[lesion.segment_id]

    return FFRResult(
        segment_id=lesion.segment_id,
        diameter_stenosis=lesion.diameter_stenosis,
        area_stenosis_pct=lesion.area_stenosis_pct,
        ffr=ffr,
        ffr_corrected=ffr_corr,
        p_aorta_mmhg=p_a * C.PA_TO_MMHG,
        p_distal_mmhg=p_d * C.PA_TO_MMHG,
        flow_rest_ml_min=q_rest * C.M3_PER_S_TO_ML_PER_MIN,
        flow_hyper_ml_min=q_hyper * C.M3_PER_S_TO_ML_PER_MIN,
        flow_reserve=q_hyper / q_rest if q_rest > 0 else float("nan"),
        ischemic=ffr <= FFR_ISCHEMIA_THRESHOLD,
    )


def ffr_curve(tree: CoronaryTree, outlets: dict[str, CoronaryOutlet],
              segment_id: str, ds_values: np.ndarray,
              lesion_length: float = 10.0e-3,
              rheology_model: str = "newtonian") -> list[FFRResult]:
    """FFR as a function of diameter-stenosis severity for one segment."""
    results = []
    for ds in ds_values:
        if ds <= 0.0:
            continue
        lesion = stenosis_mod.Stenosis(segment_id, float(ds), length=lesion_length)
        results.append(compute_ffr(tree, outlets, lesion, rheology_model))
    return results
