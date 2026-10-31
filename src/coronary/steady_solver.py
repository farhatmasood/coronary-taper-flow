"""Steady-state coronary network solver.

Under maximal hyperaemia (constant minimal microvascular resistance) coronary
flow is quasi-steady, so mean flow, mean pressure and fractional flow reserve
follow from a steady nonlinear resistive-network solve.

Each segment is a nonlinear element

    P_up - P_dn = R_conduit * Q  +  R_sten * Q  +  S_sten * Q*|Q|       (1)

with a tapered-Poiseuille conduit resistance and an optional Young-Tsai
stenosis. Flow conservation closes the system; terminal nodes are closed by
the 0D coronary outlets. The unknowns are the distal-node pressures, found
with Powell's hybrid (dogleg) Newton method as implemented in MINPACK and
exposed by scipy.optimize.root.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import root

from . import constants as C
from . import rheology, stenosis
from .coronary_bc import CoronaryOutlet
from .geometry import CoronaryTree, Segment


# --------------------------------------------------------------------------
# Element resistances
# --------------------------------------------------------------------------
def conduit_resistance(seg: Segment, mu: float) -> float:
    """Poiseuille resistance of a linearly tapered segment (Pa.s/m^3).

    R = (8 mu / pi) * integral_0^L dx / r(x)^4, evaluated in closed form.
    """
    rp, rd, length = seg.radius_prox, seg.radius_dist, seg.length
    if abs(rd - rp) < 1e-9 * rp:
        integral = length / rp**4
    else:
        integral = length / (3.0 * (rd - rp)) * (1.0 / rp**3 - 1.0 / rd**3)
    return 8.0 * mu / np.pi * integral


def invert_flow(dp: float, r_lin: float, s_quad: float) -> float:
    """Solve dp = r_lin*Q + s_quad*Q*|Q| for Q (signed)."""
    if s_quad <= 0.0:
        return dp / r_lin
    sign = 1.0 if dp >= 0.0 else -1.0
    mag = (-r_lin + np.sqrt(r_lin**2 + 4.0 * s_quad * abs(dp))) / (2.0 * s_quad)
    return sign * mag


# --------------------------------------------------------------------------
# Solution container
# --------------------------------------------------------------------------
@dataclass
class SteadySolution:
    p_distal: dict[str, float]            # distal-node pressure per segment (Pa)
    flow: dict[str, float]                # flow per segment (m^3/s)
    p_inlet: float                        # ostium pressure (Pa)
    hyperemic: bool
    converged: bool
    total_inflow: float = 0.0             # summed inlet flow (m^3/s)
    p_post_stenosis: dict[str, float] = field(default_factory=dict)
    n_evaluations: int = 0                # residual evaluations used by the solver
    residual_norm: float = 0.0            # max nodal flow imbalance (m^3/s)

    def flow_ml_min(self, seg_id: str) -> float:
        return self.flow[seg_id] * C.M3_PER_S_TO_ML_PER_MIN

    @property
    def total_inflow_ml_min(self) -> float:
        return self.total_inflow * C.M3_PER_S_TO_ML_PER_MIN


# --------------------------------------------------------------------------
# Core solve
# --------------------------------------------------------------------------
def solve_steady(tree: CoronaryTree, outlets: dict[str, CoronaryOutlet],
                 stenoses: list[stenosis.Stenosis] | None = None,
                 hyperemic: bool = False, rheology_model: str = "newtonian",
                 stiffness_scale: float = 1.0) -> SteadySolution:
    """Solve the steady coronary network. ``stiffness_scale`` is accepted for a
    uniform call signature but does not affect the steady (rigid-conduit) solve.
    """
    stenoses = stenoses or []
    sten_by_seg = {s.segment_id: s for s in stenoses}
    seg_ids = list(tree.segments.keys())
    idx = {sid: i for i, sid in enumerate(seg_ids)}
    p_in = C.P_AO_MEAN

    # frozen apparent viscosity per segment (updated by the outer Carreau loop)
    mu_seg = {sid: C.MU_BLOOD for sid in seg_ids}
    q_seg = {sid: C.Q_CORONARY_REST / len(seg_ids) for sid in seg_ids}

    def segment_flows(p_vec: np.ndarray) -> dict[str, float]:
        flows: dict[str, float] = {}
        for sid in seg_ids:
            seg = tree[sid]
            p_up = p_in if seg.parent_id not in tree.segments else p_vec[idx[seg.parent_id]]
            dp = p_up - p_vec[idx[sid]]
            r_cond = conduit_resistance(seg, mu_seg[sid])
            r_s, s_s = 0.0, 0.0
            if sid in sten_by_seg:
                r_s, s_s = stenosis.stenosis_resistances(
                    sten_by_seg[sid], seg.radius_mean, q_seg[sid], rheology_model)
            flows[sid] = invert_flow(dp, r_cond + r_s, s_s)
        return flows

    def residual(p_vec: np.ndarray) -> np.ndarray:
        """Flow conservation at every segment's distal node:
        flow through the segment = mural/terminal outlet flow + flow to daughters.
        """
        flows = segment_flows(p_vec)
        res = np.empty(len(seg_ids))
        for sid in seg_ids:
            seg = tree[sid]
            out = (outlets[sid].steady_flow(p_vec[idx[sid]], hyperemic)
                   if sid in outlets else 0.0)
            child_sum = sum(flows[c] for c in seg.children)
            res[idx[sid]] = flows[sid] - out - child_sum
        return res

    # initial guess: linear pressure ladder from inlet toward venous pressure
    p0 = np.array([p_in - 0.15 * (p_in - C.P_VENOUS) for _ in seg_ids])

    n_eval = 0
    for _outer in range(20 if rheology_model == "carreau" else 1):
        sol = root(residual, p0, method="hybr")
        n_eval += int(sol.nfev)
        p0 = sol.x
        flows = segment_flows(sol.x)
        if rheology_model != "carreau":
            break
        # refresh frozen viscosities and flows, check convergence
        max_rel = 0.0
        for sid in seg_ids:
            mu_new = rheology.effective_viscosity(
                flows[sid], tree[sid].radius_mean, model="carreau")
            max_rel = max(max_rel, abs(mu_new - mu_seg[sid]) / mu_seg[sid])
            mu_seg[sid] = mu_new
            q_seg[sid] = flows[sid]
        if max_rel < 1e-8:
            break

    p_vec = p0
    flows = segment_flows(p_vec)
    # convergence judged on the flow-conservation residual, scaled by total flow
    res_norm = float(np.max(np.abs(residual(p_vec))))
    converged = res_norm < 1e-6 * C.Q_CORONARY_REST
    p_distal = {sid: float(p_vec[idx[sid]]) for sid in seg_ids}

    # pressure immediately distal to each lesion (for FFR)
    p_post = {}
    for s in stenoses:
        seg = tree[s.segment_id]
        p_up = p_in if seg.parent_id not in tree.segments else p_distal[seg.parent_id]
        r_cond = conduit_resistance(seg, mu_seg[s.segment_id])
        dp_sten = stenosis.pressure_drop(s, seg.radius_mean, flows[s.segment_id],
                                         rheology_model)
        p_post[s.segment_id] = p_up - r_cond * s.position * flows[s.segment_id] - dp_sten

    total_in = sum(flows[sid] for sid in tree.inlets())
    return SteadySolution(p_distal=p_distal, flow=flows, p_inlet=p_in,
                          hyperemic=hyperemic, converged=converged,
                          total_inflow=total_in, p_post_stenosis=p_post,
                          n_evaluations=n_eval, residual_norm=res_norm)


# --------------------------------------------------------------------------
# Resting-resistance calibration
# --------------------------------------------------------------------------
def calibrate(tree: CoronaryTree, outlets: dict[str, CoronaryOutlet],
              target_flow: float | None = None, tol: float = 1e-9,
              max_iter: int = 30) -> float:
    """Globally rescale outlet resistances so the *solved* resting total flow
    equals ``target_flow``. Returns the achieved total flow (m^3/s).

    ``target_flow`` resolves to ``constants.Q_CORONARY_REST`` at call time
    rather than at import time, so a sensitivity sweep that perturbs the
    constant reaches this solver.
    """
    if target_flow is None:
        target_flow = C.Q_CORONARY_REST
    for _ in range(max_iter):
        sol = solve_steady(tree, outlets, hyperemic=False)
        achieved = sol.total_inflow
        factor = achieved / target_flow
        if abs(factor - 1.0) < tol:
            return achieved
        for o in outlets.values():
            o.r_term_rest *= factor
    return solve_steady(tree, outlets, hyperemic=False).total_inflow
