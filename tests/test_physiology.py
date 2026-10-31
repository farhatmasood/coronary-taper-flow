"""Physiological acceptance tests.

These assert documented physiological behaviour, not that the solver
reproduces its own previous output. A change that alters the physiology
fails the suite even if the code still runs.
"""

from __future__ import annotations

import numpy as np

from coronary import constants as C
from coronary import coronary_bc, experiments, ffr, geometry, model, steady_solver, stenosis


# ----------------------------------------------------------------- topology
def test_tree_is_well_formed():
    tree = geometry.load_default_tree()
    assert len(tree) == 17
    assert len(tree.inlets()) == 2
    for seg in tree:
        assert seg.radius_dist <= seg.radius_prox + 1e-12, f"{seg.seg_id} widens"
        assert seg.length > 0


# ------------------------------------------------------------- steady state
def test_calibration_hits_target_flow(m):
    assert m.rest.converged
    assert np.isclose(m.rest.total_inflow_ml_min, C.Q_CORONARY_REST_ML_MIN, rtol=1e-6)


def test_calibration_target_is_read_at_call_time():
    """A perturbed constant must reach the solver rather than being frozen
    into a default argument (guards the sensitivity analysis)."""
    tree = geometry.load_default_tree()
    outlets = coronary_bc.build_outlets(tree)
    achieved = steady_solver.calibrate(tree, outlets, target_flow=0.5 * C.Q_CORONARY_REST)
    assert np.isclose(achieved, 0.5 * C.Q_CORONARY_REST, rtol=1e-6)


def test_left_system_carries_most_of_the_flow(m):
    left = m.rest.flow["LMCA"] / m.rest.total_inflow
    assert 0.55 <= left <= 0.75


def test_pressure_falls_monotonically_downstream(m):
    for sol in (m.rest, m.hyperaemic()):
        for sid, seg in m.tree.segments.items():
            if seg.parent_id in m.tree.segments:
                assert sol.p_distal[sid] <= sol.p_distal[seg.parent_id] + 1e-9


def test_whole_tree_flow_reserve_is_below_the_microvascular_reserve(m):
    cfr = m.hyperaemic().total_inflow / m.rest.total_inflow
    assert 3.0 <= cfr < C.CORONARY_FLOW_RESERVE


# ---------------------------------------------------------------------- FFR
def test_ffr_decreases_monotonically_with_severity(m):
    vals = [experiments.lesion_ffr(m, "LAD", d) for d in np.arange(0.1, 0.91, 0.1)]
    assert all(a > b for a, b in zip(vals, vals[1:], strict=False))
    assert 0.0 < vals[-1] < vals[0] < 1.0


def test_mild_stenosis_is_not_ischaemic(m):
    r = ffr.compute_ffr(m.tree, m.outlets, stenosis.Stenosis("LAD", 0.30))
    assert r.ffr > 0.80 and not r.ischemic


def test_proximal_threshold_crossings_are_clinically_plausible(m):
    """Proximal LAD and LCX lesions cross FFR 0.80 near the conventional 50%
    angiographic threshold; left main lesions cross earlier."""
    cross = experiments.ffr_crossings(m).set_index("vessel").ds_at_threshold_pct
    assert 45.0 <= cross["LAD"] <= 60.0
    assert 45.0 <= cross["LCX"] <= 60.0
    assert cross["LMCA"] < cross["LAD"]


def test_separation_term_lowers_ffr(m):
    full = experiments.lesion_ffr(m, "LAD", 0.65)
    viscous = experiments.lesion_ffr(m, "LAD", 0.65, include_separation=False)
    assert viscous > full


# ----------------------------------------------------------------- pulsatile
def test_flow_is_diastolic_dominant(puls):
    for sid in ("LAD", "LCX", "RCA"):
        assert puls.diastolic_systolic_ratio(sid) > 1.2
        assert 0.70 <= puls.diastolic_flow_fraction(sid) <= 0.85


def test_removing_intramyocardial_pump_abolishes_dominance(m):
    ab = experiments.pulsatile_ablations(m)
    assert ab["lad_dsvr_no_pump"] < 1.0 < ab["lad_dsvr"]


def test_flow_never_reverses_at_rest(puls):
    for q in puls.flow.values():
        assert np.min(q) > 0.0


# ----------------------------------------------------- validation outcome
def test_validation_outcome_is_unchanged(m):
    """Six of the ten predictions agree with measurement; these four do not."""
    val = experiments.validation_table(m)
    pred = val[val.role == "predicted"]
    failing = set(pred.loc[~pred.within_2sd_or_band, "quantity"])
    assert failing == {"Wave speed, proximal LAD",
                       "TAWSS, apical LAD",
                       "Hyperaemic gradient to distal LAD",
                       "Hyperaemic gradient to distal RCA"}


def test_solution_is_deterministic(m):
    again = model.build()
    for sid in m.tree.segments:
        assert again.rest.flow[sid] == m.rest.flow[sid]
