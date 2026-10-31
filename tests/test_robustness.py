"""Comparator closures and robustness analyses."""

from __future__ import annotations

import numpy as np
import pytest

from coronary import constants as C
from coronary import coronary_bc, experiments, geometry, model


# ------------------------------------------------------- comparator closures
def test_structured_tree_resistance_scales_as_a_power_of_radius():
    radii = np.array([0.5e-3, 1.0e-3, 2.0e-3])
    r = np.array([coronary_bc.structured_tree_resistance(x) for x in radii])
    slopes = np.diff(np.log(r)) / np.diff(np.log(radii))
    assert np.all((slopes < -2.3) & (slopes > -3.0))


def test_structured_tree_split_does_not_depend_on_vessel_length_ratio():
    a = [coronary_bc.structured_tree_resistance(x, lrr=50.0) for x in (0.8e-3, 1.2e-3)]
    b = [coronary_bc.structured_tree_resistance(x, lrr=20.0) for x in (0.8e-3, 1.2e-3)]
    assert np.isclose(a[0] / a[1], b[0] / b[1], rtol=1e-12)


def test_junction_closure_sheds_only_at_radius_steps():
    tree = geometry.load_default_tree()
    _, w = coronary_bc.murray_flow_weights(tree, closure="junction")
    # LAD ends at 1.45 mm and LAD1 starts at 1.46 mm: no step, no outflow,
    # whatever the taper of LAD. LAD3 starts narrower than LAD2 ends: outflow.
    assert tree["LAD1"].radius_prox >= tree["LAD"].radius_dist
    assert w["LAD"] == 0.0
    assert tree["LAD3"].radius_prox < tree["LAD2"].radius_dist
    assert w["LAD2"] > 0.0


def test_bed_reserve_overrides_the_global_reserve():
    m = model.build()
    outlet = m.outlets["LAD4"]
    assert outlet.resistance(True) == pytest.approx(
        outlet.r_term_rest / C.CORONARY_FLOW_RESERVE)
    outlet.reserve = 2.0
    assert outlet.resistance(True) == pytest.approx(outlet.r_term_rest / 2.0)
    outlet.reserve = None


def test_closure_table_orders_the_closures_as_reported():
    t = experiments.closure_table().set_index("closure")
    assert t.loc["taper", "apex_share_pct"] < t.loc["segment", "apex_share_pct"] \
        < t.loc["junction", "apex_share_pct"] < t.loc["leaf", "apex_share_pct"]
    # structured-tree outlets at named terminals starve the LAD
    assert t.loc["structured", "lad_flow_ml_min"] < 0.5 * t.loc["taper", "lad_flow_ml_min"]


# ------------------------------------------------ monotone reference fit
def test_isotonic_fit_is_non_increasing_and_mean_preserving():
    values = [3.0, 2.8, 2.9, 2.5, 2.6, 2.0]
    fit = geometry.isotonic_nonincreasing(values)
    assert all(a >= b for a, b in zip(fit, fit[1:], strict=False))
    assert np.isclose(sum(fit), sum(values))


def test_reference_fit_is_monotone_along_every_chain():
    rng = np.random.default_rng(1)
    ref = geometry.monotone_reference(experiments.noisy_tree(0.2, rng))
    for seg in ref:
        assert seg.radius_dist <= seg.radius_prox + 1e-15
        if seg.parent_id in ref.segments and len(ref[seg.parent_id].children) == 1:
            assert seg.radius_prox <= ref[seg.parent_id].radius_dist + 1e-15


def test_reference_fit_removes_the_noise_bias_of_the_outflow_law():
    noise = experiments.radius_noise(sigmas=(0.2,), n=30, closures=("taper",))
    s = experiments.noise_summary(noise).set_index(["input", "mode"])
    raw = s.loc[("raw", "full"), "apex_share_pct_bias"]
    ref = s.loc[("reference", "full"), "apex_share_pct_bias"]
    assert raw < -5.0
    assert abs(ref) < abs(raw) / 3


# ------------------------------------------------ input calibration and physiology
def test_morphometric_calibration_brings_the_distal_gradient_into_range():
    g = experiments.gradient_decomposition().set_index(["configuration", "closure"])
    cal = g.loc[("both (calibrated)", "taper"), "gradient_distal_lad_mmhg"]
    ref = g.loc[("reference", "taper"), "gradient_distal_lad_mmhg"]
    assert cal < ref
    assert cal <= 3.0 + 2 * 3.0           # within two SD of normal arteries (3 +/- 3 mmHg)


def test_blunted_microvascular_reserve_raises_ffr():
    md = experiments.microvascular_dysfunction(reserves=(4.0, 2.0, 1.0))
    for _, g in md.groupby(["closure", "territory"]):
        assert np.all(np.diff(g.ffr_50.to_numpy()) > 0)


def test_serial_lesions_interact():
    s = experiments.serial_lesions().set_index("closure")
    for closure in ("taper", "leaf"):
        # treating one lesion gains less than its apparent drop suggests
        assert s.loc[closure, "true_delta_a"] < s.loc[closure, "apparent_delta_a"]
    assert s.loc["taper", "true_delta_a"] > s.loc["leaf", "true_delta_a"]


def test_clinical_directions_hold_across_a_virtual_cohort():
    coh = experiments.virtual_cohort(n=12)
    w = coh.pivot(index="anatomy", columns="closure", values=["ffr_dist_50", "ffr_scar_70"])
    assert (w[("ffr_dist_50", "taper")] > w[("ffr_dist_50", "leaf")]).mean() >= 0.9
    assert (w[("ffr_scar_70", "taper")] < w[("ffr_scar_70", "leaf")]).mean() >= 0.9
