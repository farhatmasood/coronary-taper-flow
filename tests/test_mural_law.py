"""The morphometric outflow law: conservation, invariance and limits."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from coronary import constants as C
from coronary import coronary_bc, experiments, geometry

TREE = geometry.load_default_tree()


@pytest.mark.parametrize("closure", coronary_bc.ALL_CLOSURES)
def test_outlet_weights_sum_to_one(closure):
    _, w = coronary_bc.murray_flow_weights(TREE, closure=closure)
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert all(v >= -1e-15 for v in w.values()), "no bed may return flow to the tree"


def test_unknown_closure_is_rejected():
    with pytest.raises(ValueError):
        coronary_bc.murray_flow_weights(TREE, closure="nonsense")


def test_leaf_closure_drains_only_at_terminals():
    _, w = coronary_bc.murray_flow_weights(TREE, closure="leaf")
    for sid, v in w.items():
        if not TREE[sid].is_terminal:
            assert v == 0.0


def test_discrete_rule_is_the_exact_integral_of_the_local_law():
    """dQ/dx = k Q r'/r integrated numerically along a linear taper must give
    Q_d/Q_p = (r_d/r_p)^k for every segment."""
    k = C.MURRAY_EXPONENT
    for seg in TREE:
        rp, rd, length = seg.radius_prox, seg.radius_dist, seg.length
        drdx = (rd - rp) / length

        def rhs(x, q, rp=rp, drdx=drdx):
            return k * q / (rp + drdx * x) * drdx

        sol = solve_ivp(rhs, (0.0, length), [1.0], rtol=1e-11, atol=1e-14)
        assert np.isclose(sol.y[0, -1], (rd / rp) ** k, rtol=1e-8), seg.seg_id


def test_taper_closure_is_invariant_to_segmentation():
    """Merging LAD1-LAD3 into one segment must not change the flow reaching
    the apical LAD; the within-segment variant is not invariant."""
    inv = experiments.segmentation_invariance().set_index("closure")
    assert inv.loc["taper", "relative_difference"] < 1e-12
    assert inv.loc["segment", "relative_difference"] > 0.05


def test_chain_pass_through_telescopes_to_end_radii():
    """Along LAD1..LAD4, whose radius never increases, the flow reaching the
    apex is (r_apex / r_LAD1)^k of the flow entering LAD1."""
    seg_flow, _ = coronary_bc.murray_flow_weights(TREE, closure="taper")
    ratio = seg_flow["LAD4"] / seg_flow["LAD1"]
    expect_entry = (TREE["LAD4"].radius_prox / TREE["LAD1"].radius_prox) ** C.MURRAY_EXPONENT
    assert np.isclose(ratio, expect_entry, rtol=1e-12)


def test_no_flow_is_created_where_the_lumen_widens():
    """LAD ends at 1.45 mm and LAD1 starts at 1.46 mm: the junction passes
    everything on and sheds nothing, it does not amplify the flow."""
    seg_flow, w = coronary_bc.murray_flow_weights(TREE, closure="taper")
    lad = TREE["LAD"]
    mural = seg_flow["LAD"] * (1 - (lad.radius_dist / lad.radius_prox) ** C.MURRAY_EXPONENT)
    assert np.isclose(w["LAD"], mural, rtol=1e-12)
    assert seg_flow["LAD1"] <= seg_flow["LAD"]


def test_flow_never_increases_along_the_lad(m):
    flows = [m.rest.flow[s] for s in experiments.LAD_PATH[1:]]
    assert all(a >= b for a, b in zip(flows, flows[1:], strict=False))


def test_taper_closure_reduces_apical_shear_relative_to_leaf_outlets():
    cc = experiments.closure_comparison().set_index(["closure", "segment"])
    taper = cc.loc[("taper", "LAD4"), "tawss_pa"]
    leaf = cc.loc[("leaf", "LAD4"), "tawss_pa"]
    assert taper < 0.5 * leaf
    # at the upper edge of the 1-2.5 Pa intermediate band (2.53 Pa), far
    # from the leaf-outlet value and not in the low-shear range
    assert 1.0 <= taper <= 2.6
