"""Clinical consequences of the outflow law.

Hyperaemic velocity along the LAD, territory shares, and the FFR of a
proximal lesion when the distal territory is scarred.
"""

from __future__ import annotations

import numpy as np
import pytest

from coronary import experiments as E


@pytest.fixture(scope="module")
def velocity():
    return E.velocity_summary().set_index(["closure", "k"])


def test_taper_closure_keeps_hyperaemic_velocity_near_the_population_value(velocity):
    """Fixed-flow QFR assumes 0.35 m/s; the taper-resolved LAD stays near it."""
    row = velocity.loc[("taper", 2.39)]
    assert row.v_min <= E.FIXED_FLOW_QFR_VELOCITY <= row.v_max
    assert row.v_max / row.v_min < 1.6


def test_exponent_two_gives_nearly_uniform_velocity(velocity):
    """Q ~ r^2 means v = Q / (pi r^2) is constant along a tapering conduit."""
    row = velocity.loc[("taper", 2.0)]
    assert row.v_max / row.v_min < 1.15


def test_leaf_outlets_drive_apical_velocity_far_above_the_taper_closure(velocity):
    assert velocity.loc[("leaf", 2.39), "v_apex"] > 3 * velocity.loc[("taper", 2.39), "v_apex"]


def test_downstream_bed_shares_sum_to_the_segment_flow():
    shares = E.bed_shares("LAD")
    for _, g in shares.groupby("closure"):
        assert np.isclose(g.share_pct.sum(), 100.0, atol=1e-6)


@pytest.fixture(scope="module")
def scar():
    return E.scar_scenario(fractions=np.linspace(0.0, 1.0, 6))


def test_ffr_rises_as_downstream_myocardium_is_lost(scar):
    """For the same stenosis, less viable myocardium means higher FFR."""
    for _, g in scar.groupby("closure"):
        for col in ("ffr_50", "ffr_70"):
            assert np.all(np.diff(g[col].to_numpy()) > 0)


def test_leaf_outlets_lose_the_whole_lad_with_an_apical_scar(scar):
    full = scar[(scar.closure == "leaf") & (scar.scar_fraction == 1.0)].iloc[0]
    assert full.flow_hyper_ml_min < 1.0
    assert full.ffr_70 > 0.95


def test_taper_closure_keeps_the_viable_proximal_territory_at_risk(scar):
    """A 70% proximal LAD lesion stays ischaemic after a distal infarct."""
    full = scar[(scar.closure == "taper") & (scar.scar_fraction == 1.0)].iloc[0]
    assert full.viable_share_pct > 50.0
    assert full.ffr_70 < 0.80


def test_scar_context_restores_the_model():
    m = E.build()
    before = {b: m.outlets[b].r_term_rest for b in E.SCAR_BEDS}
    with E.scarred(m, E.SCAR_BEDS, 0.7):
        pass
    assert all(m.outlets[b].r_term_rest == before[b] for b in E.SCAR_BEDS)
