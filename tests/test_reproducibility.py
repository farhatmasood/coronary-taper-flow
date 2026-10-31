"""The committed results must match a fresh run of the code.

If one of these fails, the code changed without ``scripts/run_experiments.py``
being re-run. Re-run it and commit the refreshed ``results/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coronary import experiments

RESULTS = Path(__file__).resolve().parents[1] / "results"
pytestmark = pytest.mark.skipif(not (RESULTS / "summary.json").exists(),
                                reason="results/ not generated yet")


def test_summary_matches_a_fresh_solve(m):
    stored = json.loads((RESULTS / "summary.json").read_text())
    hyper = m.hyperaemic()
    assert np.isclose(stored["flow_rest_total_ml_min"], m.rest.total_inflow_ml_min, rtol=1e-6)
    assert np.isclose(stored["flow_hyper_total_ml_min"], hyper.total_inflow_ml_min, rtol=1e-6)
    fresh = experiments.ffr_crossings(m).set_index("vessel").ds_at_threshold_pct
    for vessel, value in stored["ffr_crossing_pct"].items():
        assert np.isclose(value, fresh[vessel], atol=1e-3), vessel


def test_segment_table_matches_a_fresh_solve(m):
    stored = pd.read_csv(RESULTS / "segments.csv").set_index("segment")
    fresh = experiments.segment_table(m).set_index("segment")
    for col in ("flow_rest_ml_min", "tawss_pa", "pd_pa_hyper", "dsvr"):
        assert np.allclose(stored[col], fresh.loc[stored.index, col], rtol=1e-4), col
