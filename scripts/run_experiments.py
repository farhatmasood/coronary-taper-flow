"""Run every experiment on the reference coronary tree and write the results.

    python scripts/run_experiments.py
    python scripts/run_experiments.py --noise-draws 20 --cohort-size 20   # quick check

Writes
    results/*.csv            one table per experiment
    results/summary.json     model-wide scalars (flows, FFR crossings, pulsatile
                             and rheology ablations, numerics, volumetrics)
    results/timing.json      wall-clock timings (machine dependent)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import coronary
from coronary import experiments as E
from coronary import model as M

log = logging.getLogger("experiments")


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.floating | float):
        return None if math.isnan(obj) else round(float(obj), 10)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=REPO / "results")
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--noise-draws", type=int, default=200,
                    help="Monte Carlo draws per noise level (default 200)")
    ap.add_argument("--cohort-size", type=int, default=200,
                    help="anatomies in the virtual cohort (default 200)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)

    m = M.build()
    log.info("calibrated resting flow %.1f mL/min", m.rest.total_inflow_ml_min)

    seg = E.segment_table(m)
    cross = E.ffr_crossings(m)
    val = E.validation_table(m)
    sens = E.sensitivity()
    q197 = E.flow_scenario(197.0)
    vel = {c: E.velocity_profile(c) for c in ("taper", "segment", "leaf")}
    noise = E.radius_noise(sigmas=(0.05, 0.1, 0.15, 0.2), n=args.noise_draws)
    hyper = m.hyperaemic()

    tables = {
        "segments": seg,
        "closure_comparison": E.closure_comparison(),
        "segmentation_invariance": E.segmentation_invariance(),
        "ffr_curves": E.ffr_curves(m),
        "ffr_crossings": cross,
        "separation_ablation": E.separation_ablation(m),
        "waveforms": E.waveform_table(m),
        "validation": val,
        "sensitivity": sens,
        "murray_exponent": E.murray_exponent_ablation(),
        "flow_scenario_197": q197,
        "radius_scaling": E.radius_scaling(),
        "flow_at_risk": E.flow_at_risk(m),
        "velocity_summary": E.velocity_summary(),
        "velocity_profiles": pd.concat(vel.values(), ignore_index=True),
        "scar_scenario": E.scar_scenario(),
        "bed_shares": E.bed_shares(),
        "proximal_distal_ffr": E.proximal_distal_ffr(),
        "closure_table": E.closure_table(),
        "radius_noise": noise,
        "radius_noise_summary": E.noise_summary(noise),
        "virtual_cohort": E.virtual_cohort(n=args.cohort_size),
        "gradient_decomposition": E.gradient_decomposition(),
        "microvascular_dysfunction": E.microvascular_dysfunction(),
        "serial_lesions": E.serial_lesions(),
    }
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.6g")

    summary = dict(
        version=coronary.__version__,
        n_segments=len(m.tree),
        flow_rest_total_ml_min=m.rest.total_inflow_ml_min,
        flow_hyper_total_ml_min=hyper.total_inflow_ml_min,
        flow_reserve_total=hyper.total_inflow / m.rest.total_inflow,
        left_share_pct=100 * m.rest.flow["LMCA"] / m.rest.total_inflow,
        ffr_crossing_pct=dict(zip(cross.vessel, cross.ds_at_threshold_pct, strict=True)),
        ffr_crossing_pct_q197=dict(zip(q197.vessel, q197.ds_at_threshold_pct, strict=True)),
        sensitivity_baseline_ffr=sens.attrs["baseline_ffr"],
        validation_agree=int(val[val.role == "predicted"].within_2sd_or_band.sum()),
        validation_predicted=int((val.role == "predicted").sum()),
        pulsatile=E.pulsatile_ablations(m),
        rheology=E.rheology_ablation(m),
        numerics=E.numerics(m),
        volumetrics=E.volumetrics(m),
    )
    (out / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))

    if not args.skip_timing:
        timing = E.timing()
        timing.update(python=platform.python_version(), numpy=np.__version__,
                      processor=platform.processor() or platform.machine(),
                      system=f"{platform.system()} {platform.release()}")
        (out / "timing.json").write_text(json.dumps(jsonable(timing), indent=2))

    log.info("validation: %d of %d predicted quantities within 2 SD or band",
             summary["validation_agree"], summary["validation_predicted"])
    log.info("results written to %s", out)


if __name__ == "__main__":
    main()
