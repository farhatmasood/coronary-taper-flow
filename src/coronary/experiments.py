"""Experiments on the coronary model, as plain functions.

Each function builds what it needs through :func:`coronary.model.build` and
returns a pandas DataFrame or a dict of scalars. Nothing here writes files;
``scripts/run_experiments.py`` saves the returned tables under ``results/``,
and the test suite calls the same functions.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import constants as C
from . import coronary_bc, ffr, geometry, pulsatile_solver, steady_solver, waveforms, wss
from .geometry import CoronaryTree, Segment, monotone_reference
from .geometry import copy_tree as _copy_tree
from .model import Model, build
from .stenosis import Stenosis, pressure_drop

REPO = Path(__file__).resolve().parents[2]


def _reference_tree() -> CoronaryTree:
    """The default tree as the model uses it: its monotone reference fit."""
    return monotone_reference(geometry.load_default_tree())
REFERENCE_CSV = REPO / "data" / "reference_measurements.csv"

LAD_PATH = ["LMCA", "LAD", "LAD1", "LAD2", "LAD3", "LAD4"]
CARDINAL = ["LMCA", "LAD", "LCX", "RCA"]
FFR_THRESHOLD = ffr.FFR_ISCHEMIA_THRESHOLD


# ==========================================================================
# helpers
# ==========================================================================
@contextmanager
def overridden(**values) -> Iterator[None]:
    """Temporarily replace entries of :mod:`coronary.constants`."""
    saved = {k: getattr(C, k) for k in values}
    try:
        for k, v in values.items():
            setattr(C, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(C, k, v)


def lesion_ffr(m: Model, seg_id: str, ds: float, **kwargs) -> float:
    """FFR of a single focal lesion of diameter stenosis ``ds``."""
    return ffr.compute_ffr(m.tree, m.outlets, Stenosis(seg_id, ds, **kwargs)).ffr


def ffr_crossing(m: Model, seg_id: str, level: float = FFR_THRESHOLD) -> float:
    """Diameter stenosis (fraction) at which a lesion on ``seg_id`` gives FFR = level.

    FFR falls monotonically with severity, so the root is bracketed and found
    by Brent's method. Returns 0 when the lesion-free vessel is already below
    ``level`` at the lesion site.
    """
    f = lambda ds: lesion_ffr(m, seg_id, ds) - level  # noqa: E731
    lo, hi = 1e-3, 0.95
    if f(lo) <= 0.0:
        return 0.0
    return float(brentq(f, lo, hi, xtol=1e-6))


def path_distance(tree: CoronaryTree, path: list[str]) -> list[float]:
    """Cumulative distance (m) from the ostium to the distal end of each segment."""
    return list(np.cumsum([tree[s].length for s in path]))


def tawss(m: Model, puls: pulsatile_solver.PulsatileSolution, sid: str) -> float:
    tau = wss.wall_shear_stress(puls.flow[sid], m.tree[sid].radius_mean)
    return float(wss.tawss(tau, puls.time))


@contextmanager
def scarred(m: Model, beds: list[str], fraction: float) -> Iterator[None]:
    """Temporarily replace a fraction of the myocardium of ``beds`` by scar.

    Scar has no microvasculature, so a bed in which a fraction ``fraction`` of
    the myocardium is scarred keeps (1 - fraction) of its conductance, at rest
    and at hyperaemia. The viable beds and the calibration are left unchanged,
    so total coronary flow falls with the viable mass. A fully scarred bed
    (fraction = 1) is given a negligible conductance.
    """
    keep = max(1.0 - fraction, 1e-9)
    saved = {b: m.outlets[b].r_term_rest for b in beds if b in m.outlets}
    try:
        for b, r in saved.items():
            m.outlets[b].r_term_rest = r / keep
        yield
    finally:
        for b, r in saved.items():
            m.outlets[b].r_term_rest = r


# ==========================================================================
# baseline solution, one row per segment
# ==========================================================================
def segment_table(m: Model) -> pd.DataFrame:
    """Resting and lesion-free hyperaemic state of every segment."""
    puls = pulsatile_solver.solve_pulsatile(m.tree, m.outlets, m.rest)
    hyper = m.hyperaemic()
    rows = []
    for sid in m.tree.ordered_ids():
        seg = m.tree[sid]
        p_up = (C.P_AO_MEAN if seg.parent_id not in m.tree.segments
                else m.rest.p_distal[seg.parent_id])
        q_ml_s = m.rest.flow[sid] * 1e6
        rows.append(dict(
            segment=sid, clinical_name=seg.clinical_name, side=seg.side,
            length_mm=seg.length * 1e3,
            r_prox_mm=seg.radius_prox * 1e3, r_dist_mm=seg.radius_dist * 1e3,
            diameter_mm=2e3 * seg.radius_mean,
            flow_rest_ml_min=m.rest.flow_ml_min(sid),
            flow_share_pct=100 * m.rest.flow[sid] / m.rest.total_inflow,
            outlet_share_pct=(100 * m.outlets[sid].flow_weight if sid in m.outlets else 0.0),
            dp_rest_mmhg=(p_up - m.rest.p_distal[sid]) * C.PA_TO_MMHG,
            flow_hyper_ml_min=hyper.flow_ml_min(sid),
            pd_pa_hyper=hyper.p_distal[sid] / hyper.p_inlet,
            tawss_pa=tawss(m, puls, sid),
            osi=wss.osi(wss.wall_shear_stress(puls.flow[sid], seg.radius_mean), puls.time),
            wave_speed_ms=puls.wave_speed[sid],
            diastolic_fraction=puls.diastolic_flow_fraction(sid),
            dsvr=puls.diastolic_systolic_ratio(sid),
            volume_ml=seg.volume * 1e6,
            transit_time_s=seg.volume * 1e6 / q_ml_s if q_ml_s > 0 else np.nan,
        ))
    return pd.DataFrame(rows)


# ==========================================================================
# the effect of the outflow closure along the LAD
# ==========================================================================
def closure_comparison(ds: float = 0.50) -> pd.DataFrame:
    """Flow, shear, lesion-free hyperaemic pressure and lesion FFR along the
    left main to apical LAD path for each outflow closure."""
    rows = []
    for closure in coronary_bc.CLOSURES:
        m = build(closure)
        puls = pulsatile_solver.solve_pulsatile(m.tree, m.outlets, m.rest)
        hyper = m.hyperaemic()
        dist = path_distance(m.tree, LAD_PATH)
        for sid, x in zip(LAD_PATH, dist, strict=True):
            rows.append(dict(
                closure=closure, segment=sid,
                x_start_mm=(x - m.tree[sid].length) * 1e3, x_end_mm=x * 1e3,
                flow_rest_ml_min=m.rest.flow_ml_min(sid),
                tawss_pa=tawss(m, puls, sid),
                pd_pa_hyper=hyper.p_distal[sid] / hyper.p_inlet,
                ffr_lesion=lesion_ffr(m, sid, ds),
                ds_at_threshold_pct=100 * ffr_crossing(m, sid),
            ))
    return pd.DataFrame(rows)


def _merge_chain(tree: CoronaryTree, chain: list[str], new_id: str) -> CoronaryTree:
    """Replace a single-child chain of segments by one linearly tapered segment."""
    first, last = tree[chain[0]], tree[chain[-1]]
    segs = {sid: Segment(**{f: getattr(s, f) for f in (
        "seg_id", "name", "clinical_name", "side", "parent_id", "length",
        "radius_prox", "radius_dist", "is_terminal")})
        for sid, s in tree.segments.items() if sid not in chain}
    segs[new_id] = Segment(
        seg_id=new_id, name=new_id, clinical_name=new_id, side=first.side,
        parent_id=first.parent_id, length=sum(tree[s].length for s in chain),
        radius_prox=first.radius_prox, radius_dist=last.radius_dist,
        is_terminal=last.is_terminal)
    for sid in last.children:
        segs[sid].parent_id = new_id
    return CoronaryTree(segs)


def segmentation_invariance() -> pd.DataFrame:
    """Flow fraction reaching the apical LAD when LAD1-LAD3 are split into
    three named segments or merged into one, for each closure."""
    tree = _reference_tree()
    merged = _merge_chain(tree, ["LAD1", "LAD2", "LAD3"], "LAD123")
    rows = []
    for closure in coronary_bc.CLOSURES:
        split, _ = coronary_bc.murray_flow_weights(tree, closure=closure)
        joined, _ = coronary_bc.murray_flow_weights(merged, closure=closure)
        rows.append(dict(closure=closure,
                         apical_share_split=split["LAD4"],
                         apical_share_merged=joined["LAD4"],
                         relative_difference=abs(split["LAD4"] / joined["LAD4"] - 1.0)))
    return pd.DataFrame(rows)


# ==========================================================================
# clinical implications of the closure
# ==========================================================================
FIXED_FLOW_QFR_VELOCITY = 0.35   # m/s, population hyperaemic velocity of fixed-flow QFR


def velocity_profile(closure: str = "taper", murray_exponent: float | None = None,
                     path: list[str] = LAD_PATH[1:], n_per_segment: int = 12) -> pd.DataFrame:
    """Lesion-free hyperaemic mean velocity along a path, v(x) = Q(x) / (pi r(x)^2).

    Q at each segment inlet is the solved hyperaemic flow. Inside a segment
    the outflow law sheds flow continuously, Q(x) = Q_in (r(x)/r_p)^k; the
    leaf-outlet closure carries Q_in unchanged. Distances are measured from
    the coronary ostium; the start and end of every segment are both listed,
    so steps at junctions are kept.
    """
    m = build(closure, murray_exponent)
    k = m.murray_exponent
    hyper = m.hyperaemic()
    x0 = sum(m.tree[s].length for s in m.tree.path_to_root(path[0])[1:])
    rows = []
    for sid in path:
        seg = m.tree[sid]
        for xi in np.linspace(0.0, 1.0, n_per_segment):
            r = seg.radius_at(xi)
            sheds_along = closure in ("taper", "segment")
            shed = (r / seg.radius_prox) ** k if sheds_along else 1.0
            q = hyper.flow[sid] * shed
            rows.append(dict(closure=closure, k=k, segment=sid,
                             x_mm=(x0 + xi * seg.length) * 1e3,
                             velocity_ms=q / (np.pi * r * r)))
        x0 += seg.length
    return pd.DataFrame(rows)


def velocity_summary() -> pd.DataFrame:
    """Range of hyperaemic velocity along the LAD for each closure, and for
    the taper law with k = 2 (constant velocity)."""
    rows = []
    for closure, k in [("taper", None), ("taper", 2.0), ("segment", None), ("leaf", None)]:
        v = velocity_profile(closure, k).velocity_ms
        rows.append(dict(closure=closure, k=k if k else C.MURRAY_EXPONENT,
                         v_min=float(v.min()), v_max=float(v.max()),
                         v_apex=float(v.iloc[-1])))
    return pd.DataFrame(rows)


SCAR_BEDS = ["LAD3", "LAD4"]            # distal LAD and apical territory


def scar_scenario(beds: list[str] = SCAR_BEDS, lesion_segment: str = "LAD",
                  severities=(0.50, 0.70), fractions=None,
                  closures=("taper", "leaf")) -> pd.DataFrame:
    """FFR of a proximal lesion when part of the downstream myocardium is scar.

    ``fractions`` is the scarred fraction of the myocardium drained by
    ``beds``. Also returned: lesion-free hyperaemic flow of the lesion segment
    and the share of its resting flow that perfuses viable myocardium.
    """
    fractions = np.linspace(0.0, 1.0, 11) if fractions is None else fractions
    rows = []
    for closure in closures:
        m = build(closure)
        shares = _bed_flow_shares(m, lesion_segment)
        scar_share = sum(shares.get(b, 0.0) for b in beds)
        for f in fractions:
            with scarred(m, beds, float(f)):
                row = dict(closure=closure, scar_fraction=float(f),
                           flow_hyper_ml_min=m.hyperaemic().flow_ml_min(lesion_segment))
                for ds in severities:
                    row[f"ffr_{int(round(100 * ds))}"] = lesion_ffr(m, lesion_segment, ds)
            # share of the segment's (unscarred) resting flow that perfuses viable myocardium
            row["viable_share_pct"] = 100 * (1.0 - f * scar_share)
            rows.append(row)
    return pd.DataFrame(rows)


def _downstream_beds(m: Model, seg_id: str) -> set[str]:
    """Beds at or distal to the distal node of ``seg_id`` (itself included)."""
    out, stack = set(), [seg_id]
    while stack:
        s = stack.pop()
        out.add(s)
        stack.extend(m.tree[s].children)
    return out


def _bed_flow_shares(m: Model, seg_id: str) -> dict[str, float]:
    """Solved resting outflow of each bed downstream of ``seg_id`` as a
    fraction of the segment's inflow; the fractions sum to one."""
    q = m.rest.flow[seg_id]
    return {b: m.outlets[b].steady_flow(m.rest.p_distal[b], False) / q
            for b in _downstream_beds(m, seg_id) if b in m.outlets}


def bed_shares(seg_id: str = "LAD") -> pd.DataFrame:
    """Share of a segment's resting inflow drained by each downstream bed."""
    rows = []
    for closure in coronary_bc.CLOSURES:
        m = build(closure)
        for b, share in sorted(_bed_flow_shares(m, seg_id).items()):
            rows.append(dict(closure=closure, bed=b, share_pct=100 * share))
    return pd.DataFrame(rows)


def proximal_distal_ffr(segments=("LAD", "LAD3"), grid: np.ndarray | None = None) -> pd.DataFrame:
    """FFR against diameter stenosis for a proximal and a distal LAD lesion,
    taper-resolved and leaf-outlet closures side by side."""
    grid = np.arange(0.10, 0.9001, 0.025) if grid is None else grid
    rows = []
    for closure in ("taper", "leaf"):
        m = build(closure)
        for sid in segments:
            for ds in grid:
                rows.append(dict(closure=closure, segment=sid, ds_pct=100 * ds,
                                 ffr=lesion_ffr(m, sid, float(ds))))
    return pd.DataFrame(rows)


# ==========================================================================
# comparison with other outflow closures, and robustness
# ==========================================================================
def _apex_and_ffr(m: Model) -> dict[str, float]:
    """Headline quantities of one model along the LAD."""
    lad, apex = m.rest.flow["LAD"], m.rest.flow["LAD4"]
    seg = m.tree["LAD4"]
    return dict(
        apex_share_pct=100 * apex / lad,
        apex_wss_pa=float(wss.wall_shear_stress(apex, seg.radius_mean)),
        ffr_prox_50=lesion_ffr(m, "LAD", 0.50),
        ffr_dist_50=lesion_ffr(m, "LAD3", 0.50),
    )


def closure_table() -> pd.DataFrame:
    """All five outflow closures side by side on the reference tree."""
    rows = []
    for closure in coronary_bc.ALL_CLOSURES:
        m = build(closure)
        v = velocity_profile(closure).velocity_ms
        rows.append(dict(closure=closure, lad_flow_ml_min=m.rest.flow_ml_min("LAD"),
                         **_apex_and_ffr(m), v_max=float(v.max()),
                         ffr_scar_70=_scar_ffr(m, 1.0, 0.70)))
    return pd.DataFrame(rows)


def _scar_ffr(m: Model, fraction: float, ds: float) -> float:
    with scarred(m, SCAR_BEDS, fraction):
        return lesion_ffr(m, "LAD", ds)


def exponent_band(values=(2.24, 2.54)) -> dict[float, dict[str, pd.DataFrame]]:
    """Taper-resolved profiles along the LAD at the ends of the 95 % CI of k."""
    out = {}
    for k in values:
        m = build("taper", murray_exponent=k)
        puls = pulsatile_solver.solve_pulsatile(m.tree, m.outlets, m.rest)
        dist = path_distance(m.tree, LAD_PATH)
        steps = pd.DataFrame([dict(segment=s, x_start_mm=(x - m.tree[s].length) * 1e3,
                                   x_end_mm=x * 1e3, flow_rest_ml_min=m.rest.flow_ml_min(s),
                                   tawss_pa=tawss(m, puls, s), ffr_lesion=lesion_ffr(m, s, 0.5))
                              for s, x in zip(LAD_PATH, dist, strict=True)])
        out[k] = dict(steps=steps, velocity=velocity_profile("taper", k))
    return out


def noisy_tree(sigma_mm: float, rng: np.random.Generator,
               base: CoronaryTree | None = None, r_floor_mm: float = 0.3) -> CoronaryTree:
    """Reference tree with independent Gaussian errors on every end radius,
    as from edge detection on angiography or CT (sigma in mm)."""
    tree = _copy_tree(_reference_tree() if base is None else base)
    for seg in tree:
        seg.radius_prox = max(seg.radius_prox + rng.normal(0.0, sigma_mm) * 1e-3,
                              r_floor_mm * 1e-3)
        seg.radius_dist = max(seg.radius_dist + rng.normal(0.0, sigma_mm) * 1e-3,
                              r_floor_mm * 1e-3)
    return tree


def _model_with_closure_from(tree: CoronaryTree, weights_tree: CoronaryTree,
                             closure: str) -> Model:
    """Model on ``tree`` whose outflow shares are computed from ``weights_tree``."""
    outlets = coronary_bc.build_outlets(weights_tree, None, closure)
    steady_solver.calibrate(tree, outlets)
    rest = steady_solver.solve_steady(tree, outlets, hyperemic=False)
    return Model(tree=tree, outlets=outlets, rest=rest, closure=closure,
                 murray_exponent=C.MURRAY_EXPONENT)


def radius_noise(sigmas=(0.1, 0.2), n: int = 200, closures=("taper", "leaf"),
                 seed: int = 2027) -> pd.DataFrame:
    """Monte Carlo propagation of radius measurement error.

    mode 'full': every radius used anywhere is noisy (resistances and outflow
    shares). mode 'closure': only the outflow shares see the noisy radii, the
    conduit resistances use the true ones, which isolates the error added by
    the outflow law itself. ``input`` is 'raw' (noisy edge radii used as they
    are) or 'reference' (a monotone reference diameter fit of the noisy radii,
    :func:`geometry.monotone_reference`, the default first step of
    :func:`model.build`). The same draws are used in every setting.
    """
    truth = _reference_tree()
    rows = []
    for sigma in sigmas:
        rng = np.random.default_rng(seed)
        draws = [noisy_tree(sigma, rng) for _ in range(n)]
        for inp in ("raw", "reference"):
            for i, noisy in enumerate(draws):
                measured = noisy if inp == "raw" else monotone_reference(noisy)
                for mode in ("full", "closure"):
                    for closure in closures:
                        if mode == "full":
                            m = build(closure, tree=_copy_tree(measured), reference_fit=False)
                        else:
                            m = _model_with_closure_from(_copy_tree(truth), measured, closure)
                        rows.append(dict(sigma_mm=sigma, input=inp, mode=mode, draw=i,
                                         closure=closure, **_apex_and_ffr(m)))
    return pd.DataFrame(rows)


def noise_summary(noise: pd.DataFrame) -> pd.DataFrame:
    """Bias and spread of the noisy results relative to the noise-free model."""
    ref = {c: _apex_and_ffr(build(c)) for c in noise.closure.unique()}
    rows = []
    for (sigma, inp, mode, closure), g in noise.groupby(["sigma_mm", "input", "mode", "closure"]):
        row = dict(sigma_mm=sigma, input=inp, mode=mode, closure=closure)
        for q in ("apex_share_pct", "ffr_prox_50", "ffr_dist_50"):
            row[f"{q}_bias"] = float(g[q].mean() - ref[closure][q])
            row[f"{q}_sd"] = float(g[q].std())
        rows.append(row)
    return pd.DataFrame(rows)


def virtual_cohort(n: int = 200, seed: int = 7, cv_global: float = 0.10,
                   cv_segment: float = 0.10, cv_length: float = 0.15) -> pd.DataFrame:
    """In-silico cohort of anatomies around the reference tree.

    Each anatomy scales all radii by one factor (body size, CV ``cv_global``),
    each segment's radii by its own factor (keeping its taper, CV
    ``cv_segment``) and each length by its own factor (CV ``cv_length``).
    """
    base = _reference_tree()
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        tree = _copy_tree(base)
        g = max(rng.normal(1.0, cv_global), 0.5)
        for seg in tree:
            s = g * max(rng.normal(1.0, cv_segment), 0.5)
            seg.radius_prox *= s
            seg.radius_dist *= s
            seg.length *= max(rng.normal(1.0, cv_length), 0.5)
        for closure in ("taper", "leaf"):
            m = build(closure, tree=_copy_tree(tree))
            row = dict(anatomy=i, closure=closure, **_apex_and_ffr(m),
                       ffr_scar_70=_scar_ffr(m, 1.0, 0.70))
            h = m.hyperaemic()
            row["gradient_distal_lad_mmhg"] = (h.p_inlet - h.p_distal["LAD3"]) * C.PA_TO_MMHG
            rows.append(row)
    return pd.DataFrame(rows)


# Normal human lumen diameters (Dodge et al., Circulation 1992; mean, SD in mm)
# used to place the reference tree on normal morphometry.
DODGE_NORMAL_DIAMETER = {"LMCA": (4.5, 0.5), "LAD": (3.7, 0.4), "RCA": (3.9, 0.6)}


def morphometry_scale(tree: CoronaryTree | None = None) -> float:
    """Single radius scale that brings the left main, proximal LAD and
    proximal RCA diameters, on average, to the normal means of Dodge et al."""
    tree = _reference_tree() if tree is None else tree
    return float(np.mean([mean / (2e3 * tree[s].radius_prox)
                          for s, (mean, _) in DODGE_NORMAL_DIAMETER.items()]))


def gradient_decomposition(q_measured: float = 197.0) -> pd.DataFrame:
    """Lesion-free hyperaemic gradient to the distal LAD and RCA as the inputs
    move from the reference values to measured flow and normal morphometry,
    with the key clinical results under each configuration."""
    scale = morphometry_scale()
    configs = [("reference", 1.0, C.Q_CORONARY_REST_ML_MIN),
               ("resting flow at measured mean", 1.0, q_measured),
               ("radii at normal morphometry", scale, C.Q_CORONARY_REST_ML_MIN),
               ("both (calibrated)", scale, q_measured)]
    rows = []
    for name, s, q in configs:
        with overridden(Q_CORONARY_REST=q * C.ML_PER_MIN_TO_M3_PER_S):
            for closure in ("taper", "leaf"):
                m = build(closure, tree=scaled_tree(s))
                h = m.hyperaemic()
                lad = m.tree["LAD"]
                rows.append(dict(
                    configuration=name, radius_scale=s, q_rest_ml_min=q, closure=closure,
                    gradient_distal_lad_mmhg=(h.p_inlet - h.p_distal["LAD3"]) * C.PA_TO_MMHG,
                    gradient_distal_rca_mmhg=(h.p_inlet - h.p_distal["RCA"]) * C.PA_TO_MMHG,
                    q_hyper_total_ml_min=h.total_inflow_ml_min,
                    v_hyper_prox_lad=h.flow["LAD"] / (np.pi * lad.radius_prox ** 2),
                    ffr_scar_70=_scar_ffr(m, 1.0, 0.70),
                    **_apex_and_ffr(m)))
    return pd.DataFrame(rows)


def microvascular_dysfunction(reserves=(4.0, 3.0, 2.0, 1.5, 1.0),
                              ds_values=(0.50, 0.70)) -> pd.DataFrame:
    """FFR of a proximal LAD lesion as the microvascular resistance reserve of
    the LAD territory falls (all beds downstream of the proximal LAD), and as
    that of the distal-LAD and apical beds alone falls (regional)."""
    rows = []
    for closure in ("taper", "leaf"):
        m = build(closure)
        territories = {"LAD territory": sorted(_downstream_beds(m, "LAD") & set(m.outlets)),
                       "distal LAD and apex": [b for b in SCAR_BEDS if b in m.outlets]}
        for name, beds in territories.items():
            for rr in reserves:
                for b in beds:
                    m.outlets[b].reserve = rr
                row = dict(closure=closure, territory=name, reserve=rr,
                           flow_hyper_lad_ml_min=m.hyperaemic().flow_ml_min("LAD"))
                for ds in ds_values:
                    row[f"ffr_{int(round(100 * ds))}"] = lesion_ffr(m, "LAD", ds)
                rows.append(row)
            for b in beds:
                m.outlets[b].reserve = None
    return pd.DataFrame(rows)


def serial_lesions(ds_a: float = 0.50, ds_b: float = 0.50, seg_a: str = "LAD",
                   seg_b: str = "LAD2") -> pd.DataFrame:
    """Two serial lesions on the LAD separated by side-branch take-offs.

    FFR is read at the distal end of the second lesion's segment. For each
    lesion: its apparent contribution (trans-lesional drop with both present)
    and its true contribution (rise of distal FFR when it alone is treated),
    as in the serial-stenosis analysis of Pijls et al. (2000).
    """
    rows = []
    for closure in ("taper", "leaf"):
        m = build(closure)
        a, b = Stenosis(seg_a, ds_a), Stenosis(seg_b, ds_b)
        both = steady_solver.solve_steady(m.tree, m.outlets, [a, b], hyperemic=True)
        b_only = steady_solver.solve_steady(m.tree, m.outlets, [b], hyperemic=True)
        a_only = steady_solver.solve_steady(m.tree, m.outlets, [a], hyperemic=True)

        def distal(sol):
            return sol.p_distal[seg_b] / sol.p_inlet

        rows.append(dict(
            closure=closure, ffr_both=distal(both),
            ffr_after_treating_a=distal(b_only), ffr_after_treating_b=distal(a_only),
            apparent_delta_a=stenosis_drop(m, both, a) / both.p_inlet,
            apparent_delta_b=stenosis_drop(m, both, b) / both.p_inlet,
            true_delta_a=distal(b_only) - distal(both),
            true_delta_b=distal(a_only) - distal(both)))
    return pd.DataFrame(rows)


def stenosis_drop(m: Model, sol: steady_solver.SteadySolution, lesion: Stenosis) -> float:
    """Trans-lesional pressure drop (Pa) of ``lesion`` in solution ``sol``."""
    seg = m.tree[lesion.segment_id]
    return pressure_drop(lesion, seg.radius_mean, sol.flow[lesion.segment_id])


# ==========================================================================
# fractional flow reserve
# ==========================================================================
def ffr_curves(m: Model, vessels: list[str] = CARDINAL,
               grid: np.ndarray | None = None) -> pd.DataFrame:
    grid = np.arange(0.10, 0.9001, 0.025) if grid is None else grid
    rows = []
    for sid in vessels:
        for r in ffr.ffr_curve(m.tree, m.outlets, sid, grid):
            rows.append(dict(vessel=sid, ds_pct=100 * r.diameter_stenosis,
                             area_stenosis_pct=r.area_stenosis_pct, ffr=r.ffr,
                             ffr_venous_corrected=r.ffr_corrected,
                             flow_rest_ml_min=r.flow_rest_ml_min,
                             flow_hyper_ml_min=r.flow_hyper_ml_min))
    return pd.DataFrame(rows)


def ffr_crossings(m: Model, vessels: list[str] = CARDINAL) -> pd.DataFrame:
    return pd.DataFrame([dict(vessel=s, ds_at_threshold_pct=100 * ffr_crossing(m, s),
                              lesion_free_pd_pa=lesion_ffr(m, s, 1e-6))
                         for s in vessels])


def separation_ablation(m: Model, seg_id: str = "LAD") -> pd.DataFrame:
    grid = np.arange(0.30, 0.9001, 0.025)
    return pd.DataFrame([dict(ds_pct=100 * d, full=lesion_ffr(m, seg_id, d),
                              viscous_only=lesion_ffr(m, seg_id, d, include_separation=False))
                         for d in grid])


# ==========================================================================
# pulsatile physiology
# ==========================================================================
def waveform_table(m: Model) -> pd.DataFrame:
    on = pulsatile_solver.solve_pulsatile(m.tree, m.outlets, m.rest)
    off = pulsatile_solver.solve_pulsatile(m.tree, m.outlets, m.rest, im_factor=0.0)
    t = on.time
    return pd.DataFrame(dict(
        t=t,
        p_ao=waveforms.aortic_pressure(t) * C.PA_TO_MMHG,
        p_im=waveforms.intramyocardial_pressure(t) * C.PA_TO_MMHG,
        q_lad=on.flow_ml_min("LAD"), q_lcx=on.flow_ml_min("LCX"),
        q_rca=on.flow_ml_min("RCA"), q_lad_no_pump=off.flow_ml_min("LAD")))


def pulsatile_ablations(m: Model) -> dict[str, float]:
    """Intramyocardial pump removed, and wall stiffness raised so that the
    LAD wave speed matches the directly measured value."""
    base = pulsatile_solver.solve_pulsatile(m.tree, m.outlets, m.rest)
    off = pulsatile_solver.solve_pulsatile(m.tree, m.outlets, m.rest, im_factor=0.0)
    ref = reference_measurements().loc["wave_speed", "mean"]
    scale = (ref / base.wave_speed["LAD"]) ** 2
    stiff = pulsatile_solver.solve_pulsatile(m.tree, m.outlets, m.rest, stiffness_scale=scale)
    return dict(
        lad_df=base.diastolic_flow_fraction("LAD"),
        lad_df_no_pump=off.diastolic_flow_fraction("LAD"),
        lad_dsvr=base.diastolic_systolic_ratio("LAD"),
        lad_dsvr_no_pump=off.diastolic_systolic_ratio("LAD"),
        stiffness_scale=scale,
        lad_wave_speed_stiff=stiff.wave_speed["LAD"],
        lad_dsvr_stiff=stiff.diastolic_systolic_ratio("LAD"),
        rca_dsvr_stiff=stiff.diastolic_systolic_ratio("RCA"),
        lad_df_stiff=stiff.diastolic_flow_fraction("LAD"),
        lad_peak_flow=float(np.max(base.flow_ml_min("LAD"))),
        lad_peak_flow_stiff=float(np.max(stiff.flow_ml_min("LAD"))),
    )


# ==========================================================================
# validation against published measurements
# ==========================================================================
def reference_measurements() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_CSV).set_index("key")


def validation_table(m: Model) -> pd.DataFrame:
    """Model output beside the published measurement it should reproduce.

    ``role`` separates the one calibration target from the predictions.
    For a mean and standard deviation the z-score is (model - mean) / sd;
    for a band the check is containment.
    """
    ref = reference_measurements()
    puls = pulsatile_solver.solve_pulsatile(m.tree, m.outlets, m.rest)
    hyper = m.hyperaemic()

    def gradient(sid: str) -> float:
        return (hyper.p_inlet - hyper.p_distal[sid]) * C.PA_TO_MMHG

    spec = [
        ("Resting total flow", "total_flow_rest", m.rest.total_inflow_ml_min, "calibrated"),
        ("Hyperaemic total flow", "total_flow_hyper", hyper.total_inflow_ml_min, "predicted"),
        ("Resting LCX flow", "lcx_flow_rest", m.rest.flow_ml_min("LCX"), "predicted"),
        ("Resting RCA flow", "rca_flow_rest", m.rest.flow_ml_min("RCA"), "predicted"),
        ("DSVR, proximal LAD", "dsvr_left_rest", puls.diastolic_systolic_ratio("LAD"),
         "predicted"),
        ("DSVR, RCA", "dsvr_right_rest", puls.diastolic_systolic_ratio("RCA"), "predicted"),
        ("Wave speed, proximal LAD", "wave_speed", puls.wave_speed["LAD"], "predicted"),
        ("TAWSS, proximal LAD", "tawss_band", tawss(m, puls, "LAD"), "predicted"),
        ("TAWSS, apical LAD", "tawss_band", tawss(m, puls, "LAD4"), "predicted"),
        ("Hyperaemic gradient to distal LAD", "gradient_hyper_normal", gradient("LAD3"),
         "predicted"),
        ("Hyperaemic gradient to distal RCA", "gradient_hyper_normal", gradient("RCA"),
         "predicted"),
    ]
    rows = []
    for label, key, value, role in spec:
        r = ref.loc[key]
        if pd.notna(r["sd"]):
            z = (value - r["mean"]) / r["sd"]
            agrees = abs(z) <= 2.0
        else:
            z = np.nan
            agrees = bool(r["low"] <= value <= r["high"])
        rows.append(dict(quantity=label, reference_key=key, role=role, unit=r["unit"],
                         model=value, ref_mean=r["mean"], ref_sd=r["sd"],
                         ref_low=r["low"], ref_high=r["high"], z=z,
                         within_2sd_or_band=agrees, source=r["source"], doi=r["doi"],
                         bibkey=r["bibkey"]))
    return pd.DataFrame(rows)


# ==========================================================================
# sensitivity
# ==========================================================================
SENSITIVITY_PARAMETERS = [
    ("Resting coronary flow", "Q_CORONARY_REST"),
    ("Microvascular flow reserve", "CORONARY_FLOW_RESERVE"),
    ("Mean aortic pressure", "P_AO_MEAN"),
    ("Coronary venous pressure", "P_VENOUS"),
    ("Blood viscosity", "MU_BLOOD"),
    ("Blood density", "RHO_BLOOD"),
    ("Morphometric exponent", "MURRAY_EXPONENT"),
]


def sensitivity(seg_id: str = "LAD", ds: float = 0.50, step: float = 0.20) -> pd.DataFrame:
    """One-at-a-time response of a lesion's FFR, the model fully re-solved
    (outflow distribution and calibration included) for every perturbation."""
    def run(**over) -> float:
        with overridden(**over):
            return lesion_ffr(build(), seg_id, ds)

    base = run()
    rows = []
    for label, key in SENSITIVITY_PARAMETERS:
        nominal = getattr(C, key)
        rows.append(dict(parameter=label, constant=key, nominal=nominal,
                         minus=run(**{key: nominal * (1 - step)}) - base,
                         plus=run(**{key: nominal * (1 + step)}) - base))
    m = build()
    length = Stenosis(seg_id, ds).length
    rows.append(dict(parameter="Lesion length", constant="Stenosis.length", nominal=length,
                     minus=lesion_ffr(m, seg_id, ds, length=length * (1 - step)) - base,
                     plus=lesion_ffr(m, seg_id, ds, length=length * (1 + step)) - base))
    df = pd.DataFrame(rows)
    df["span"] = df[["minus", "plus"]].abs().max(axis=1)
    df.attrs["baseline_ffr"] = base
    return df.sort_values("span").reset_index(drop=True)


def murray_exponent_ablation(values=(2.0, 2.24, 2.39, 2.54, 3.0)) -> pd.DataFrame:
    rows = []
    for k in values:
        m = build(murray_exponent=k)
        puls = pulsatile_solver.solve_pulsatile(m.tree, m.outlets, m.rest)
        rows.append(dict(k=k, **{f"{s.lower()}_share_pct": 100 * m.rest.flow[s]
                                 / m.rest.total_inflow for s in ("LMCA", "LAD", "LCX", "RCA")},
                         apical_tawss_pa=tawss(m, puls, "LAD4"),
                         lad_ffr_50=lesion_ffr(m, "LAD", 0.50)))
    return pd.DataFrame(rows)


def flow_scenario(q_rest_ml_min: float) -> pd.DataFrame:
    """Threshold crossings with the resting flow set to another value."""
    with overridden(Q_CORONARY_REST=q_rest_ml_min * C.ML_PER_MIN_TO_M3_PER_S):
        m = build()
        out = ffr_crossings(m)
        out["flow_rest_total"] = m.rest.total_inflow_ml_min
        out["flow_hyper_total"] = m.hyperaemic().total_inflow_ml_min
    return out


def scaled_tree(scale: float) -> CoronaryTree:
    """The default tree with every lumen radius multiplied by ``scale``."""
    tree = geometry.load_default_tree()
    for seg in tree:
        seg.radius_prox *= scale
        seg.radius_dist *= scale
    return tree


def radius_scaling(scales=(1.0, 1.05, 1.10, 1.15, 1.20)) -> pd.DataFrame:
    """Lesion-free hyperaemic aorta-to-distal gradient against a uniform
    scaling of the reference radii (Poiseuille resistance scales as r^-4)."""
    rows = []
    for s in scales:
        m = build(tree=scaled_tree(s))
        h = m.hyperaemic()
        rows.append(dict(radius_scale=s,
                         gradient_distal_lad_mmhg=(h.p_inlet - h.p_distal["LAD3"]) * C.PA_TO_MMHG,
                         gradient_distal_rca_mmhg=(h.p_inlet - h.p_distal["RCA"]) * C.PA_TO_MMHG,
                         lad_ds_at_threshold_pct=100 * ffr_crossing(m, "LAD")))
    return pd.DataFrame(rows)


def rheology_ablation(m: Model) -> dict[str, float]:
    carreau = steady_solver.solve_steady(m.tree, m.outlets, hyperemic=False,
                                         rheology_model="carreau")
    lad = m.tree["LAD"]
    return dict(
        lad_wss_newtonian=float(wss.wall_shear_stress(m.rest.flow["LAD"], lad.radius_mean)),
        lad_wss_carreau=float(wss.wall_shear_stress(carreau.flow["LAD"], lad.radius_mean,
                                                    rheology_model="carreau")))


# ==========================================================================
# numerical verification
# ==========================================================================
def numerics(m: Model) -> dict[str, float]:
    out: dict[str, float] = {}
    for closure in coronary_bc.CLOSURES:
        _, w = coronary_bc.murray_flow_weights(m.tree, closure=closure)
        out[f"closure_error_{closure}"] = abs(sum(w.values()) - 1.0)
    hyper = m.hyperaemic()
    lesion = steady_solver.solve_steady(m.tree, m.outlets, [Stenosis("LAD", 0.7)],
                                        hyperemic=True)
    out["max_residual_rel"] = max(m.rest.residual_norm, hyper.residual_norm,
                                  lesion.residual_norm) / C.Q_CORONARY_REST
    out["max_evaluations"] = max(m.rest.n_evaluations, hyper.n_evaluations,
                                 lesion.n_evaluations)
    coarse = pulsatile_solver.solve_pulsatile(m.tree, m.outlets, m.rest)
    fine = pulsatile_solver.solve_pulsatile(m.tree, m.outlets, m.rest,
                                            n_time=1600, n_harmonics=192)
    err = 0.0
    for sid in m.tree.segments:
        q_c = np.interp(fine.time, coarse.time, coarse.flow[sid], period=C.T_CYCLE)
        err = max(err, float(np.max(np.abs(q_c - fine.flow[sid])) / np.max(fine.flow[sid])))
    out["harmonic_truncation_error"] = err
    out["max_mach_rest"] = coarse.max_mach(m.tree)
    return out


def timing(repeats: int = 20) -> dict[str, float]:
    """Wall-clock time of the main solves (median of ``repeats``), in seconds."""
    def median(fn) -> float:
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))
    m = build()
    return dict(
        build_and_calibrate=median(build),
        pulsatile=median(lambda: pulsatile_solver.solve_pulsatile(m.tree, m.outlets, m.rest)),
        single_lesion_ffr=median(lambda: lesion_ffr(m, "LAD", 0.5)),
    )


# ==========================================================================
# volumetrics and flow at risk
# ==========================================================================
def volumetrics(m: Model) -> dict[str, float]:
    def path_time(path):
        return sum(m.tree[s].volume / m.rest.flow[s] for s in path)
    vol = {side: sum(s.volume for s in m.tree if s.side == side) * 1e6 for side in "LR"}
    total = vol["L"] + vol["R"]
    return dict(volume_total_ml=total, volume_left_ml=vol["L"], volume_right_ml=vol["R"],
                transit_tree_s=total * 1e-6 / m.rest.total_inflow,
                transit_lad_path_s=path_time(LAD_PATH),
                transit_rca_pda_path_s=path_time(["RCA", "PDA"]))


def flow_at_risk(m: Model) -> pd.DataFrame:
    sites = [("Left main", "LMCA"), ("Proximal LAD", "LAD"),
             ("Proximal LCX", "LCX"), ("Proximal RCA", "RCA")]
    return pd.DataFrame([dict(site=name, segment=sid,
                              flow_share_pct=100 * m.rest.flow[sid] / m.rest.total_inflow,
                              flow_ml_min=m.rest.flow_ml_min(sid)) for name, sid in sites])
