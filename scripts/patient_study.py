"""The outflow law on patient coronary trees from CT angiography.

    python scripts/patient_study.py --data <CCA folder>

``--data`` is a folder with ``labels/<i>.nii.gz`` (lumen masks) and
``images/<i>.nii.gz`` (the CT volumes), as in the training set of the CCA
dataset (Yang et al., arXiv:2305.04208; https://huggingface.co/datasets/MedHK23/CCA,
GPL-3.0; not redistributed here). Needs ``pip install -e ".[patients]"``.

Stage 1 extracts every coronary tree (centreline, radii, ostium) and caches
it under ``<data>/derived/`` as a segment table in the format of
``data/coronary_tree_geometry.csv``, a JSON record and a QC figure. Stage 2
keeps the trees whose ostium meets the contrast-filled aorta on the CT,
solves each with its share of the resting coronary flow (left and right as
in the reference tree) and compares, along the dominant (main) vessel:

* the full patient tree with the taper-resolved law (reference) and with
  outlets at the visible ends only (leaf outlets, as in CT-derived FFR);
* the main vessel alone, as quantitative angiography sees it, with the law
  (taper), with step-downs at visible branches only (junction, as in Murray-
  law QFR) and with constant flow (leaf, as in fixed-flow QFR);

and the Murray exponent at every visible bifurcation. Results go to
``results/patients_trees.csv``, ``results/patients_bifurcations.csv`` and
``results/patients_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from coronary import constants as C  # noqa: E402
from coronary import coronary_bc, experiments, model  # noqa: E402
from coronary import patient as P  # noqa: E402
from coronary.geometry import CoronaryTree, monotone_reference  # noqa: E402

log = logging.getLogger("patient_study")

# Header spacing of the CCA volumes is 0.5 mm, but they have been resampled
# twofold: at 0.5 mm the 832 x 832 x 576 field of view would be 416 x 416 x
# 288 mm and the aortic root about 75 mm wide; at 0.25 mm they are 208 x 208 x
# 144 mm and about 37 mm, as in cardiac CT angiography.
SPACING_MM = 0.25
# Lesion sites: 25 % and 75 % of the treatable main vessel, the part with a
# reference diameter of at least 2 mm (smaller vessels are not stented).
SITES = {"prox": 0.25, "dist": 0.75}
MIN_TREATABLE_DIAMETER_MM = 2.0
MODELS = ("full_taper", "full_leaf", "single_taper", "single_junction", "single_leaf")


# ----------------------------------------------------------------- stage 1
def extract(data: Path, cache: Path, ids: list[int]) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for i in ids:
        record = cache / f"p{i}.json"
        if record.exists():
            continue
        trees = P.trees_from_mask(data / "labels" / f"{i}.nii.gz", spacing_mm=SPACING_MM,
                                  image_path=data / "images" / f"{i}.nii.gz")
        out = {}
        for t in trees:
            tree = P.to_coronary_tree([t])
            P.write_tree_csv(tree, cache / f"p{i}_{t.side}.csv", f"CCA case {i} (derived)")
            out[t.side] = dict(
                root_piece=t.pieces[t.root][0],
                aortic_contact=t.aortic_contact,
                ostium_ras_mm=[float(x) for x in t.ostium_world_mm],
                n_branches=len(t.branches),
                n_bifurcations=sum(1 for b in t.branches.values() if len(b.children) >= 2),
                n_terminals=sum(1 for b in t.branches.values() if not b.children),
                centreline_mm=float(sum(b.length for b in t.branches.values())),
                bifurcations=[dict(k=r["k"], r_parent_mm=r["r_parent_mm"],
                                   r_children_mm=r["r_children_mm"])
                              for r in P.murray_exponents([t], tree)])
        record.write_text(json.dumps(out, indent=1), encoding="utf-8")
        _qc_figure(trees, cache / f"p{i}_qc.png", f"CCA case {i}")
        log.info("case %d: %s", i, {s: (v["n_branches"], v["aortic_contact"])
                                     for s, v in out.items()})


def _qc_figure(trees, path: Path, title: str) -> None:
    """Centrelines of both trees (line width ~ radius) and the ostia, in two views."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for t in trees:
        colour = "tab:red" if t.side == "R" else "tab:blue"
        for ax, (a, b) in zip(axes, [(0, 2), (0, 1)], strict=True):
            for br in t.branches.values():
                ax.plot(br.xyz[:, a], br.xyz[:, b], color=colour,
                        lw=0.4 + 1.2 * float(np.median(br.r)))
            ax.plot(t.ostium_world_mm[a], t.ostium_world_mm[b], "ko", ms=6)
            ax.annotate(t.side, (t.ostium_world_mm[a], t.ostium_world_mm[b]))
    for ax, lab in zip(axes, ("z (superior, mm)", "y (anterior, mm)"), strict=True):
        ax.set_xlabel("x (patient right, mm)")
        ax.set_ylabel(lab)
        ax.set_aspect("equal")
    fig.suptitle(title)
    fig.savefig(path, dpi=60, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- stage 2
def _treatable(m, path: list[str]) -> list[str]:
    """Main-vessel segments up to the first one narrower than the treatable
    diameter."""
    out = []
    for s in path:
        if 2e3 * m.tree[s].radius_mean < MIN_TREATABLE_DIAMETER_MM:
            break
        out.append(s)
    return out or path[:1]


def _site_segment(m, path: list[str], frac: float) -> str:
    cum = np.cumsum([m.tree[s].length for s in path])
    return path[min(int(np.searchsorted(cum, frac * cum[-1])), len(path) - 1)]


def _beyond_view_share(tree: CoronaryTree) -> float:
    """Share of a tree's inflow that the law sends to branches below the
    resolution of the scan: what does not leave through the visible ends."""
    fitted = monotone_reference(tree)
    seg_flow, _ = coronary_bc.murray_flow_weights(fitted, closure="taper")
    k = C.MURRAY_EXPONENT
    out_ends = sum(seg_flow[s.seg_id] * (s.radius_dist / s.radius_prox) ** k
                   for s in fitted if s.is_terminal)
    return 1.0 - out_ends / sum(seg_flow[s] for s in fitted.inlets())


def analyse(cache: Path, ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = model.build()
    left_share = ref.rest.flow["LMCA"] / ref.rest.total_inflow
    rows, bif = [], []
    for i in ids:
        meta = json.loads((cache / f"p{i}.json").read_text(encoding="utf-8"))
        for side, mt in meta.items():
            if mt["aortic_contact"] == 0:            # ostium not confirmed on the CT
                log.info("case %d %s excluded: no centreline end meets the aorta", i, side)
                continue
            tree = CoronaryTree.from_csv(cache / f"p{i}_{side}.csv")
            q = C.Q_CORONARY_REST * (left_share if side == "L" else 1.0 - left_share)
            full = {c: P.model_from_tree(tree, c, q) for c in ("taper", "leaf")}
            root = mt["root_piece"]
            path = P.dominant_path(full["taper"], root)
            single_tree = P.path_tree(tree, path)
            ms = {"full_taper": full["taper"], "full_leaf": full["leaf"]}
            for c in ("taper", "junction", "leaf"):
                ms[f"single_{c}"] = P.model_from_tree(single_tree, c, q)
            row = dict(case=i, side=side, aortic_contact=mt["aortic_contact"],
                       n_branches=mt["n_branches"], n_bifurcations=mt["n_bifurcations"],
                       centreline_mm=mt["centreline_mm"],
                       ostial_radius_mm=ms["full_taper"].tree[root].radius_prox * 1e3,
                       main_vessel_mm=sum(ms["full_taper"].tree[s].length for s in path) * 1e3,
                       main_vessel_segments=len(path), q_rest_ml_min=q * 60e6,
                       beyond_view_share=_beyond_view_share(tree))
            treat = _treatable(ms["full_taper"], path)
            row["treatable_mm"] = sum(ms["full_taper"].tree[s].length for s in treat) * 1e3
            for site, frac in SITES.items():
                seg = _site_segment(ms["full_taper"], treat, frac)
                for name, m in ms.items():
                    row[f"share_{site}_{name}"] = m.rest.flow[seg] / m.rest.flow[root]
                    row[f"ffr_{site}_{name}"] = experiments.lesion_ffr(m, seg, 0.50)
            rows.append(row)
            for b in mt["bifurcations"]:
                bif.append(dict(case=i, side=side, **b))
        log.info("case %d analysed", i)
    return pd.DataFrame(rows), pd.DataFrame(bif)


# ----------------------------------------------------------------- outputs
def summary(trees: pd.DataFrame, bif: pd.DataFrame) -> dict[str, float]:
    k = bif.k.dropna()
    s = dict(n_cases=trees.case.nunique(), n_trees=len(trees),
             n_excluded=2 * trees.case.nunique() - len(trees),
             n_bifurcations=len(bif), n_k=len(k), k_median=k.median(),
             k_lo=k.quantile(0.25), k_hi=k.quantile(0.75),
             main_vessel_median=trees.main_vessel_mm.median(),
             beyond_view_median=100 * trees.beyond_view_share.median(),
             beyond_view_lo=100 * trees.beyond_view_share.quantile(0.25),
             beyond_view_hi=100 * trees.beyond_view_share.quantile(0.75))
    for site in SITES:
        ref = trees[f"ffr_{site}_full_taper"]
        for name in MODELS[1:]:
            d = trees[f"ffr_{site}_{name}"] - ref
            s[f"ffr_{site}_{name}_bias"] = d.mean()
            s[f"ffr_{site}_{name}_sd"] = d.std()
            s[f"ffr_{site}_{name}_mae"] = d.abs().mean()
            s[f"ffr_{site}_{name}_flip"] = int(((trees[f"ffr_{site}_{name}"] <= 0.80)
                                                != (ref <= 0.80)).sum())
            sh = trees[f"share_{site}_{name}"] - trees[f"share_{site}_full_taper"]
            s[f"share_{site}_{name}_mae"] = 100 * sh.abs().mean()
        s[f"ffr_{site}_ref_median"] = ref.median()
        s[f"share_{site}_ref_median"] = 100 * trees[f"share_{site}_full_taper"].median()
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--outdir", type=Path, default=REPO / "results")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ids = list(range(args.cases))
    cache = args.data / "derived"
    extract(args.data, cache, ids)
    trees, bif = analyse(cache, ids)
    args.outdir.mkdir(parents=True, exist_ok=True)
    trees.to_csv(args.outdir / "patients_trees.csv", index=False, float_format="%.6g")
    bif.to_csv(args.outdir / "patients_bifurcations.csv", index=False, float_format="%.6g")
    s = summary(trees, bif)
    (args.outdir / "patients_summary.json").write_text(
        json.dumps({k: float(v) for k, v in s.items()}, indent=1), encoding="utf-8")
    log.info("%d trees from %d cases; median k %.2f", s["n_trees"], s["n_cases"], s["k_median"])


if __name__ == "__main__":
    main()
