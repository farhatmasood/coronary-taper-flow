"""Coronary arterial tree: segment geometry and topology.

The tree is read from ``data/coronary_tree_geometry.csv``. Per-segment lengths
and tapered radii are the open-access geometry of Duanmu et al. (2019); the
branching topology is the standard clinical (AHA/SYNTAX) coronary anatomy
encoded by the ``parent_id`` column.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Pseudo-roots: the two coronary ostia at the aortic sinuses.
AORTA_LEFT = "AORTA_L"
AORTA_RIGHT = "AORTA_R"
ROOTS = (AORTA_LEFT, AORTA_RIGHT)


@dataclass
class Segment:
    """A single coronary arterial segment, modelled as a tapered elastic tube."""

    seg_id: str
    name: str
    clinical_name: str
    side: str                       # 'L' or 'R'
    parent_id: str
    length: float                   # m
    radius_prox: float              # m  (lumen radius, proximal end)
    radius_dist: float              # m  (lumen radius, distal end)
    is_terminal: bool
    children: list[str] = field(default_factory=list)

    # --- reference geometry ------------------------------------------------
    @property
    def radius_mean(self) -> float:
        """Length-averaged radius of the linearly tapered lumen."""
        return 0.5 * (self.radius_prox + self.radius_dist)

    @property
    def area_prox(self) -> float:
        return np.pi * self.radius_prox**2

    @property
    def area_dist(self) -> float:
        return np.pi * self.radius_dist**2

    @property
    def area_mean(self) -> float:
        return np.pi * self.radius_mean**2

    def radius_at(self, xi: float) -> float:
        """Lumen radius at normalised position xi in [0, 1] (0 = proximal)."""
        return self.radius_prox + (self.radius_dist - self.radius_prox) * xi

    def area_at(self, xi: float) -> float:
        return np.pi * self.radius_at(xi) ** 2

    @property
    def volume(self) -> float:
        """Lumen volume of the linearly tapered (truncated-cone) segment."""
        rp, rd = self.radius_prox, self.radius_dist
        return np.pi * self.length * (rp * rp + rp * rd + rd * rd) / 3.0


class CoronaryTree:
    """Container for the coronary segments plus topology helpers."""

    def __init__(self, segments: dict[str, Segment]):
        self.segments = segments
        self._link_children()

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_csv(cls, csv_path: str | Path) -> CoronaryTree:
        segments: dict[str, Segment] = {}
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                seg = Segment(
                    seg_id=row["segment_id"].strip(),
                    name=row["segment_name"].strip(),
                    clinical_name=row["clinical_name"].strip(),
                    side=row["side"].strip(),
                    parent_id=row["parent_id"].strip(),
                    length=float(row["length_mm"]) * 1e-3,
                    radius_prox=float(row["radius_prox_mm"]) * 1e-3,
                    radius_dist=float(row["radius_dist_mm"]) * 1e-3,
                    is_terminal=bool(int(row["is_terminal"])),
                )
                segments[seg.seg_id] = seg
        return cls(segments)

    # -------------------------------------------------------------- topology
    def _link_children(self) -> None:
        for seg in self.segments.values():
            seg.children = []
        for seg in self.segments.values():
            if seg.parent_id in self.segments:
                self.segments[seg.parent_id].children.append(seg.seg_id)
        # Validate that declared terminals really have no children, and that
        # every non-root parent reference resolves.
        for seg in self.segments.values():
            if seg.is_terminal and seg.children:
                raise ValueError(f"segment {seg.seg_id} flagged terminal but has children")
            if not seg.is_terminal and not seg.children:
                raise ValueError(f"segment {seg.seg_id} is non-terminal but has no children")
            if seg.parent_id not in self.segments and seg.parent_id not in ROOTS:
                raise ValueError(f"segment {seg.seg_id} has unknown parent {seg.parent_id}")

    def inlets(self) -> list[str]:
        """Segments fed directly from an aortic ostium (tree roots)."""
        return [s.seg_id for s in self.segments.values() if s.parent_id in ROOTS]

    def terminals(self) -> list[str]:
        return [s.seg_id for s in self.segments.values() if s.is_terminal]

    def ordered_ids(self) -> list[str]:
        """Segment ids in a parent-before-child (breadth-first) order."""
        order: list[str] = []
        queue = list(self.inlets())
        while queue:
            sid = queue.pop(0)
            order.append(sid)
            queue.extend(self.segments[sid].children)
        return order

    def path_to_root(self, seg_id: str) -> list[str]:
        """Segment ids from the given segment up to (and including) an inlet."""
        path = [seg_id]
        cur = self.segments[seg_id]
        while cur.parent_id in self.segments:
            cur = self.segments[cur.parent_id]
            path.append(cur.seg_id)
        return path

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, seg_id: str) -> Segment:
        return self.segments[seg_id]

    def __iter__(self):
        return iter(self.segments.values())

    # --------------------------------------------------------------- summary
    def total_volume(self) -> float:
        return sum(s.volume for s in self.segments.values())

    def describe(self) -> str:
        lines = [f"Coronary tree: {len(self)} segments, "
                 f"{len(self.inlets())} inlets, {len(self.terminals())} terminals",
                 f"Total epicardial lumen volume: {self.total_volume()*1e6:.3f} mL"]
        for sid in self.ordered_ids():
            s = self.segments[sid]
            kind = "terminal" if s.is_terminal else f"-> {', '.join(s.children)}"
            lines.append(
                f"  {sid:7s} {s.clinical_name:34s} "
                f"L={s.length*1e3:6.1f} mm  d={2e3*s.radius_mean:4.2f} mm  {kind}")
        return "\n".join(lines)


def load_default_tree() -> CoronaryTree:
    """Load the coronary tree shipped with the repository."""
    here = Path(__file__).resolve().parents[2]
    return CoronaryTree.from_csv(here / "data" / "coronary_tree_geometry.csv")


def copy_tree(tree: CoronaryTree) -> CoronaryTree:
    """Independent copy of a tree (segments are mutable)."""
    fields = ("seg_id", "name", "clinical_name", "side", "parent_id", "length",
              "radius_prox", "radius_dist", "is_terminal")
    return CoronaryTree({sid: Segment(**{f: getattr(s, f) for f in fields})
                         for sid, s in tree.segments.items()})


def isotonic_nonincreasing(values: list[float]) -> list[float]:
    """Least-squares non-increasing fit (pool-adjacent-violators)."""
    blocks: list[list[float]] = []            # [mean, weight]
    for v in values:
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            m2, w2 = blocks.pop()
            m1, w1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    out: list[float] = []
    for mean, w in blocks:
        out.extend([mean] * int(round(w)))
    return out


def monotone_reference(tree: CoronaryTree) -> CoronaryTree:
    """Reference diameter function: a non-increasing least-squares fit of the
    measured radii along every unbranched chain of segments.

    Quantitative angiography reports a smooth reference diameter function
    rather than raw edge radii; this is the minimal version of that step, and
    the first step of :func:`coronary.model.build`. Chains start at an ostium
    or at a daughter of a bifurcation and follow single-child continuations.
    Daughters are not capped at their parent's radius: at a bifurcation the
    law itself caps the pass-through fraction at one, and the parent's radius
    next to a junction is the least reliable measurement. A tree whose chains
    already narrow monotonically is returned unchanged (as a copy).
    """
    ref = copy_tree(tree)
    for sid in ref.ordered_ids():
        seg = ref[sid]
        parent = ref.segments.get(seg.parent_id)
        if parent is not None and len(parent.children) == 1:
            continue                              # inside a chain already fitted
        chain = [sid]
        while len(ref[chain[-1]].children) == 1:
            chain.append(ref[chain[-1]].children[0])
        values = []
        for s in chain:
            values += [ref[s].radius_prox, ref[s].radius_dist]
        fitted = isotonic_nonincreasing(values)
        for i, s in enumerate(chain):
            ref[s].radius_prox, ref[s].radius_dist = fitted[2 * i], fitted[2 * i + 1]
    return ref
