"""Patient coronary trees from CT angiography lumen segmentations.

A binary lumen mask (NIfTI) is reduced, tree by tree, to a
:class:`~coronary.geometry.CoronaryTree` of linearly tapered segments that the
model solves like the reference tree:

1. each connected component of the mask is one coronary tree;
2. the lumen is thinned to a one-voxel centreline (3D thinning) and the
   voxel graph is reduced to a spanning tree that opens every loop (vessels
   touching each other) at its narrowest point;
3. spurs shorter than twice the local radius are pruned; the ostia are the
   closest pair of centreline ends, one per tree, that open into the
   contrast-filled aortic root (read from the CT image), or, without an
   image, the end from which the lumen narrows most consistently downstream;
   junction clusters left by thinning are merged and terminal stubs shorter
   than three radii (or 3 mm) dropped;
4. the local radius is the equivalent-circle radius of the lumen nearest to
   a short window of centreline; each piece of a branch takes the median of
   it, ignoring the lumen within one radius of a junction and the ostial
   funnel;
5. every branch is cut into pieces of at most ``piece_mm`` and each piece is
   a linearly tapered segment.

Segment names are ``L``/``R`` + branch number + ``_`` + piece number; the
tree lying further to the patient's right is the right coronary tree. Needs
the optional dependencies nibabel, scikit-image and networkx
(``pip install -e ".[patients]"``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import constants as C
from . import coronary_bc, steady_solver
from .geometry import AORTA_LEFT, AORTA_RIGHT, CoronaryTree, Segment, monotone_reference
from .model import Model

_OFFSETS = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
            if (i, j, k) != (0, 0, 0)]


# --------------------------------------------------------------------------
# Centreline branches
# --------------------------------------------------------------------------
@dataclass
class Branch:
    """One centreline branch between two key points (ostium, junction, end)."""

    bid: int
    parent: int | None
    s: np.ndarray                       # arc length from the branch start (mm)
    r: np.ndarray                       # local lumen radius (mm)
    xyz: np.ndarray                     # centreline points, RAS+ (mm)
    start_junction: bool                # starts at a junction (else at the ostium)
    end_junction: bool                  # ends at a junction (else a terminal end)
    children: list[int] = field(default_factory=list)

    @property
    def length(self) -> float:
        return float(self.s[-1])


@dataclass
class PatientTree:
    """Centreline branches of one coronary tree and its model geometry."""

    side: str                           # 'L' or 'R'
    branches: dict[int, Branch]
    root: int                           # id of the ostial branch
    ostium_world_mm: np.ndarray         # ostium position in RAS+ coordinates
    pieces: dict[int, list[str]] = field(default_factory=dict)   # model segments
    aortic_contact: int = 0             # bright voxels next to the ostium (0: none)

    def descendants(self, bid: int) -> list[int]:
        out, stack = [], [bid]
        while stack:
            b = stack.pop()
            out.append(b)
            stack.extend(self.branches[b].children)
        return out


def _voxel_graph(skel: np.ndarray, spacing: np.ndarray):
    import networkx as nx
    pts = np.argwhere(skel)
    index = {tuple(p): i for i, p in enumerate(pts)}
    g = nx.Graph()
    g.add_nodes_from(range(len(pts)))
    for i, p in enumerate(pts):
        for o in _OFFSETS:
            j = index.get((p[0] + o[0], p[1] + o[1], p[2] + o[2]))
            if j is not None and j > i:
                g.add_edge(i, j, weight=float(np.linalg.norm(np.multiply(o, spacing))))
    return pts, g


def _local_lumen(mask: np.ndarray, skel: np.ndarray, pts: np.ndarray, tree,
                 spacing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lumen volume (mm^3) nearest to each centreline voxel and the centreline
    length (mm) it stands for."""
    from scipy import ndimage as ndi
    _, idx = ndi.distance_transform_edt(~skel, sampling=spacing, return_indices=True)
    owner_lin = np.ravel_multi_index(tuple(idx[:, mask]), mask.shape)
    order = np.full(int(np.prod(mask.shape)), -1, dtype=np.int64)
    order[np.ravel_multi_index(tuple(pts.T), mask.shape)] = np.arange(len(pts))
    counts = np.bincount(order[owner_lin], minlength=len(pts)).astype(float)
    ell = np.zeros(len(pts))
    for i, j, d in tree.edges(data=True):
        ell[i] += 0.5 * d["weight"]
        ell[j] += 0.5 * d["weight"]
    return counts * float(np.prod(spacing)), ell


def _branches_from_tree(tree, root: int, radius: np.ndarray, world: np.ndarray,
                        lumen: tuple[np.ndarray, np.ndarray] | None = None,
                        window: int = 7) -> tuple[dict[int, Branch], int]:
    """Split a rooted centreline tree into branches between key points.

    With ``lumen`` (volume and length per centreline voxel) the radius along a
    branch is the equivalent-circle radius of the lumen over a moving window
    of ``window`` centreline voxels, which removes the voxel-stepping noise of
    oblique vessels; otherwise it is ``radius``.
    """
    parent = {root: None}
    order = [root]
    for u in order:                                   # breadth-first orientation
        for v in tree.neighbors(u):
            if v not in parent:
                parent[v] = u
                order.append(v)
    kids = {u: [] for u in order}
    for v in order[1:]:
        kids[parent[v]].append(v)

    branches: dict[int, Branch] = {}
    queue = [(root, None)]
    while queue:
        start, pbid = queue.pop(0)
        for first in (kids[start] if pbid is not None else [start]):
            nodes = [start, first] if pbid is not None else [start]
            while len(kids[nodes[-1]]) == 1:
                nodes.append(kids[nodes[-1]][0])
            if pbid is None and len(nodes) == 1:      # root end directly at a junction
                continue
            seg = [tree[a][b]["weight"] for a, b in zip(nodes, nodes[1:], strict=False)]
            if lumen is None:
                r = radius[nodes]
            else:
                ker = np.ones(min(window, len(nodes)))
                vol = np.convolve(lumen[0][nodes], ker, mode="same")
                ell = np.convolve(lumen[1][nodes], ker, mode="same")
                r = np.sqrt(vol / (math.pi * np.maximum(ell, 1e-9)))
            bid = len(branches)
            b = Branch(bid=bid, parent=pbid, s=np.r_[0.0, np.cumsum(seg)],
                       r=r, xyz=world[nodes], start_junction=pbid is not None,
                       end_junction=len(kids[nodes[-1]]) >= 2)
            branches[bid] = b
            if pbid is not None:
                branches[pbid].children.append(bid)
            if b.end_junction:
                queue.append((nodes[-1], bid))
    return branches, 0


def _spur(tree, end: int) -> tuple[list[int], float, int]:
    """Nodes of the terminal run from ``end`` to the next junction, its length
    and the junction node."""
    path, length, u, prev = [end], 0.0, end, None
    while True:
        nxt = [v for v in tree.neighbors(u) if v != prev]
        if len(nxt) != 1:
            return path, length, u
        v = nxt[0]
        length += tree[u][v]["weight"]
        prev, u = u, v
        if tree.degree(u) != 2:
            return path, length, u
        path.append(u)


def _prune(tree, root: int, radius: np.ndarray, min_mm: float, ratio: float) -> None:
    """Remove terminal spurs shorter than max(min_mm, ratio x junction radius)."""
    changed = True
    while changed:
        changed = False
        for end in [n for n, d in tree.degree() if d == 1 and n != root]:
            path, length, j = _spur(tree, end)
            if tree.degree(j) >= 3 and length < max(min_mm, ratio * radius[j]):
                tree.remove_nodes_from(path)
                changed = True
                break


def _contract_short(branches: dict[int, Branch], min_mm: float) -> dict[int, Branch]:
    """Merge internal branches shorter than ``min_mm`` (junction clusters left
    by thinning) into their parent: their children become the parent's."""
    for bid in sorted(branches):
        b = branches.get(bid)
        if b is None or b.parent is None or not b.children or b.length >= min_mm:
            continue
        p = branches[b.parent]
        p.children = [c for c in p.children if c != bid] + b.children
        for c in b.children:
            branches[c].parent = p.bid
        del branches[bid]
    return branches


def _drop_stubs(branches: dict[int, Branch], min_mm: float, ratio: float) -> dict[int, Branch]:
    """Remove terminal branches shorter than max(min_mm, ratio x their median
    radius): ostial-funnel and junction remnants, too short to be vessels."""
    changed = True
    while changed:
        changed = False
        for bid, b in list(branches.items()):
            if b.children or b.parent is None:
                continue
            if b.length < max(min_mm, ratio * float(np.median(b.r))):
                branches[b.parent].children.remove(bid)
                del branches[bid]
                changed = True
    return branches


def _component(lab: np.ndarray, c: int, sl, spacing: np.ndarray, affine: np.ndarray) -> dict:
    """Centreline spanning tree, inscribed radius and world position of one tree."""
    import networkx as nx
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    sl = tuple(slice(max(s.start - 2, 0), s.stop + 2) for s in sl)
    m = lab[sl] == c
    pts, g = _voxel_graph(skeletonize(m) > 0, spacing)
    r_edt = ndi.distance_transform_edt(m, sampling=spacing)[tuple(pts.T)]
    # Loops (vessels touching each other) are opened at their narrowest edge,
    # so a contact bridge is cut rather than a vessel trunk.
    for u, v, d in g.edges(data=True):
        d["cost"] = 1.0 / max(min(r_edt[u], r_edt[v]), 1e-3)
    tree = nx.minimum_spanning_tree(g.subgraph(max(nx.connected_components(g), key=len)),
                                    weight="cost").copy()
    corner = np.array([s.start for s in sl])
    world = (affine @ np.c_[corner + pts, np.ones(len(pts))].T)[:3].T
    return dict(mask=m, pts=pts, tree=tree, r_edt=r_edt, world=world, corner=corner)


def _aortic_contact(voxels: np.ndarray, image: np.ndarray, lumen: np.ndarray,
                    radius_vox: float, threshold: float) -> np.ndarray:
    """Number of contrast-bright voxels outside the coronary lumen within a
    sphere around each centreline end: large only where the lumen opens into
    the contrast-filled aortic root."""
    r = int(math.ceil(radius_vox))
    g = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    ball = (g ** 2).sum(axis=0) <= radius_vox ** 2
    out = np.zeros(len(voxels), int)
    for i, v in enumerate(voxels):
        lo = np.maximum(v - r, 0)
        hi = np.minimum(v + r + 1, image.shape)
        b = ball[tuple(slice(a - (c - r), d - (c - r)) for a, d, c in zip(lo, hi, v, strict=True))]
        sl = tuple(slice(a, d) for a, d in zip(lo, hi, strict=True))
        out[i] = int(((image[sl] > threshold) & ~lumen[sl] & b).sum())
    return out


def _inversion_cost(tree, root: int, r: np.ndarray, world: np.ndarray) -> float:
    """Length-weighted widening downstream when the tree is rooted at ``root``:
    the sum over branches of length x max(0, median radius - parent median).
    Lumen narrows away from the ostium, so the ostium minimises it."""
    branches, _ = _branches_from_tree(tree, root, r, world)
    med = {bid: float(np.median(b.r)) for bid, b in branches.items()}
    return sum(b.length * max(0.0, med[bid] - med[b.parent])
               for bid, b in branches.items() if b.parent is not None)


def trees_from_mask(path: str | Path, spacing_mm: float | None = None,
                    image_path: str | Path | None = None, bright_hu: float = 300.0,
                    contact_mm: float = 3.0, strong_contact: float = 0.3,
                    min_spur_mm: float = 2.0, spur_ratio: float = 2.0,
                    min_internal_mm: float = 1.0, min_stub_mm: float = 3.0,
                    stub_ratio: float = 3.0) -> list[PatientTree]:
    """Centreline trees of the connected components of a lumen mask.

    ``spacing_mm`` overrides the (isotropic) voxel spacing of the file
    header. After spur pruning, the ostia are found where the lumen opens
    into the contrast-filled aortic root: the centreline ends with at least
    ``strong_contact`` of their tree's largest count of voxels above
    ``bright_hu`` outside the lumen within ``contact_mm`` (from the CT image
    ``image_path``), and of those, for two trees, the closest pair, since
    both ostia lie on the aortic root. Without an image, or where no end
    touches a bright region, the ostium is the end that minimises
    :func:`_inversion_cost`. The tree lying further to the patient's right
    is the right coronary tree.
    """
    import itertools

    import nibabel as nib
    from scipy import ndimage as ndi

    img = nib.load(str(path))
    mask = np.asarray(img.dataobj) > 0
    header_spacing = np.array(img.header.get_zooms()[:3], float)
    spacing = header_spacing if spacing_mm is None else np.full(3, float(spacing_mm))
    affine = img.affine.copy()
    affine[:3, :3] *= (spacing / header_spacing)[None, :]
    lab, _ = ndi.label(mask, structure=np.ones((3, 3, 3)))
    image = None if image_path is None else np.asarray(nib.load(str(image_path)).dataobj)
    comps = []
    for c, sl in enumerate(ndi.find_objects(lab), start=1):
        if (lab[sl] == c).sum() < 1000:
            continue
        cp = _component(lab, c, sl, spacing, affine)
        _prune(cp["tree"], -1, cp["r_edt"], min_spur_mm, spur_ratio)
        cp["ends"] = [u for u, d in cp["tree"].degree() if d == 1]
        cp["contact"] = np.zeros(len(cp["ends"]), int)
        if image is not None:
            cp["contact"] = _aortic_contact(cp["corner"] + cp["pts"][cp["ends"]], image, mask,
                                            contact_mm / spacing[0], bright_hu)
        if cp["contact"].max() > 0:
            strong = cp["contact"] >= strong_contact * cp["contact"].max()
            cp["candidates"] = [e for e, ok in zip(cp["ends"], strong, strict=True) if ok]
        else:
            cp["candidates"] = [min(cp["ends"], key=lambda u, cp=cp: _inversion_cost(
                cp["tree"], u, cp["r_edt"], cp["world"]))]
        comps.append(cp)
    if len(comps) == 2:
        roots = min(itertools.product(comps[0]["candidates"], comps[1]["candidates"]),
                    key=lambda pr: np.linalg.norm(comps[0]["world"][pr[0]]
                                                  - comps[1]["world"][pr[1]]))
    else:
        roots = [max(cp["candidates"], key=lambda u, cp=cp: cp["contact"][cp["ends"].index(u)])
                 for cp in comps]
    out = []
    for cp, root in zip(comps, roots, strict=True):
        tree, r, world = cp["tree"], cp["r_edt"], cp["world"]
        lumen = _local_lumen(cp["mask"], _mask_of(tree, cp["pts"], cp["mask"].shape),
                             cp["pts"], tree, spacing)
        branches, rb = _branches_from_tree(tree, root, r, world, lumen=lumen)
        branches = _drop_stubs(_contract_short(branches, min_internal_mm),
                               min_stub_mm, stub_ratio)
        branches = _contract_short(branches, min_internal_mm)
        out.append(PatientTree(side="?", branches=branches, root=rb,
                               ostium_world_mm=world[root],
                               aortic_contact=int(cp["contact"][cp["ends"].index(root)])))
    # side: the right coronary tree lies to the patient's right (+x in RAS+)
    if len(out) == 2:
        centre_x = [np.mean(np.concatenate([b.xyz[:, 0] for b in t.branches.values()]))
                    for t in out]
        right = int(np.argmax(centre_x))
        for i, t in enumerate(out):
            t.side = "R" if i == right else "L"
    return out


def _mask_of(tree, pts: np.ndarray, shape) -> np.ndarray:
    sk = np.zeros(shape, bool)
    nodes = np.fromiter(tree.nodes, dtype=np.int64)
    sk[tuple(pts[nodes].T)] = True
    return sk


# --------------------------------------------------------------------------
# Model geometry
# --------------------------------------------------------------------------
def _radius_profile(b: Branch, edges: np.ndarray, parent_r: float | None = None,
                    ostium_mm: float = 3.0) -> np.ndarray:
    """Reference radius at the piece boundaries ``edges`` (mm along the branch).

    Each piece takes the median local radius of its lumen, leaving out the
    lumen within one radius of a junction (the larger of the branch's and its
    parent's median radius at its start, where the junction flares) and the
    ostial funnel (``ostium_mm`` or two radii); the reference radius runs
    linearly between piece centres and is constant beyond the first and the
    last. A branch with no lumen left takes its median radius.
    """
    r_med = float(np.median(b.r))
    lead = max(r_med, parent_r or 0.0) if b.start_junction else max(ostium_mm, 2 * r_med)
    tail = r_med if b.end_junction else 0.0
    keep = (b.s >= lead) & (b.s <= b.length - tail)
    if not keep.any():
        return np.full(len(edges), r_med)
    s_k, r_k = b.s[keep], b.r[keep]
    centres, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        inside = (s_k >= lo) & (s_k <= hi)
        if inside.any():
            centres.append(float(np.mean(s_k[inside])))
            medians.append(float(np.median(r_k[inside])))
    return np.interp(edges, centres, medians)


def to_coronary_tree(trees: list[PatientTree], piece_mm: float = 10.0) -> CoronaryTree:
    """Model geometry: every branch cut into linearly tapered pieces."""
    segs: dict[str, Segment] = {}
    for t in trees:
        root_parent = AORTA_LEFT if t.side == "L" else AORTA_RIGHT
        last_piece: dict[int, str] = {}
        for bid in t.descendants(t.root):
            b = t.branches[bid]
            n = max(1, math.ceil(b.length / piece_mm))
            edges = np.linspace(0.0, b.length, n + 1)
            parent_r = None if b.parent is None else float(np.median(t.branches[b.parent].r))
            r_at = np.maximum(_radius_profile(b, edges, parent_r), 0.1)
            parent = root_parent if b.parent is None else last_piece[b.parent]
            t.pieces[bid] = []
            for k in range(n):
                sid = f"{t.side}{bid}_{k}"
                t.pieces[bid].append(sid)
                segs[sid] = Segment(
                    seg_id=sid, name=sid, clinical_name=f"{t.side} branch {bid}",
                    side=t.side, parent_id=parent,
                    length=max(edges[k + 1] - edges[k], 0.1) * 1e-3,
                    radius_prox=r_at[k] * 1e-3, radius_dist=r_at[k + 1] * 1e-3,
                    is_terminal=(k == n - 1 and not b.children))
                parent = sid
            last_piece[bid] = parent
    return CoronaryTree(segs)


def murray_exponents(trees: list[PatientTree], tree: CoronaryTree) -> list[dict]:
    """Exponent k solving r_p^k = sum r_c^k at every visible bifurcation, from
    the fitted radii at the junction (None where no k in [1, 6] solves it)."""
    from scipy.optimize import brentq
    rows = []
    for t in trees:
        for b in t.branches.values():
            if len(b.children) < 2:
                continue
            rp = tree[t.pieces[b.bid][-1]].radius_dist
            rc = [tree[t.pieces[c][0]].radius_prox for c in b.children]
            def f(k, rp=rp, rc=rc):
                return rp ** k - sum(r ** k for r in rc)
            k = None
            if f(1.0) < 0 < f(6.0):
                k = brentq(f, 1.0, 6.0)
            rows.append(dict(side=t.side, branch=b.bid, n_children=len(rc),
                             r_parent_mm=rp * 1e3, r_children_mm=[r * 1e3 for r in rc], k=k))
    return rows


def model_from_tree(tree: CoronaryTree, closure: str = "taper",
                    q_rest: float | None = None, reference_fit: bool = True) -> Model:
    """Calibrated model of a patient tree (resting inflow ``q_rest``, m^3/s)."""
    if reference_fit:
        tree = monotone_reference(tree)
    outlets = coronary_bc.build_outlets(tree, None, closure)
    steady_solver.calibrate(tree, outlets, target_flow=q_rest)
    rest = steady_solver.solve_steady(tree, outlets, hyperemic=False)
    return Model(tree=tree, outlets=outlets, rest=rest, closure=closure,
                 murray_exponent=C.MURRAY_EXPONENT)


def path_tree(tree: CoronaryTree, path: list[str]) -> CoronaryTree:
    """The unbranched vessel made of the segments ``path`` (ostium first)."""
    segs = {}
    for i, sid in enumerate(path):
        s = tree[sid]
        segs[sid] = Segment(seg_id=sid, name=s.name, clinical_name=s.clinical_name,
                            side=s.side, parent_id=s.parent_id if i == 0 else path[i - 1],
                            length=s.length, radius_prox=s.radius_prox,
                            radius_dist=s.radius_dist, is_terminal=(i == len(path) - 1))
    return CoronaryTree(segs)


def dominant_path(m: Model, root_sid: str) -> list[str]:
    """From the ostium, follow at every bifurcation the child with most flow."""
    path = [root_sid]
    while m.tree[path[-1]].children:
        path.append(max(m.tree[path[-1]].children, key=lambda c: m.rest.flow[c]))
    return path


def write_tree_csv(tree: CoronaryTree, path: str | Path, source: str) -> None:
    """Save a tree in the column format of ``data/coronary_tree_geometry.csv``."""
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["segment_id", "segment_name", "clinical_name", "side", "parent_id",
                    "length_mm", "radius_prox_mm", "radius_dist_mm", "is_terminal",
                    "geometry_source"])
        for sid in tree.ordered_ids():
            s = tree[sid]
            w.writerow([sid, s.name, s.clinical_name, s.side, s.parent_id,
                        f"{s.length * 1e3:.4f}", f"{s.radius_prox * 1e3:.4f}",
                        f"{s.radius_dist * 1e3:.4f}", int(s.is_terminal), source])
