"""Patient-tree extraction on a synthetic lumen of known geometry."""

from __future__ import annotations

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("skimage")
pytest.importorskip("networkx")

from coronary import patient as P  # noqa: E402

SPACING = 0.25      # mm
# trunk (2.0 mm) splits into daughters of 1.6 and 1.2 mm: 2^2 = 1.6^2 + 1.2^2,
# so Murray's law holds exactly with k = 2.
TRUNK = ((15.0, 15.0, 4.0), (15.0, 15.0, 22.0), 2.0)
DAUGHTERS = [((15.0, 15.0, 22.0), (7.0, 15.0, 44.0), 1.6),
             ((15.0, 15.0, 22.0), (23.0, 15.0, 44.0), 1.2)]


def _tube(shape, a, b, r):
    g = np.indices(shape).reshape(3, -1).T * SPACING
    a, b = np.asarray(a), np.asarray(b)
    t = np.clip((g - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
    d = np.linalg.norm(g - (a + t[:, None] * (b - a)), axis=1)
    return (d <= r).reshape(shape)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    shape = (120, 120, 196)
    mask = _tube(shape, *TRUNK)
    for seg in DAUGHTERS:
        mask |= _tube(shape, *seg)
    path = tmp_path_factory.mktemp("lumen") / "y.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.diag([SPACING] * 3 + [1.0])), path)
    trees = P.trees_from_mask(path)
    return trees, P.to_coronary_tree(trees)


def test_one_tree_with_a_trunk_and_two_daughters(synthetic):
    trees, _ = synthetic
    assert len(trees) == 1
    t = trees[0]
    root = t.branches[t.root]
    assert len(root.children) == 2
    assert all(not t.branches[c].children for c in root.children)


def test_ostium_is_the_wide_end_of_the_trunk(synthetic):
    trees, _ = synthetic
    assert trees[0].ostium_world_mm[2] < 8.0          # trunk starts at z = 4 mm


def test_radii_are_recovered(synthetic):
    trees, tree = synthetic
    t = trees[0]
    trunk = tree[t.pieces[t.root][0]]
    assert trunk.radius_prox * 1e3 == pytest.approx(2.0, rel=0.08)
    got = sorted(tree[t.pieces[c][0]].radius_prox * 1e3 for c in t.branches[t.root].children)
    assert got == pytest.approx([1.2, 1.6], rel=0.10)


def test_murray_exponent_is_recovered(synthetic):
    trees, tree = synthetic
    (row,) = P.murray_exponents(trees, tree)
    assert row["k"] == pytest.approx(2.0, abs=0.35)


def test_main_vessel_follows_the_larger_daughter(synthetic):
    trees, tree = synthetic
    t = trees[0]
    m = P.model_from_tree(tree, "taper", q_rest=1e-6)
    path = P.dominant_path(m, t.pieces[t.root][0])
    big = max(t.branches[t.root].children, key=lambda c: np.median(t.branches[c].r))
    assert path[-1] == t.pieces[big][-1]
