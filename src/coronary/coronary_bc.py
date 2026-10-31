"""Zero-dimensional coronary outlet (microvascular) boundary model.

Every segment can drain into a downstream microvascular bed represented by a
lumped element attached to its distal node. Two views of the same bed are
provided:

* Steady view -- a single resistance from the distal node to the coronary
  venous pressure. Used by the steady network solver and for FFR.

* Pulsatile view -- a coronary-specific RC network driven by an
  intramyocardial pressure source P_im(t):

      P_node --R_a-- (node 1, C_a) --R_micro-- (node 2, C_im ref. P_im) --R_v--

  The compression of C_im by the contracting myocardium reproduces the
  diastolic-dominant coronary inflow (see pulsatile_solver).

Resting resistances are distributed across beds by the morphometric flow
split of :func:`murray_flow_weights`, then globally calibrated so the solved
total resting coronary flow matches the physiological target
(see steady_solver.calibrate).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

from . import constants as C
from .geometry import CoronaryTree

# Outlet RC time constant: sets the bed compliance from C_out = TAU / R_term.
# Documented modelling input controlling microvascular waveform damping.
OUTLET_TAU = 0.15        # s  [MODEL INPUT]

# Closures for the distribution of microvascular outflow over the tree.
#   "taper"      morphometric equilibrium Q ~ d^k imposed at every station of
#                the lumen: flow is shed along each tapering segment and at
#                every radius step between a segment and its continuation
#                (default).
#   "segment"    the same law applied inside named segments only; radius steps
#                at segment boundaries pass all flow on (version 1.0 behaviour).
#   "junction"   step-downs at segment junctions only, no shedding along a
#                segment; analogous to a step-down reference diameter built from
#                visible branches (as in Murray-law QFR).
#   "leaf"       no mural outflow; every bed sits at a terminal segment, with
#                flow divided at bifurcations by r^k.
#   "structured" no mural outflow; every terminal segment drains into an
#                Olufsen-type structured tree whose steady resistance sets its
#                share of the flow.
CLOSURES = ("taper", "segment", "leaf")                   # compared along the LAD
ALL_CLOSURES = ("taper", "segment", "junction", "leaf", "structured")

# Structured-tree outlets (Olufsen, Am J Physiol 1999): an asymmetric binary
# tree whose daughters have radii alpha*r and beta*r, each vessel of length
# LRR*r, truncated below a minimum radius. Only relative resistances matter
# here (the global calibration removes the scale), so LRR and the viscosity
# cancel; ALPHA and BETA are Olufsen's values.
STRUCT_ALPHA = 0.9
STRUCT_BETA = 0.6
STRUCT_LRR = 50.0
STRUCT_R_MIN = 10e-6     # m, truncation radius (arteriolar scale) [MODEL INPUT]


@dataclass
class CoronaryOutlet:
    """Lumped microvascular bed attached to the distal node of one segment."""

    segment_id: str
    side: str                       # 'L' or 'R' coronary system
    r_term_rest: float              # total resting resistance (Pa.s/m^3)
    flow_weight: float              # share of total coronary flow at design
    # microvascular resistance reserve of this bed; None uses the global value
    # (constants.CORONARY_FLOW_RESERVE). Lower values model regional
    # microvascular dysfunction.
    reserve: float | None = None

    # pulsatile RC sub-network (populated by build_pulsatile)
    r_a: float = 0.0
    r_micro: float = 0.0
    r_v: float = 0.0
    c_a: float = 0.0
    c_im: float = 0.0

    # ------------------------------------------------------------- steady
    def resistance(self, hyperemic: bool) -> float:
        """Total outlet resistance; hyperaemia divides it by the bed's reserve."""
        if not hyperemic:
            return self.r_term_rest
        reserve = C.CORONARY_FLOW_RESERVE if self.reserve is None else self.reserve
        return self.r_term_rest / reserve

    def steady_flow(self, p_distal: float, hyperemic: bool) -> float:
        """Steady outlet flow for a given distal-node pressure."""
        return (p_distal - C.P_VENOUS) / self.resistance(hyperemic)

    # ---------------------------------------------------------- pulsatile
    def build_pulsatile(self, hyperemic: bool) -> None:
        """Split the total resistance and compliance over the RC sub-network."""
        r_tot = self.resistance(hyperemic)
        self.r_a = C.R_FRACTION_PROXIMAL * r_tot
        self.r_micro = C.R_FRACTION_MICRO * r_tot
        self.r_v = C.R_FRACTION_VENOUS * r_tot
        c_out = OUTLET_TAU / r_tot
        self.c_a = C.C_FRACTION_ARTERIAL * c_out
        self.c_im = C.C_FRACTION_INTRAMYO * c_out


def murray_flow_weights(tree: CoronaryTree,
                        murray_exponent: float | None = None,
                        closure: str = "taper",
                        ) -> tuple[dict[str, float], dict[str, float]]:
    """Morphometric flow distribution over the coronary tree.

    Murray's principle Q = kappa * d^k is imposed pointwise along the lumen.
    For a segment of proximal radius r_p and distal radius r_d this gives

    * along the segment, a pass-through fraction alpha = (r_d / r_p)^k, the
      exact integral of the local shedding law q' = -k Q r'/r, with the
      remainder 1 - alpha shed to the segment's mural bed;
    * at the distal node, a pass-through fraction
      beta = min(sum_c r_{p,c}^k / r_d^k, 1) over the daughters c, so that a
      radius step to a narrower continuation sheds flow to the unnamed
      branches that leave there;
    * the flow passed on is divided among daughters in proportion to r_{p,c}^k.

    Total flow is divided between the two ostia by inlet radius^k. Both
    fractions are capped at one, so no flow is created where the lumen widens.
    For a chain of segments whose radius never increases, the product of the
    fractions telescopes to (r_end / r_start)^k, so the result does not
    depend on how the vessel is divided into named segments.

    Returns
    -------
    (segment_flow, outlet_flow) as fractions of total coronary flow:
    the flow *entering* each segment, and the flow drained by the bed at
    each segment's distal node. The outlet fractions sum to one exactly.
    """
    if closure not in ALL_CLOSURES:
        raise ValueError(f"unknown closure {closure!r}; expected one of {ALL_CLOSURES}")
    if closure == "structured":
        return structured_tree_weights(tree)
    # resolved at call time so a sweep that perturbs the constant reaches here
    k = C.MURRAY_EXPONENT if murray_exponent is None else murray_exponent
    seg_flow: dict[str, float] = {}
    outlet_flow: dict[str, float] = {}

    inlets = tree.inlets()
    inlet_raw = {sid: tree[sid].radius_prox ** k for sid in inlets}
    inlet_sum = sum(inlet_raw.values())
    for sid in inlets:
        seg_flow[sid] = inlet_raw[sid] / inlet_sum

    for sid in tree.ordered_ids():
        seg = tree[sid]
        q_in = seg_flow[sid]
        if seg.is_terminal:
            outlet_flow[sid] = q_in                      # leaf drains all inflow
            continue
        raw = {c: tree[c].radius_prox ** k for c in seg.children}
        raw_sum = sum(raw.values())
        alpha = (min((seg.radius_dist / seg.radius_prox) ** k, 1.0)
                 if closure in ("taper", "segment") else 1.0)
        beta = (min(raw_sum / seg.radius_dist ** k, 1.0)
                if closure in ("taper", "junction") else 1.0)
        passed = q_in * alpha * beta
        outlet_flow[sid] = q_in - passed                 # mural + junction shedding
        for c in seg.children:
            seg_flow[c] = passed * raw[c] / raw_sum
    return seg_flow, outlet_flow


def structured_tree_resistance(r_root: float, alpha: float = STRUCT_ALPHA,
                               beta: float = STRUCT_BETA, lrr: float = STRUCT_LRR,
                               r_min: float = STRUCT_R_MIN,
                               mu: float | None = None) -> float:
    """Steady (zero-frequency) resistance of an Olufsen structured tree.

    R(r) = R_vessel(r) + [1/R(alpha r) + 1/R(beta r)]^-1 with the Poiseuille
    resistance R_vessel = 8 mu (lrr r) / (pi r^4) of a vessel of length lrr*r;
    vessels narrower than r_min end the tree with zero resistance.
    """
    mu = C.MU_BLOOD if mu is None else mu

    @cache
    def resistance(i: int, j: int) -> float:
        r = r_root * alpha ** i * beta ** j
        if r < r_min:
            return 0.0
        own = 8.0 * mu * lrr / (math.pi * r ** 3)
        ra, rb = resistance(i + 1, j), resistance(i, j + 1)
        parallel = 0.0 if ra == 0.0 or rb == 0.0 else 1.0 / (1.0 / ra + 1.0 / rb)
        return own + parallel

    return resistance(0, 0)


def structured_tree_weights(tree: CoronaryTree) -> tuple[dict[str, float], dict[str, float]]:
    """Flow shares for structured-tree outlets at the terminal segments.

    Each terminal bed draws in proportion to the conductance of a structured
    tree rooted at its distal radius; there is no mural outflow. A segment's
    share of the total is the sum of its downstream terminal shares.
    """
    conductance = {s.seg_id: 1.0 / structured_tree_resistance(s.radius_dist)
                   for s in tree if s.is_terminal}
    total = sum(conductance.values())
    outlet_flow = {sid: 0.0 for sid in tree.segments}
    outlet_flow.update({sid: g / total for sid, g in conductance.items()})
    seg_flow: dict[str, float] = {}
    for sid in reversed(tree.ordered_ids()):
        seg = tree[sid]
        seg_flow[sid] = outlet_flow[sid] + sum(seg_flow[c] for c in seg.children)
    return seg_flow, outlet_flow


def build_outlets(tree: CoronaryTree,
                  murray_exponent: float | None = None,
                  closure: str = "taper",
                  ) -> dict[str, CoronaryOutlet]:
    """Create one microvascular outlet per draining segment.

    Each bed receives a share w_i of the total resting coronary flow
    (:func:`murray_flow_weights`); its resting resistance is
    R = (MAP - P_v) / (w_i Q_rest). A subsequent global calibration
    (steady_solver.calibrate) rescales all resistances so the *solved* total
    flow equals the physiological target.
    """
    _, outlet_weight = murray_flow_weights(tree, murray_exponent, closure)
    outlets: dict[str, CoronaryOutlet] = {}
    for sid, w in outlet_weight.items():
        # a segment that sheds nothing (untapered, no radius step) has no bed
        if w < 1e-9:
            continue
        q_i = w * C.Q_CORONARY_REST
        r_term = (C.P_AO_MEAN - C.P_VENOUS) / q_i
        outlets[sid] = CoronaryOutlet(
            segment_id=sid, side=tree[sid].side, r_term_rest=r_term, flow_weight=w)
    return outlets
