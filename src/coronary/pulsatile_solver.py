"""Pulsatile 1D pulse-wave solver coupled to the 0D coronary outlets.

The pulsatile component solves the *linearised* one-dimensional blood-flow
equations in the frequency domain. About the reference state each segment is a
viscoelastic transmission line governed by the telegrapher's equations

    dp/dx = -(R + i w L) q ,      dq/dx = -(i w C) p                    (1)

with per-unit-length resistance R = 8*pi*mu/A0^2, inertance L = rho/A0 and
compliance C = dA/dp. The propagation constant and characteristic impedance

    gamma = sqrt((R + i w L)(i w C)) ,  Z0 = sqrt((R + i w L)/(i w C))   (2)

give each segment a two-port admittance matrix. The coronary tree is assembled
by nodal analysis; terminal nodes carry the 0D coronary outlet, which is a
linear sub-network excited both by the epicardial pressure and by the
intramyocardial pressure source. The aortic pressure harmonics drive the
inlet. Superposing harmonics reconstructs the time-domain waveforms; the cycle
mean is supplied by the (nonlinear) steady solver.

The neglected convective term scales with u/c, which stays below 0.05 in
the resting tree (largest in the left main at peak diastolic flow), so the
linearisation is well justified for pulse-wave propagation; the nonlinear,
flow-limiting behaviour of stenoses is retained in the steady/FFR component.

All beds are driven by the left-ventricular intramyocardial pressure. The
seventeen-segment tree resolves no right-ventricular free-wall branches; the
right coronary trunk sheds under 5 % of its flow along its length, and its
daughters perfuse the inferior left ventricle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from . import rheology, wall, waveforms
from ._compat import trapezoid
from .coronary_bc import CoronaryOutlet
from .geometry import CoronaryTree
from .steady_solver import SteadySolution


@dataclass
class PulsatileSolution:
    time: np.ndarray                       # s, one cardiac cycle
    pressure: dict[str, np.ndarray]        # Pa, distal-node pressure per segment
    flow: dict[str, np.ndarray]            # m^3/s, flow per segment
    inflow_total: np.ndarray               # m^3/s, summed coronary inlet flow
    aortic_pressure: np.ndarray            # Pa, inlet waveform
    wave_speed: dict[str, float]           # m/s, reference wave speed per segment

    def flow_ml_min(self, seg_id: str) -> np.ndarray:
        return self.flow[seg_id] * C.M3_PER_S_TO_ML_PER_MIN

    def _systole_mask(self) -> np.ndarray:
        return self.time < C.SYSTOLE_FRACTION * C.T_CYCLE

    def diastolic_flow_fraction(self, seg_id: str) -> float:
        """Share of segment flow delivered during diastole."""
        sys = self._systole_mask()
        q = self.flow[seg_id]
        total = trapezoid(q, self.time)
        if abs(total) < 1e-20:
            return float("nan")
        dia = trapezoid(np.where(~sys, q, 0.0), self.time)
        return float(dia / total)

    def diastolic_systolic_ratio(self, seg_id: str) -> float:
        """Mean diastolic over mean systolic flow (DSVR for a fixed lumen).

        With the lumen area held at its reference value in the linear model,
        the flow ratio equals the diastolic-to-systolic velocity ratio
        reported by intracoronary Doppler studies.
        """
        sys = self._systole_mask()
        q = self.flow[seg_id]
        return float(np.mean(q[~sys]) / np.mean(q[sys]))

    def max_mach(self, tree: CoronaryTree) -> float:
        """Largest ratio of mean cross-sectional velocity to wave speed."""
        out = 0.0
        for sid, q in self.flow.items():
            area = np.pi * tree[sid].radius_mean ** 2
            out = max(out, float(np.max(np.abs(q))) / area / self.wave_speed[sid])
        return out


def _segment_two_port(omega: float, length: float, r_per_len: float,
                      l_per_len: float, c_per_len: float
                      ) -> tuple[complex, complex, complex, complex]:
    """Two-port admittance entries (Y11, Y12, Y21, Y22) of one segment, eq. (1)-(2)."""
    series = r_per_len + 1j * omega * l_per_len
    shunt = 1j * omega * c_per_len
    gamma = np.sqrt(series * shunt)
    z0 = np.sqrt(series / shunt)
    gl = gamma * length
    sinh, cosh = np.sinh(gl), np.cosh(gl)
    y_series = 1.0 / (z0 * sinh)          # = -Y12
    y_self = cosh / (z0 * sinh)           # = Y11 = Y22
    return y_self, -y_series, -y_series, y_self


def _outlet_admittance(outlet: CoronaryOutlet, omega: float) -> tuple[complex, complex]:
    """Outlet flow Q_in = Y_oo*P_term + Y_os*P_im : return (Y_oo, Y_os).

    Intramyocardial-pump formulation: the microvascular bed (node 2, compliance
    C_im) and its venous drainage discharge against the intramyocardial
    pressure P_im. When the myocardium contracts P_im rises, node 2 rises with
    it and the trans-bed gradient -- hence coronary inflow -- collapses, which
    reproduces the systolic flow impediment of the left coronary circulation.

    Node equations (oscillatory part; P_v is a DC quantity):
      node 1: (P_term-P1)/R_a - P1*(iw*C_a) - (P1-P2)/R_micro = 0
      node 2: (P1-P2)/R_micro - (P2-P_im)*(iw*C_im) - (P2-P_im)/R_v = 0
    """
    g_a = 1.0 / outlet.r_a
    g_m = 1.0 / outlet.r_micro
    g_v = 1.0 / outlet.r_v
    mat = np.array([[g_a + g_m + 1j * omega * outlet.c_a, -g_m],
                    [-g_m, g_m + g_v + 1j * omega * outlet.c_im]], dtype=complex)
    # response of P1 to P_term (P_im = 0)
    p1_term = np.linalg.solve(mat, np.array([g_a, 0.0], dtype=complex))[0]
    # response of P1 to P_im (P_term = 0): P_im drives node 2 through R_v and C_im
    p1_im = np.linalg.solve(
        mat, np.array([0.0, g_v + 1j * omega * outlet.c_im], dtype=complex))[0]
    y_oo = (1.0 - p1_term) * g_a
    y_os = -p1_im * g_a
    return y_oo, y_os


def solve_pulsatile(tree: CoronaryTree, outlets: dict[str, CoronaryOutlet],
                    steady: SteadySolution, n_time: int = 400,
                    n_harmonics: int = 24, im_factor: float | None = None,
                    hyperemic: bool = False, rheology_model: str = "newtonian",
                    stiffness_scale: float = 1.0) -> PulsatileSolution:
    """Solve the coupled 1D(linear)+0D pulsatile model over one cardiac cycle."""
    for o in outlets.values():
        o.build_pulsatile(hyperemic=hyperemic)

    seg_ids = list(tree.segments.keys())
    idx = {sid: i for i, sid in enumerate(seg_ids)}
    n_seg = len(seg_ids)

    # per-segment linearised coefficients about the reference geometry
    r_per_len, l_per_len, c_per_len, wave_c = {}, {}, {}, {}
    for sid in seg_ids:
        seg = tree[sid]
        r0 = seg.radius_mean
        a0 = np.pi * r0**2
        mu = C.MU_BLOOD
        if rheology_model == "carreau":
            mu = rheology.effective_viscosity(steady.flow[sid], r0, model="carreau")
        r_per_len[sid] = 8.0 * np.pi * mu / a0**2
        l_per_len[sid] = C.RHO_BLOOD / a0
        c_per_len[sid] = wall.compliance_per_length(r0, stiffness_scale)
        wave_c[sid] = float(wall.wave_speed_c0(r0, stiffness_scale))

    # driving-waveform harmonics
    t = waveforms.cycle_time_grid(n_time)
    p_ao_h = waveforms.harmonics(waveforms.aortic_pressure(t))
    p_im_h = waveforms.harmonics(
        waveforms.intramyocardial_pressure(t, "LV", im_factor))
    n_harmonics = min(n_harmonics, len(p_ao_h) - 1)

    # oscillatory harmonic spectra for distal-node pressure and segment flow
    p_spec = {sid: np.zeros(len(p_ao_h), dtype=complex) for sid in seg_ids}
    q_spec = {sid: np.zeros(len(p_ao_h), dtype=complex) for sid in seg_ids}

    for k in range(1, n_harmonics + 1):
        omega = 2.0 * np.pi * k / C.T_CYCLE
        mat = np.zeros((n_seg, n_seg), dtype=complex)
        rhs = np.zeros(n_seg, dtype=complex)
        p_root = p_ao_h[k]
        two_port = {sid: _segment_two_port(omega, tree[sid].length, r_per_len[sid],
                                           l_per_len[sid], c_per_len[sid])
                    for sid in seg_ids}

        for sid in seg_ids:
            seg = tree[sid]
            y11, y12, y21, y22 = two_port[sid]
            d = idx[sid]
            at_root = seg.parent_id not in tree.segments
            # KCL at the distal node d (segment's distal port)
            mat[d, d] += y22
            if at_root:
                rhs[d] -= y21 * p_root
            else:
                p = idx[seg.parent_id]
                mat[d, p] += y21
                # KCL at the proximal node p (segment's proximal port)
                mat[p, p] += y11
                mat[p, d] += y12

        # every segment's distal node carries a microvascular outlet
        for sid, outlet in outlets.items():
            d = idx[sid]
            y_oo, y_os = _outlet_admittance(outlet, omega)
            mat[d, d] += y_oo
            rhs[d] -= y_os * p_im_h[k]

        p_nodes = np.linalg.solve(mat, rhs)
        for sid in seg_ids:
            p_spec[sid][k] = p_nodes[idx[sid]]
        # proximal flow of each segment = current entering its proximal port
        for sid in seg_ids:
            seg = tree[sid]
            y11, y12, _, _ = two_port[sid]
            p_prox = p_root if seg.parent_id not in tree.segments else p_nodes[idx[seg.parent_id]]
            q_spec[sid][k] = y11 * p_prox + y12 * p_nodes[idx[sid]]

    # reconstruct time domain: oscillatory part + steady (nonlinear) mean
    pressure, flow = {}, {}
    for sid in seg_ids:
        pressure[sid] = waveforms.reconstruct(p_spec[sid], n_time) + steady.p_distal[sid]
        flow[sid] = waveforms.reconstruct(q_spec[sid], n_time) + steady.flow[sid]

    inflow_total = sum(flow[sid] for sid in tree.inlets())
    return PulsatileSolution(time=t, pressure=pressure, flow=flow,
                             inflow_total=inflow_total,
                             aortic_pressure=waveforms.aortic_pressure(t),
                             wave_speed=wave_c)
