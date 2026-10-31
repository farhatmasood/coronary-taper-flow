"""Verification of the discretisation and the solvers against closed forms."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from coronary import constants as C
from coronary import experiments, pulsatile_solver, steady_solver, stenosis, wall
from coronary._compat import trapezoid


def test_tapered_poiseuille_resistance_matches_quadrature(m):
    for seg in m.tree:
        def integrand(x, seg=seg):
            return 1.0 / (seg.radius_prox + (seg.radius_dist - seg.radius_prox)
                          * x / seg.length) ** 4
        numeric = 8 * C.MU_BLOOD / np.pi * quad(integrand, 0, seg.length, epsabs=0,
                                                epsrel=1e-12)[0]
        assert np.isclose(steady_solver.conduit_resistance(seg, C.MU_BLOOD), numeric,
                          rtol=1e-9), seg.seg_id


def test_segment_volume_matches_truncated_cone_quadrature(m):
    for seg in m.tree:
        x = np.linspace(0.0, seg.length, 20001)
        r = seg.radius_prox + (seg.radius_dist - seg.radius_prox) * x / seg.length
        assert np.isclose(seg.volume, trapezoid(np.pi * r**2, x), rtol=1e-6)


@pytest.mark.parametrize("dp", [-2000.0, -1.0, 0.0, 3.0, 500.0, 2.0e4])
def test_flow_inversion_round_trips(dp):
    r_lin, s_quad = 3.0e9, 1.5e15
    q = steady_solver.invert_flow(dp, r_lin, s_quad)
    assert np.isclose(r_lin * q + s_quad * q * abs(q), dp, rtol=1e-10, atol=1e-10)


def test_two_port_is_reciprocal_and_tends_to_poiseuille_at_low_frequency():
    length, r0 = 0.05, 1.5e-3
    a0 = np.pi * r0**2
    r_pl = 8 * np.pi * C.MU_BLOOD / a0**2
    y11, y12, y21, y22 = pulsatile_solver._segment_two_port(
        1e-6, length, r_pl, C.RHO_BLOOD / a0, wall.compliance_per_length(r0))
    assert y12 == y21 and y11 == y22
    assert np.isclose(-1.0 / y12.real, r_pl * length, rtol=1e-3)


def test_wave_speed_follows_the_olufsen_law():
    r0 = 1.5e-3
    eh_r0 = C.WALL_K1 * np.exp(C.WALL_K2 * r0) + C.WALL_K3
    assert np.isclose(wall.wave_speed_c0(r0), np.sqrt(2.0 / 3.0 * eh_r0 / C.RHO_BLOOD))


def test_steady_solves_converge(m):
    lesion = steady_solver.solve_steady(m.tree, m.outlets, [stenosis.Stenosis("LAD", 0.8)],
                                        hyperemic=True)
    for sol in (m.rest, m.hyperaemic(), lesion):
        assert sol.converged
        assert sol.residual_norm < 1e-9 * C.Q_CORONARY_REST
        assert sol.n_evaluations <= 60


def test_mass_is_conserved_at_every_node(m):
    for sol, hyper in ((m.rest, False), (m.hyperaemic(), True)):
        for sid, seg in m.tree.segments.items():
            out = (m.outlets[sid].steady_flow(sol.p_distal[sid], hyper)
                   if sid in m.outlets else 0.0)
            children = sum(sol.flow[c] for c in seg.children)
            assert np.isclose(sol.flow[sid], out + children, rtol=1e-7, atol=1e-13)


def test_harmonic_truncation_error_is_small(m):
    assert experiments.numerics(m)["harmonic_truncation_error"] < 0.02


def test_linearisation_is_justified(m, puls):
    assert puls.max_mach(m.tree) < 0.05


def test_pulsatile_mean_equals_steady_mean(m, puls):
    for sid in m.tree.segments:
        assert np.isclose(np.mean(puls.flow[sid]), m.rest.flow[sid], rtol=1e-9)
