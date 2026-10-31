"""Periodic driving waveforms: aortic, ventricular and intramyocardial pressure.

These are prescribed model inputs. The aortic pressure is the network inlet
boundary condition; the intramyocardial pressure drives the 0D coronary
outlets. Each waveform is a fully specified analytic function of cardiac phase,
parameterised only by physiological pressures and timing from ``constants``.
"""

from __future__ import annotations

import numpy as np

from . import constants as C

# Shape parameters of the analytic aortic pressure waveform [MODEL INPUT].
_AO_RISE_FRACTION = 0.40      # fraction of systole spent on the upstroke
_AO_NOTCH_FRACTION = 0.40     # dicrotic-notch height as a fraction of pulse pressure
_AO_DIASTOLIC_TAU = 0.30      # diastolic decay constant (fraction of the cycle)


def _phase(t: np.ndarray) -> np.ndarray:
    """Cardiac phase in [0, 1)."""
    return np.mod(np.asarray(t, dtype=float) / C.T_CYCLE, 1.0)


def aortic_pressure(t: np.ndarray | float) -> np.ndarray:
    """Aortic (coronary ostial) pressure waveform (Pa).

    Piecewise-analytic: raised-cosine systolic upstroke and downstroke to a
    dicrotic notch, followed by an exponential diastolic runoff whose asymptote
    is fixed by periodic continuity (p at end-diastole = diastolic pressure).
    """
    phi = _phase(t)
    sbp, dbp = C.P_AO_SYS, C.P_AO_DIA
    pp = sbp - dbp
    fs = C.SYSTOLE_FRACTION
    phi_peak = fs * _AO_RISE_FRACTION
    p_notch = dbp + _AO_NOTCH_FRACTION * pp

    # diastolic-runoff asymptote enforcing p(phi=1) = dbp
    decay_end = np.exp(-(1.0 - fs) / _AO_DIASTOLIC_TAU)
    p_inf = (dbp - p_notch * decay_end) / (1.0 - decay_end)

    p = np.empty_like(phi)

    up = phi < phi_peak
    p[up] = dbp + pp * 0.5 * (1.0 - np.cos(np.pi * phi[up] / phi_peak))

    down = (phi >= phi_peak) & (phi < fs)
    u = (phi[down] - phi_peak) / (fs - phi_peak)
    p[down] = p_notch + (sbp - p_notch) * 0.5 * (1.0 + np.cos(np.pi * u))

    dia = phi >= fs
    p[dia] = p_inf + (p_notch - p_inf) * np.exp(-(phi[dia] - fs) / _AO_DIASTOLIC_TAU)
    return p


def _ventricular_pressure(t: np.ndarray | float, p_sys: float, p_dia: float) -> np.ndarray:
    """Generic ventricular pressure: a sinusoidal systolic bump, low in diastole."""
    phi = _phase(t)
    fs = C.SYSTOLE_FRACTION
    p = np.full_like(phi, p_dia)
    sys = phi < fs
    p[sys] = p_dia + (p_sys - p_dia) * np.sin(np.pi * phi[sys] / fs)
    return p


def lv_pressure(t: np.ndarray | float) -> np.ndarray:
    """Left-ventricular pressure waveform (Pa)."""
    return _ventricular_pressure(t, C.P_LV_SYS_MMHG * C.MMHG_TO_PA,
                                 C.P_LV_DIA_MMHG * C.MMHG_TO_PA)


def rv_pressure(t: np.ndarray | float) -> np.ndarray:
    """Right-ventricular pressure waveform (Pa)."""
    return _ventricular_pressure(t, C.P_RV_SYS_MMHG * C.MMHG_TO_PA,
                                 C.P_RV_DIA_MMHG * C.MMHG_TO_PA)


def intramyocardial_pressure(t: np.ndarray | float, ventricle: str = "LV",
                             im_factor: float | None = None) -> np.ndarray:
    """Intramyocardial pressure driving a coronary bed (Pa).

    A transmural-average fraction of the surrounding ventricular pressure.
    """
    factor = C.IM_PRESSURE_FACTOR if im_factor is None else im_factor
    vp = lv_pressure(t) if ventricle.upper() == "LV" else rv_pressure(t)
    return factor * vp


def cycle_time_grid(n: int) -> np.ndarray:
    """Uniform time grid over one cardiac cycle (excludes the periodic endpoint)."""
    return np.linspace(0.0, C.T_CYCLE, n, endpoint=False)


def harmonics(signal: np.ndarray) -> np.ndarray:
    """Complex Fourier harmonics of a one-cycle signal (index 0 = mean)."""
    n = len(signal)
    return np.fft.rfft(signal) / n


def reconstruct(harm: np.ndarray, n: int) -> np.ndarray:
    """Inverse of :func:`harmonics` back to an n-sample one-cycle signal."""
    return np.fft.irfft(harm * n, n=n)
