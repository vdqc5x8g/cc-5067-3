"""Biquad sections (RBJ cookbook Butterworth) with TDF-II state.

State mapping: for a sustained tone the steady-state TDF-II state of a
biquad is analytic, so a retuned section can have its memory aligned to
the signal instead of restarting from zero (or suffering a click).
"""

import cmath
import math

SQRT1_2 = math.sqrt(0.5)  # 1/(2Q) for Q = 1/sqrt(2) (Butterworth)


def butter_lp2(fc, fs):
    w0 = math.tau * fc / fs
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw * SQRT1_2
    a0 = 1.0 + alpha
    b0 = (1.0 - cw) * 0.5 / a0
    return (b0, (1.0 - cw) / a0, b0, -2.0 * cw / a0, (1.0 - alpha) / a0)


def butter_hp2(fc, fs):
    w0 = math.tau * fc / fs
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw * SQRT1_2
    a0 = 1.0 + alpha
    b0 = (1.0 + cw) * 0.5 / a0
    return (b0, -(1.0 + cw) / a0, b0, -2.0 * cw / a0, (1.0 - alpha) / a0)


def biquad_H(c, w):
    """Complex response of coeffs c=(b0,b1,b2,a1,a2) at digital freq w (rad)."""
    b0, b1, b2, a1, a2 = c
    z1 = cmath.exp(-1j * w)
    z2 = z1 * z1
    return (b0 + b1 * z1 + b2 * z2) / (1.0 + a1 * z1 + a2 * z2)


class Biquad:
    """Transposed direct form II biquad; state = (s1, s2)."""

    __slots__ = ("c", "s1", "s2")

    def __init__(self, coeffs):
        self.c = coeffs
        self.s1 = 0.0
        self.s2 = 0.0

    def step(self, x):
        b0, b1, b2, a1, a2 = self.c
        y = b0 * x + self.s1
        self.s1 = b1 * x - a1 * y + self.s2
        self.s2 = b2 * x - a2 * y
        return y

    def set_coeffs(self, c):
        self.c = c

    def steady_state(self, w, X, n_next=0):
        """Set TDF-II state to the steady state for input phasor X.

        x[n] = Re(X * e^(j w n)); state is set as it would be just before
        sample n_next, so the next step() output lies on the steady-state
        trajectory of *this* filter's current coefficients.
        """
        b0, b1, b2, a1, a2 = self.c
        Y = biquad_H(self.c, w) * X
        S2 = b2 * X - a2 * Y
        S1 = (b1 * X - a1 * Y) + S2 * cmath.exp(-1j * w)
        e = cmath.exp(1j * w * (n_next - 1))
        self.s1 = (S1 * e).real
        self.s2 = (S2 * e).real
