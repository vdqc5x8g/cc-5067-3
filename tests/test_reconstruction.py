"""Reconstruction: declared flat-magnitude allpass response, verified with
sweeps, impulses and band-boundary dual tones. No arbitrary LP/HP stacking.
"""

import math

from util import chirp, db, dft, maxdiff, noise_like, run, tone

from mbdyn import MultibandEngine
from mbdyn.crossover import AllpassCascade, MultibandCrossover

FS = 24000.0
XOVER = [400.0, 1600.0]


def test_analytic_sum_is_flat_allpass():
    xo = MultibandCrossover(XOVER, FS)
    ref = AllpassCascade(XOVER, FS)
    worst_mag = 0.0
    worst_phase = 0.0
    for i in range(400):
        f = 20.0 * (FS / 2 / 20.0) ** (i / 399)
        w = math.tau * f / FS
        hs = xo.H_sum(w)
        hr = ref.H(w)
        worst_mag = max(worst_mag, abs(abs(hs) - 1.0))
        # sum of band responses must equal the cascade of stage allpasses
        worst_phase = max(worst_phase, abs(hs - hr))
    assert worst_mag < 1e-9, f"|H_sum| deviates by {worst_mag}"
    assert worst_phase < 1e-9, f"H_sum != cascade of allpasses: {worst_phase}"


def test_no_dips_or_peaks_near_crossovers():
    # dense grid straddling each crossover: the declared response must not
    # sag or bump the way a naive LP/HP stack would
    xo = MultibandCrossover([200.0, 800.0, 3200.0], FS)
    for fc in (200.0, 800.0, 3200.0):
        for k in range(-40, 41):
            f = fc * 1.1 ** (k / 8)
            mag = abs(xo.H_sum(math.tau * f / FS))
            assert abs(mag - 1.0) < 1e-9, f"dip/peak at {f:.1f} Hz: {mag}"


def test_unity_gain_time_domain_noise():
    eng = MultibandEngine(FS, XOVER)
    x = noise_like(6000)
    y = run(eng, x)
    ref = AllpassCascade(XOVER, FS).process(x)
    err = maxdiff(y, ref)
    assert err < 1e-9, f"unity-gain reconstruction error {err}"


def test_impulse_reconstruction():
    eng = MultibandEngine(FS, XOVER)
    x = [1.0] + [0.0] * 4095
    y = run(eng, x)
    ref = AllpassCascade(XOVER, FS).process(x)
    err = maxdiff(y, ref)
    assert err < 1e-9, f"impulse reconstruction error {err}"
    # impulse is scattered by the allpass phase, not passed through raw
    assert maxdiff(y, x) > 0.1


def test_sweep_reconstruction():
    eng = MultibandEngine(FS, XOVER)
    x = chirp(FS, 20.0, 11000.0, 0.8, 0.5)
    y = run(eng, x)
    ref = AllpassCascade(XOVER, FS).process(x)
    err = maxdiff(y, ref)
    assert err < 1e-9, f"sweep reconstruction error {err}"


def test_dual_tone_at_band_boundaries():
    eng = MultibandEngine(FS, XOVER)
    eng.probe_bands = True
    f1, f2 = XOVER
    x = [0.4 * math.sin(math.tau * f1 * i / FS) + 0.4 * math.sin(math.tau * f2 * i / FS)
         for i in range(int(FS * 0.4))]
    run(eng, x)
    skip = int(FS * 0.1)
    for k, tr in enumerate(eng.band_trace):
        tr = tr[skip:]
        a1, a2 = abs(dft(tr, FS, f1)), abs(dft(tr, FS, f2))
        if k == 0:
            assert abs(db(a1 / 0.4) - (-6.02)) < 0.4, f"band0 at f1: {db(a1/0.4)}"
            assert db(a2 / 0.4) < -20.0, f"band0 leaks f2: {db(a2/0.4)}"
        elif k == 1:
            assert abs(db(a1 / 0.4) - (-6.02)) < 0.4, f"band1 at f1: {db(a1/0.4)}"
            assert abs(db(a2 / 0.4) - (-6.02)) < 0.4, f"band1 at f2: {db(a2/0.4)}"
        else:
            assert db(a1 / 0.4) < -20.0, f"band2 leaks f1: {db(a1/0.4)}"
            assert abs(db(a2 / 0.4) - (-6.02)) < 0.4, f"band2 at f2: {db(a2/0.4)}"
    # vector sum of band phasors at each boundary tone = unity (allpass)
    for f in (f1, f2):
        s = sum(dft(tr[skip:], FS, f) for tr in eng.band_trace)
        assert abs(abs(s) - 0.4) < 1e-6, f"phasor sum at {f} Hz: {abs(s)}"
