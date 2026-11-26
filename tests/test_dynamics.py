"""Per-band dynamics: static curve, timing, independent LF/HF control,
and state continuity across audio blocks.
"""

import math

from util import db, lf_hf_material, maxdiff, run, tone

from mbdyn import MultibandEngine, static_gain_db
from mbdyn.dynamics import BandConfig, BandDynamics

FS = 24000.0


def test_static_curve_points():
    # hard knee
    assert static_gain_db(-30.0, -18.0, 4.0, 0.0) == 0.0
    assert abs(static_gain_db(-6.0, -18.0, 4.0, 0.0) - (-9.0)) < 1e-12
    # soft knee: below / inside / above
    assert static_gain_db(-22.0, -18.0, 4.0, 6.0) == 0.0
    assert abs(static_gain_db(-15.0, -18.0, 4.0, 6.0) - (-2.25)) < 1e-12
    assert abs(static_gain_db(-6.0, -18.0, 4.0, 6.0) - (-9.0)) < 1e-12


def test_attack_release_timing():
    cfg = BandConfig()
    cfg.attack_ms = 5.0
    cfg.release_ms = 50.0
    det = BandDynamics(FS, cfg)
    n_att = int(FS * cfg.attack_ms * 1e-3)
    for _ in range(n_att):
        det.track_only(1.0)
    assert abs(det.env - (1.0 - math.exp(-1.0))) < 0.02, det.env
    n_rel = int(FS * cfg.release_ms * 1e-3)
    env0 = det.env
    for _ in range(n_rel):
        det.track_only(0.0)
    assert abs(det.env - env0 * math.exp(-1.0)) < 0.02, det.env


def _seg_avg(trace, fs, t0, t1):
    seg = trace[int(t0 * fs):int(t1 * fs)]
    return sum(seg) / len(seg)


def test_independent_lf_hf_control():
    eng = MultibandEngine(FS, [300.0, 3000.0])
    eng.set_band(0, threshold_db=-20.0, ratio=6.0, knee_db=0.0,
                 attack_ms=2.0, release_ms=50.0)
    eng.set_band(2, threshold_db=-6.0, ratio=6.0, knee_db=0.0,
                 attack_ms=2.0, release_ms=50.0)
    eng.probe_gains = True
    x = lf_hf_material(FS, dur=2.0, lf=60.0, hf=6000.0)
    run(eng, x, block=512)
    g_lf, _, g_hf = eng.gain_trace
    # static predictions: level = 20log10(amp); gr = (L-T)(1/R-1)
    lf_loud = (db(0.90) - (-20.0)) * (1.0 / 6.0 - 1.0)
    hf_loud = (db(0.80) - (-6.0)) * (1.0 / 6.0 - 1.0)
    # segment 1 [0, .5): LF quiet, HF loud
    assert abs(_seg_avg(g_lf, FS, 0.25, 0.45)) < 0.5
    assert abs(_seg_avg(g_hf, FS, 0.25, 0.45) - hf_loud) < 1.0
    # segment 2 [.5, 1): LF loud, HF loud -> LF compresses, HF unchanged by LF
    assert abs(_seg_avg(g_lf, FS, 0.65, 0.90) - lf_loud) < 1.0
    assert abs(_seg_avg(g_hf, FS, 0.65, 0.90) - hf_loud) < 1.0
    # segment 3 [1, 1.5): LF loud, HF quiet -> HF releases despite loud LF
    assert abs(_seg_avg(g_hf, FS, 1.30, 1.45)) < 0.5
    assert abs(_seg_avg(g_lf, FS, 1.30, 1.45) - lf_loud) < 1.0
    # segment 4 [1.5, 2): both quiet, both recovered
    assert abs(_seg_avg(g_lf, FS, 1.80, 1.95)) < 0.5
    assert abs(_seg_avg(g_hf, FS, 1.80, 1.95)) < 0.5


def test_envelope_state_crosses_blocks():
    eng1 = MultibandEngine(FS, [300.0, 3000.0])
    eng2 = MultibandEngine(FS, [300.0, 3000.0])
    for e in (eng1, eng2):
        e.set_band(0, threshold_db=-30.0, ratio=4.0, attack_ms=3.0,
                   release_ms=60.0)
    x = tone(FS, 100.0, 0.8, 0.6)
    y1 = run(eng1, x)
    y2 = run(eng2, x, block=97)
    assert y1 == y2, "block segmentation must not change the output"


def test_disabled_band_still_passes_and_tracks():
    eng = MultibandEngine(FS, [300.0, 3000.0])
    eng.set_band(0, threshold_db=-40.0, ratio=8.0)
    eng.set_band(0, enabled=False)
    eng.probe_bands = True
    x = tone(FS, 100.0, 0.6, 0.3)
    y = run(eng, x)
    # band 0 content is present in the sum (not silently deleted)
    assert max(abs(v) for v in y) > 0.3
    # detector kept tracking the disabled band
    assert eng._core.dyns[0].env > 0.3
