"""Engine semantics: bypass / processing switch / solo / mute / band-disable,
delay alignment, limits, block invariance, drum-hit cross-block recovery,
and the bypass-recording vs live-path sweep fixture.
"""

import math

from util import chirp, db, dft, drum, maxdiff, noise_like, run, tone

from mbdyn import MultibandEngine
from mbdyn.crossover import AllpassCascade

FS = 24000.0
XOVER = [300.0, 3000.0]


def test_bypass_is_bit_exact_and_records():
    eng = MultibandEngine(FS, XOVER)
    eng.set_bypass(True)
    eng.record_input = True
    x = noise_like(5000)
    y = run(eng, x, block=313)
    assert y == x, "bypass must be bit-exact"
    assert eng.recorded == x, "bypass recording must capture the same stream"


def test_processing_off_is_declared_allpass():
    eng = MultibandEngine(FS, XOVER)
    eng.set_processing(False)
    x = noise_like(4000)
    y = run(eng, x)
    ref = AllpassCascade(XOVER, FS).process(x)
    assert maxdiff(y, ref) < 1e-9
    eng.set_bypass(True)
    assert run(eng, x) == x, "bypass and processing-off are distinct"


def test_band_disable_keeps_its_content():
    x = tone(FS, 800.0, 0.9, 0.3)  # lives in band 1
    ref = AllpassCascade(XOVER, FS).process(x)
    eng = MultibandEngine(FS, XOVER)
    eng.set_band(1, threshold_db=-40.0, ratio=8.0)  # would compress hard
    eng.set_band(1, enabled=False)
    y = run(eng, x)
    assert maxdiff(y, ref) < 1e-9, "disabled band lost content or changed sum"
    eng2 = MultibandEngine(FS, XOVER)
    eng2.set_band(1, threshold_db=-40.0, ratio=8.0)
    y2 = run(eng2, x)
    assert maxdiff(y2, ref) > 0.01, "control run: compression never engaged"


def test_solo_and_mute_semantics():
    x = noise_like(4000)
    eng = MultibandEngine(FS, XOVER)
    eng.set_band(0, threshold_db=-30.0, ratio=4.0)
    eng.probe_bands = True
    full = run(eng, x)
    bands = [list(tr) for tr in eng.band_trace]
    for k in range(eng.num_bands):
        solo_eng = MultibandEngine(FS, XOVER)
        solo_eng.set_band(0, threshold_db=-30.0, ratio=4.0)
        solo_eng.set_solo(k, True)
        y = run(solo_eng, x)
        assert maxdiff(y, bands[k]) < 1e-12, f"solo {k} != band {k} signal"
    # mute k == full minus band k
    for k in range(eng.num_bands):
        mute_eng = MultibandEngine(FS, XOVER)
        mute_eng.set_band(0, threshold_db=-30.0, ratio=4.0)
        mute_eng.set_mute(k, True)
        y = run(mute_eng, x)
        expect = [f - b for f, b in zip(full, bands[k])]
        assert maxdiff(y, expect) < 1e-9, f"mute {k} inconsistent"


def test_block_size_invariance():
    x = noise_like(3000)
    outs = []
    for block in (1, 97, 511, 0):
        eng = MultibandEngine(FS, XOVER)
        eng.set_band(1, threshold_db=-25.0, ratio=3.0, attack_ms=2.0)
        outs.append(run(eng, x, block=block))
    for o in outs[1:]:
        assert o == outs[0], "per-band state must continue across blocks"


def test_limits_and_sample_rate_validation():
    MultibandEngine(16000.0, [100.0])                    # 2 bands ok
    MultibandEngine(44100.0, [200.0, 1000.0, 5000.0])    # 4 bands ok
    for fs, xo in ((1000.0, [100.0]), (200000.0, [100.0]),
                   (FS, [30.0]), (FS, [0.5 * FS]), (FS, [0.46 * FS]),
                   (FS, [400.0, 500.0]), (FS, [900.0, 400.0]),
                   (FS, [400.0, 900.0, 2000.0, 5000.0])):
        try:
            MultibandEngine(fs, xo)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted fs={fs} crossovers={xo}")


def test_crossover_tracks_sample_rate():
    for fs in (16000.0, 44100.0):
        eng = MultibandEngine(fs, [400.0, 2000.0])
        eng.probe_bands = True
        run(eng, tone(fs, 400.0, 0.5, 0.25))
        tr0 = eng.band_trace[0][int(0.1 * fs):]
        tr1 = eng.band_trace[1][int(0.1 * fs):]
        assert abs(db(abs(dft(tr0, fs, 400.0)) / 0.5) - (-6.02)) < 0.5
        assert abs(db(abs(dft(tr1, fs, 400.0)) / 0.5) - (-6.02)) < 0.5


def test_delay_alignment_and_latency():
    n = 2048
    eng = MultibandEngine(FS, XOVER, lookahead_ms=2.0)
    L = eng.latency_samples
    assert L == int(0.002 * FS) == 48
    x = [1.0] + [0.0] * (n - 1)
    y = run(eng, x)
    ref = AllpassCascade(XOVER, FS).process(x)
    assert max(abs(v) for v in y[:L]) == 0.0, "lookahead leaked early"
    assert maxdiff(y[L:], ref[:n - L]) < 1e-9, "bands not delay-aligned"


def test_drum_hit_cross_block_recovery():
    x = drum(FS, dur=1.0, hits=(0.05,))
    def make():
        eng = MultibandEngine(FS, [250.0, 2500.0])
        eng.set_band(0, threshold_db=-18.0, ratio=3.0, knee_db=0.0,
                     attack_ms=25.0, release_ms=200.0)
        eng.probe_gains = True
        return eng
    eng_whole = make()
    y_whole = run(eng_whole, x)
    eng_chunk = make()
    y_chunk = run(eng_chunk, x, block=113)  # boundaries straddle the hit
    assert y_whole == y_chunk, "cross-block recovery decision changed output"

    i0 = int(0.05 * FS)
    win = slice(i0, i0 + int(0.012 * FS))
    # first peak after real summing, relative to the unity-gain allpass
    # output (isolates the compressor attack from allpass dispersion)
    unity = MultibandEngine(FS, [250.0, 2500.0])
    y_unity = run(unity, x)
    peak_ref = max(abs(v) for v in y_unity[win])
    peak_out = max(abs(v) for v in y_whole[win])
    ratio = peak_out / peak_ref
    assert 0.85 < ratio < 1.15, f"first peak vs unity reference: {ratio}"

    g = eng_whole.gain_trace[0]
    assert min(g) < -4.0, f"LF dip never bit: {min(g)}"
    assert g[-1] > -1.0, f"LF gain did not recover across blocks: {g[-1]}"


def test_same_sweep_to_bypass_recording_and_live_path():
    x = chirp(FS, 30.0, 11000.0, 0.8, 0.6)
    rec_eng = MultibandEngine(FS, XOVER)
    rec_eng.set_bypass(True)
    rec_eng.record_input = True
    live = MultibandEngine(FS, XOVER, warmup_frames=8192)
    block = 256
    moves = ([500.0, 3000.0], [800.0, 3000.0], [500.0, 2500.0])
    y_live = []
    y_rec = []
    for bi, i in enumerate(range(0, len(x), block)):
        chunk = x[i:i + block]
        y_rec.extend(rec_eng.process(chunk))
        if bi == 4:
            live.move_crossovers(moves[0], policy="deferred")
        if bi == 8:
            live.move_crossovers(moves[1], policy="deferred")
        if bi == 12:
            live.move_crossovers(moves[2], policy="deferred")
        y_live.extend(live.process(chunk))
    assert y_rec == x and rec_eng.recorded == x
    peak = max(abs(v) for v in x)
    assert max(abs(v) for v in y_live) < 1.3 * peak
    logged = [e for e in live.switch_log if e["policy"] == "deferred"]
    assert len(logged) == 3, live.switch_log
    assert live.freqs == list(moves[2])
