"""Crossover moves during program material: state mapping vs double-run
crossfade vs deferred band switch, and in-/anti-phase tail-ringing groups.
"""

import math

from util import db, dft, maxdiff, run, tone

from mbdyn import MultibandEngine
from mbdyn.crossover import MultibandCrossover

FS = 24000.0
OLD = [400.0, 2000.0]
NEW = [900.0, 2000.0]
TONE_F = 300.0
TONE_A = 0.5


def _ideal_sin(n0, count, freqs):
    """Steady-state unity-gain output for the global tone after the move."""
    xo = MultibandCrossover(freqs, FS)
    h = xo.H_sum(math.tau * TONE_F / FS)
    ph = math.atan2(h.imag, h.real)
    return [TONE_A * math.sin(math.tau * TONE_F * (n0 + k) / FS + ph)
            for k in range(count)]


def _primed_engine(warmup=8192):
    eng = MultibandEngine(FS, OLD, warmup_frames=warmup)
    run(eng, tone(FS, TONE_F, TONE_A, 0.3))
    return eng


def test_map_vs_xfade_vs_swap_on_sustained_tone():
    n_move = int(FS * 0.3)  # 7200 samples = exactly 90 cycles of 300 Hz
    post = tone(FS, TONE_F, TONE_A, 0.15)
    ideal = _ideal_sin(n_move, len(post), NEW)

    # analytic comb bound for the crossfade: |(1-w)*old + w*new - new|
    # <= |old - new| = TONE_A * |H_old - H_new| at the tone frequency
    w = math.tau * TONE_F / FS
    dh = abs(MultibandCrossover(OLD, FS).H_sum(w)
             - MultibandCrossover(NEW, FS).H_sum(w))
    comb_bound = TONE_A * dh

    results = {}
    xfade_during = xfade_after = None
    for policy in ("map", "xfade", "swap"):
        eng = _primed_engine()
        # tone is sine; the hint phasor uses the cosine convention
        hint = (TONE_F, TONE_A, math.tau * TONE_F * n_move / FS - math.pi / 2)
        used = eng.move_crossovers(NEW, policy=policy,
                                   tone_hint=hint if policy == "map" else None)
        assert used == policy
        y = run(eng, post)
        results[policy] = maxdiff(y, ideal)
        if policy == "xfade":
            frames = [e for e in eng.switch_log
                      if e["policy"] == "xfade"][0]["frames"]
            xfade_during = maxdiff(y[:frames], ideal[:frames])
            xfade_after = maxdiff(y[frames:], ideal[frames:])

    # state conversion is (near-)exact for a sustained tone
    assert results["map"] < 1e-6, f"map error {results['map']}"
    # crossfade error is comb-shaped, bounded, and confined to the fade
    assert results["map"] < results["xfade"]
    assert xfade_during <= comb_bound + 1e-6, (xfade_during, comb_bound)
    assert xfade_after < 1e-6, f"xfade not clean after fade: {xfade_after}"
    # the raw swap clicks
    assert results["swap"] > 1e-2, f"swap should click: {results['swap']}"


def test_map_aligns_every_band_memory():
    eng = _primed_engine()
    eng.probe_bands = True
    n_move = int(FS * 0.3)
    hint = (TONE_F, TONE_A, math.tau * TONE_F * n_move / FS - math.pi / 2)
    eng.move_crossovers(NEW, policy="map", tone_hint=hint)
    run(eng, tone(FS, TONE_F, TONE_A, 0.15))
    xo = MultibandCrossover(NEW, FS)
    w = math.tau * TONE_F / FS
    for k, tr in enumerate(eng.band_trace):
        hk = xo.H_band(k, w)
        # DFT of a steady sine y[n]=Im{Hk X e^{jwn}} over a window at n_move
        expected = hk * TONE_A * complex(math.cos(w * n_move - math.pi / 2),
                                         math.sin(w * n_move - math.pi / 2))
        got = dft(tr, FS, TONE_F)
        assert abs(got - expected) < 1e-6, \
            f"band {k} memory misaligned: {abs(got - expected)}"


def _tail_engine(silence_samples):
    """Engine whose stage-1 filters ring from a 400 Hz burst, then silence."""
    eng = MultibandEngine(FS, OLD, warmup_frames=8192)
    run(eng, tone(FS, 400.0, 0.8, 0.03))
    run(eng, [0.0] * silence_samples)
    return eng


def test_inphase_vs_antiphase_tails_equal_input():
    # 400 Hz ringing: 30 samples = half a period at fs=24k flips the tail
    eng_a = _tail_engine(48)            # tail lands in phase with new input
    eng_b = _tail_engine(48 + 30)       # tail lands anti-phase
    sa = eng_a._core.crossover.lp[0].bq[0].s1
    sb = eng_b._core.crossover.lp[0].bq[0].s1
    assert abs(sa - sb) > 1e-4, f"tails not actually different: {sa} vs {sb}"

    post = tone(FS, TONE_F, TONE_A, 0.12)   # identical input from the switch
    ideal = _ideal_sin(0, len(post), NEW)
    hint = (TONE_F, TONE_A, -math.pi / 2)   # sine starts at phase 0
    errs = {}
    for name, eng in (("in", eng_a), ("anti", eng_b)):
        eng.move_crossovers(NEW, policy="map", tone_hint=hint)
        errs[name] = maxdiff(run(eng, post), ideal)
    assert errs["in"] < 1e-6 and errs["anti"] < 1e-6, errs

    # raw swap, by contrast, is at the mercy of the tails
    eng_c, eng_d = _tail_engine(48), _tail_engine(78)
    for eng in (eng_c, eng_d):
        eng.move_crossovers(NEW, policy="swap")
    e_c = maxdiff(run(eng_c, post), ideal)
    e_d = maxdiff(run(eng_d, post), ideal)
    assert e_c > 1e-3 and e_d > 1e-3, (e_c, e_d)
    assert abs(e_c - e_d) > 1e-4, "tail phase must matter for the naive swap"


def test_auto_policy_selection():
    # enough warmup frames -> double-run crossfade
    eng = MultibandEngine(FS, OLD, warmup_frames=8192)
    run(eng, tone(FS, TONE_F, TONE_A, 0.2))
    assert eng.move_crossovers(NEW, policy="auto") == "xfade"
    # only old state, tone hint available -> map memory
    eng = MultibandEngine(FS, OLD, warmup_frames=64)
    run(eng, tone(FS, TONE_F, TONE_A, 0.05))
    assert eng.move_crossovers(NEW, policy="auto",
                             tone_hint=(TONE_F, TONE_A, 0.0)) == "map"
    # only old state, no hint -> deferred band switch
    eng = MultibandEngine(FS, OLD, warmup_frames=64)
    run(eng, tone(FS, TONE_F, TONE_A, 0.05))
    assert eng.move_crossovers(NEW, policy="auto") == "deferred"


def test_deferred_switch_takes_effect_after_settle():
    eng = MultibandEngine(FS, OLD, warmup_frames=64)
    run(eng, tone(FS, TONE_F, TONE_A, 0.05))
    assert eng.move_crossovers(NEW, policy="auto") == "deferred"
    settle = eng.settle_frames(400.0)
    # during the deferred window the old split is still live: output must
    # equal a reference engine that never moved
    ref = MultibandEngine(FS, OLD, warmup_frames=64)
    run(ref, tone(FS, TONE_F, TONE_A, 0.05))
    probe = tone(FS, 900.0, 0.5, 0.002)  # 48 samples < settle
    assert settle > 48
    assert run(eng, probe) == run(ref, probe)
    # after the settle window the new split is in effect
    eng.probe_bands = True
    run(eng, tone(FS, 900.0, 0.5, 0.2))
    tr0 = eng.band_trace[0][-1200:]
    tr1 = eng.band_trace[1][-1200:]
    assert abs(db(abs(dft(tr0, FS, 900.0)) / 0.5) - (-6.02)) < 0.6
    assert abs(db(abs(dft(tr1, FS, 900.0)) / 0.5) - (-6.02)) < 0.6
    policies = [e["policy"] for e in eng.switch_log]
    assert "deferred" in policies and "deferred-applied" in policies


def test_move_validation_and_unknown_policy():
    eng = MultibandEngine(FS, OLD)
    for bad in ([10.0, 2000.0], [900.0, 1000.0], [2000.0, 900.0],
                [500.0, 1000.0, 2000.0, 4000.0]):
        try:
            eng.move_crossovers(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted bad crossovers {bad}")
    for kwargs in ({"policy": "teleport"}, {"policy": "map"}):
        try:
            eng.move_crossovers(NEW, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {kwargs}")
