"""State split: filter memory vs envelope memory across topology changes.

The envelope detector may follow a band only when its *detection source*
(the frequency interval) is unchanged.  These tests replay the two cases
the split has to get right -- a pure band-index renumbering, and a real
sidechain reconnection -- and read the pre-merge band signals plus the
post-merge loudness so a cancellation between bands cannot hide a band
carrying the wrong memory.
"""

import math

from util import dft, db, maxdiff, run, tone

from mbdyn import MultibandEngine
from mbdyn.bands import BandProcessing, band_ranges, match_band_intervals

FS = 24000.0


def _prime(eng, freqs, x, band_params=None):
    if band_params:
        for k, p in band_params.items():
            eng.set_band(k, **p)
    run(eng, x)


def _punchy():
    return dict(threshold_db=-24.0, ratio=8.0, knee_db=0.0,
                attack_ms=2.0, release_ms=400.0)


def test_source_change_rebuilds_envelope_not_index():
    """A stale gain must not follow the band index onto a different source."""
    eng = MultibandEngine(FS, [900.0])
    for k in range(eng.num_bands):
        eng.set_band(k, **_punchy())
    # loud 500 Hz lives in band 0 (0-900) only
    run(eng, tone(FS, 500.0, 0.8, 0.5))
    old_env = [d["envelope"] for d in eng.envelope_state()]
    assert old_env[0] > 0.5, old_env  # detector really saw the 500 Hz
    assert eng.envelope_state()[0]["gain_db"] < -6.0

    # move crossover down: band 0 is now 0-400, which never contained 500 Hz
    eng.move_crossovers([400.0], policy="map", tone_hint=(500.0, 0.8, 0.0))
    st = eng.envelope_state()
    # 500 Hz moved to band 1; band 0's source genuinely changed -> rebuilt
    assert st[0]["envelope"] == 0.0, f"band0 kept stale history: {st[0]}"
    assert st[0]["gain_db"] == 0.0, f"band0 kept stale gain: {st[0]}"
    assert st[1]["envelope"] == 0.0, st[1]


def test_index_shift_inherits_envelope_for_unchanged_source():
    """Renumbering a band whose interval is unchanged inherits its history."""
    eng = MultibandEngine(FS, [900.0])
    for k in range(eng.num_bands):
        eng.set_band(k, **_punchy())
    # loud 2000 Hz lives in the top band (900-12000)
    run(eng, tone(FS, 2000.0, 0.8, 0.5))
    old = eng.envelope_state()
    assert old[1]["envelope"] > 0.5, old

    # adding a 400 Hz split renumbers the top band 1 -> 2, same source
    eng.move_crossovers([400.0, 900.0], policy="map",
                        tone_hint=(2000.0, 0.8, 0.0))
    new = eng.envelope_state()
    assert new[2]["range"] == old[1]["range"]
    assert abs(new[2]["envelope"] - old[1]["envelope"]) < 1e-12, \
        f"identical source did not inherit: {new[2]} vs {old[1]}"
    # the genuinely new bands did not inherit the loud top-band history
    assert new[0]["envelope"] == 0.0 and new[1]["envelope"] == 0.0


def test_recombine_maps_only_identical_intervals():
    old_f = [900.0]
    new_f = [400.0, 900.0]
    m = match_band_intervals(old_f, new_f, FS)
    assert m == {2: 1}, m   # only (900, fs/2) survives, index 1 -> 2
    # a move that keeps no interval must map nothing
    assert match_band_intervals([900.0], [400.0], FS) == {}
    assert band_ranges([900.0], FS) == [(0.0, 900.0), (900.0, FS / 2)]


def test_same_center_different_source_is_rebuilt():
    """A band with the same centre frequency but a different interval is a
    different detection source and must be rebuilt, not inherited."""
    # old middle band (400,1600) and new middle band (600,1400) share centre
    # 1000 Hz but measure different material -> must not inherit
    old_ranges = band_ranges([400.0, 1600.0], FS)
    new_ranges = band_ranges([600.0, 1400.0], FS)
    assert old_ranges[1] == (400.0, 1600.0) and new_ranges[1] == (600.0, 1400.0)
    assert 0.5 * sum(old_ranges[1]) == 0.5 * sum(new_ranges[1]) == 1000.0
    assert match_band_intervals([400.0, 1600.0], [600.0, 1400.0], FS) == {}

    eng = MultibandEngine(FS, [400.0, 1600.0])
    for k in range(eng.num_bands):
        eng.set_band(k, **_punchy())
    run(eng, tone(FS, 1000.0, 0.8, 0.5))   # loud tone at the shared centre
    before = eng.envelope_state()
    assert before[1]["envelope"] > 0.5, before
    eng.move_crossovers([600.0, 1400.0], policy="map",
                        tone_hint=(1000.0, 0.8, 0.0))
    after = eng.envelope_state()
    assert after[1]["range"] == (600.0, 1400.0)
    assert after[1]["envelope"] == 0.0, \
        f"matched by centre, not source: {after[1]}"


def test_recombine_boundary_unit():
    """BandProcessing.recombine_envelopes inherits/rebuilds by source."""
    cfgA, cfgB = object(), object()
    old = BandProcessing(FS, [900.0], [cfgA, cfgB], 0)
    old.dyns[0].set_envelope(0.7)
    old.dyns[1].set_envelope(0.2)
    snap = old.envelope_snapshot()

    new = BandProcessing(FS, [400.0, 900.0], [cfgA, cfgA, cfgB], 0)
    mapping = new.recombine_envelopes([900.0], snap)
    assert mapping == {2: 1}
    assert new.dyns[2].envelope() == 0.2       # inherited
    assert new.dyns[0].envelope() == 0.0       # rebuilt
    assert new.dyns[1].envelope() == 0.0       # rebuilt


def _tail_dft(trace, f, frac=0.5):
    seg = trace[int(len(trace) * (1.0 - frac)):]
    return abs(dft(seg, FS, f))


def test_pre_merge_transients_and_post_merge_loudness():
    """Read both bands before the sum, then the sum, for both move kinds.

    The probe tone is well below threshold, so a band carrying a stale
    gain is *visibly* attenuated even though nothing else is compressing.
    Band signals (pre-sum) and the recombined output (post-sum) are each
    measured by steady-state DFT in the tail, where a cancellation between
    two bands cannot hide a single band's wrong memory.
    """
    loud = tone(FS, 500.0, 0.8, 0.5)          # above threshold, lives in old band 0
    p_f, p_a = 100.0, 0.02                     # below threshold, lives in new band 0
    post = tone(FS, p_f, p_a, 0.20)
    hint = (p_f, p_a, -math.pi / 2)            # sine starts at phase 0 -> cos phasor

    def prepared(policy):
        e = MultibandEngine(FS, [900.0])
        for k in range(e.num_bands):
            e.set_band(k, **_punchy())
        run(e, loud)
        e.move_crossovers([400.0], policy=policy,
                          tone_hint=hint if policy == "map" else None)
        e.probe_bands = True
        return e, run(e, post)

    em, ym = prepared("map")     # source-aware: band 0 rebuilt
    es, ys = prepared("swap")    # baseline: band 0 keeps the loud envelope

    # --- pre-merge: band 0 after a genuine sidechain reconnection ---
    am = _tail_dft(em.band_trace[0], p_f)
    asw = _tail_dft(es.band_trace[0], p_f)
    assert abs(db(am / p_a)) < 1.0, f"map band0 probe not at unity: {db(am / p_a):.2f} dB"
    assert db(asw / p_a) < -6.0, f"swap band0 did not inherit stale gain: {db(asw / p_a):.2f} dB"

    # --- post-merge: recombined loudness of the quiet probe ---
    sm = _tail_dft(ym, p_f)
    ss = _tail_dft(ys, p_f)
    assert abs(db(sm / p_a)) < 1.0, f"map sum probe not at unity: {db(sm / p_a):.2f} dB"
    assert db(ss / p_a) < -6.0, f"swap sum not dimmed by stale gain: {db(ss / p_a):.2f} dB"

    # --- pure renumbering: the top band keeps its memory across 1 -> 2 ---
    eng2 = MultibandEngine(FS, [900.0])
    for k in range(eng2.num_bands):
        eng2.set_band(k, **_punchy())
    hi = tone(FS, 2000.0, 0.8, 0.5)
    run(eng2, hi)
    eng2.move_crossovers([400.0, 900.0], policy="map",
                         tone_hint=(2000.0, 0.8, 0.0))
    eng2.probe_gains = True
    run(eng2, tone(FS, 2000.0, 0.8, 0.05))  # continues the loud source
    g_top = eng2.gain_trace[2]
    # inherited history keeps the top band compressed from the first sample
    assert g_top[0] < -3.0, f"top band lost its inherited gain: {g_top[0]}"


def test_swap_keeps_stale_memory_map_rebuilds_it():
    """At one instant, raw swap is the bug; source-rebuilt map is right."""
    loud = tone(FS, 500.0, 0.8, 0.5)
    post = tone(FS, 100.0, 0.02, 0.05)
    hint = (100.0, 0.02, -math.pi / 2)

    def prepared(policy):
        e = MultibandEngine(FS, [900.0])
        for k in range(e.num_bands):
            e.set_band(k, **_punchy())
        run(e, loud)
        e.move_crossovers([400.0], policy=policy,
                          tone_hint=hint if policy == "map" else None)
        e.probe_gains = True
        run(e, post)
        return e

    sw = prepared("swap")
    mp = prepared("map")
    # the swap baseline applies the old 500 Hz envelope on the first sample
    assert sw.gain_trace[0][0] < -10.0, sw.gain_trace[0][0]
    # the source-rebuilt implementation starts from a fresh envelope
    assert mp.gain_trace[0][0] == 0.0, mp.gain_trace[0][0]


def test_block_length_difference_across_move():
    """Moving at a fixed sample index must not depend on block length."""
    x = [0.6 * math.sin(math.tau * 300.0 * i / FS)
         + 0.3 * math.sin(math.tau * 3000.0 * i / FS) for i in range(int(FS * 0.4))]
    split = int(FS * 0.2)

    def go(block):
        e = MultibandEngine(FS, [400.0, 2000.0])
        for k in range(e.num_bands):
            e.set_band(k, **_punchy())
        out = []
        i = 0
        while i < split:
            j = min(i + block, split)   # never overshoot the move instant
            out.extend(e.process(x[i:j]))
            i = j
        e.move_crossovers([900.0, 2000.0], policy="map",
                          tone_hint=(300.0, 0.6, 0.0))
        while i < len(x):
            j = min(i + block, len(x))
            out.extend(e.process(x[i:j]))
            i = j
        return out

    base = go(4096)
    for block in (1, 97, 511):
        assert go(block) == base, f"block length {block} changed the move"


def test_bypass_toggle_is_state_transparent():
    """Bypass returns input bit-exact but keeps the state path running."""
    x = [0.5 * math.sin(math.tau * 120.0 * i / FS) for i in range(int(FS * 0.5))]
    ref = MultibandEngine(FS, [300.0, 3000.0])
    ref.set_band(0, threshold_db=-30.0, ratio=4.0, attack_ms=3.0)
    ref.set_band(2, threshold_db=-10.0, ratio=4.0, attack_ms=3.0)

    tog = MultibandEngine(FS, [300.0, 3000.0])
    tog.set_band(0, threshold_db=-30.0, ratio=4.0, attack_ms=3.0)
    tog.set_band(2, threshold_db=-10.0, ratio=4.0, attack_ms=3.0)

    half = len(x) // 2
    tog.set_bypass(True)
    yt = run(tog, x[:half])
    assert yt == x[:half], "bypass must stay bit-exact while toggled"
    tog.set_bypass(False)
    yt += run(tog, x[half:])
    yr = run(ref, x)
    assert maxdiff(yt[half:], yr[half:]) < 1e-12, \
        "bypass toggle changed the processing state trajectory"


def test_legacy_core_adapter_still_runs():
    """The pre-refactor _Core entry point remains a working adapter."""
    from mbdyn.engine import _Core
    eng = MultibandEngine(FS, [300.0, 3000.0])
    core = _Core(FS, [300.0, 3000.0], eng.bands, 0)
    assert hasattr(core, "crossover") and hasattr(core, "dyns")
    y = [core.step(0.5, eng, False) for _ in range(64)]
    assert all(isinstance(v, float) for v in y)
    # and the engine's live core exposes the same surface
    assert eng._core.crossover is not None
    assert len(eng._core.dyns) == eng.num_bands


def test_reconstruction_unchanged_after_state_split():
    """Declared flat response and delay relation survive the refactor."""
    from mbdyn.crossover import AllpassCascade
    xo = [400.0, 1600.0]
    eng = MultibandEngine(FS, xo, lookahead_ms=2.0)
    L = eng.latency_samples
    x = [1.0] + [0.0] * 4095
    y = run(eng, x)
    ref = AllpassCascade(xo, FS).process(x)
    assert max(abs(v) for v in y[:L]) == 0.0
    assert maxdiff(y[L:], ref[:len(x) - L]) < 1e-9
