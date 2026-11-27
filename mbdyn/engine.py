"""Multiband dynamics engine.

Split -> per-band detect/gain -> recombine, block-based, state persists
across blocks. Declared unity response (all band gains = 1, processing
on): the cascade of the stage allpasses, i.e. exactly flat magnitude with
the allpass phase of the LR4 stages (see crossover.py).

State is owned by three distinct layers (see bands.py):
  filter memory    -> MultibandCrossover biquad states
  envelope memory  -> one BandDynamics.env per band (keyed by detection
                      source = the band's frequency interval)
  band gain        -> pure function of (env, config); no memory

A crossover move is a *filter-state recombination boundary*: the new
topology's filters are established (retuned in place or warmed up), and
each band's envelope is then inherited or rebuilt strictly by whether its
detection source is unchanged.  An unreleased gain from an old band can
therefore never follow a band index onto a different source.

Switch semantics (each independently verifiable):
  bypass=True          output is bit-exact the input (cores keep running)
  processing_on=False  bands sum at unity gain through the crossover
  band.enabled=False   that band passes at unity gain; its content is NOT
                       removed from the sum
  solo                 only soloed bands reach the output
  mute                 band excluded from the sum

Crossover moves while running use one of these policies:
  xfade     double-run old/new topologies and crossfade; needs >= settle
            frames of warmup history.  The new topology is warmed on that
            live history, so every band's envelope is rebuilt from the new
            source (never inherited from the old topology).
  map       retune/map filter memory to a known sustained tone and migrate
            envelopes by source; instantaneous.  Needs tone_hint.
  deferred  keep the old split while the new topology warms up on the live
            stream, then switch (the band switch is delayed).
  swap      raw coefficient swap, states kept by index (baseline for
            comparison: this is the stale-envelope bug made visible).
'auto' picks xfade when enough warmup frames exist, else map when a tone
hint is given, else deferred.
"""

import cmath
import math
from collections import deque

from .bands import BandProcessing, band_ranges, configs_for_topology
from .crossover import AllpassCascade
from .dynamics import BandConfig

# Compatibility adapter: the pre-refactor entry point constructed as
# ``_Core(fs, freqs, band_cfgs, lookahead)``.  It now IS the band-processing
# unit (same constructor, plus a ``step`` alias), so old callers that only
# touched ``.crossover`` / ``.dyns`` keep working.
_Core = BandProcessing


class MultibandEngine:
    MIN_CROSSOVER_HZ = 40.0
    MAX_CROSSOVER_RATIO = 0.42   # of sample rate
    MIN_SPACING_OCT = 0.5        # band-shape sanity; reconstruction is exact regardless
    MIN_BANDS = 2
    MAX_BANDS = 4
    MIN_FS = 8000.0
    MAX_FS = 192000.0

    def __init__(self, sample_rate=44100.0, crossovers=(200.0, 1000.0, 5000.0),
                 warmup_frames=8192, lookahead_ms=0.0):
        if not (self.MIN_FS <= float(sample_rate) <= self.MAX_FS):
            raise ValueError(
                f"sample_rate {sample_rate} out of range "
                f"[{self.MIN_FS}, {self.MAX_FS}] Hz")
        self.fs = float(sample_rate)
        self._validate(crossovers)
        freqs = [float(f) for f in crossovers]
        configs = [BandConfig() for _ in range(len(freqs) + 1)]
        self.lookahead = max(0, int(round(lookahead_ms * self.fs * 1e-3)))
        self.bypass = False
        self.processing_on = True
        self.record_input = False
        self.recorded = []
        self.warmup = deque(maxlen=int(warmup_frames))
        self.probe_bands = False
        self.probe_gains = False
        self.band_trace = [[] for _ in configs]
        self.gain_trace = [[] for _ in configs]
        self.switch_log = []
        self.n = 0
        self._proc = BandProcessing(self.fs, freqs, configs, self.lookahead)
        self._xfade = None      # [old_proc, new_proc, pos, frames]
        self._shadow = None     # [new_proc, countdown]
        self._pending_freqs = None

    # ---------------------------------------------------------- configuration
    def _validate(self, freqs):
        freqs = [float(f) for f in freqs]
        nb = len(freqs) + 1
        if not (self.MIN_BANDS <= nb <= self.MAX_BANDS):
            raise ValueError(
                f"band count {nb} unsupported: need {self.MIN_BANDS}..{self.MAX_BANDS} "
                f"bands ({self.MIN_BANDS - 1}..{self.MAX_BANDS - 1} crossovers)")
        fmax = self.MAX_CROSSOVER_RATIO * self.fs
        for f in freqs:
            if not (self.MIN_CROSSOVER_HZ <= f <= fmax):
                raise ValueError(
                    f"crossover {f} Hz out of range "
                    f"[{self.MIN_CROSSOVER_HZ}, {fmax:.1f}] Hz at fs={self.fs:.0f}")
        if any(b <= a for a, b in zip(freqs, freqs[1:])):
            raise ValueError(f"crossovers must be strictly increasing: {freqs}")
        ratio = 2.0 ** self.MIN_SPACING_OCT
        for a, b in zip(freqs, freqs[1:]):
            if b / a < ratio:
                raise ValueError(
                    f"crossover spacing {a}/{b} below {self.MIN_SPACING_OCT} octave")

    @property
    def freqs(self):
        return list(self._proc.freqs)

    @property
    def pending_freqs(self):
        """Crossover set of an in-progress move, or None."""
        return None if self._pending_freqs is None else list(self._pending_freqs)

    @property
    def bands(self):
        """Canonical per-band configs (owned by the active topology)."""
        return self._proc.configs

    @property
    def num_bands(self):
        return self._proc.num_bands

    @property
    def any_solo(self):
        return any(c.solo for c in self.bands)

    @property
    def latency_samples(self):
        return self.lookahead

    @property
    def _core(self):
        """Compat: pre-refactor handle exposing ``.crossover`` / ``.dyns``."""
        return self._proc

    @property
    def crossover(self):
        """The active filter bank (read-only view for analysis/plotting)."""
        return self._proc.crossover

    def set_band(self, k, **params):
        self._proc.update_config(k, **params)

    def set_solo(self, k, on):
        self.bands[k].solo = bool(on)

    def set_mute(self, k, on):
        self.bands[k].mute = bool(on)

    def set_bypass(self, on):
        self.bypass = bool(on)

    def set_processing(self, on):
        self.processing_on = bool(on)

    def reset_states(self, clear_log=True):
        self._finish_transition()  # never silently drop a pending band switch
        self._proc = BandProcessing(self.fs, self._proc.freqs,
                                    self._proc.configs, self.lookahead)
        self._xfade = None
        self._shadow = None
        self._pending_freqs = None
        self.warmup.clear()
        self.band_trace = [[] for _ in self.bands]
        self.gain_trace = [[] for _ in self.bands]
        if clear_log:
            self.switch_log.clear()

    def settle_frames(self, fmin=None):
        f = fmin if fmin else min(self._proc.freqs)
        return max(1, int(math.ceil(1.6 * self.fs / f)))

    # ------------------------------------------------------------- processing
    def process(self, block):
        out = []
        append = self.warmup.append
        rec = self.record_input
        recorded = self.recorded
        for x in block:
            append(x)
            if rec:
                recorded.append(x)
            out.append(self._step(x))
        return out

    def process_block(self, block):
        """Unified block entry point (alias of :meth:`process`)."""
        return self.process(block)

    def _step(self, x):
        self.n += 1
        if self._shadow is not None:
            sh = self._shadow
            switch_now = sh[1] <= 1
            # probe only on the sample that actually reaches the output, so
            # the new core is stepped exactly once per sample (no double-step
            # on the switch sample) and traces stay aligned with the output
            y_new = sh[0].process_sample(x, self, probe=switch_now)
            sh[1] -= 1
            if switch_now:
                self._install(sh[0])
                self._shadow = None
                self.switch_log.append({"n": self.n, "policy": "deferred-applied",
                                        "freqs": list(self._proc.freqs)})
                y = y_new
            else:
                y = self._proc.process_sample(x, self, probe=True)
        elif self._xfade is not None:
            xf = self._xfade
            o = xf[0].process_sample(x, self, probe=False)
            n = xf[1].process_sample(x, self, probe=True)
            w = 0.5 * (1.0 - math.cos(math.pi * (xf[2] + 1) / xf[3]))
            y = (1.0 - w) * o + w * n
            xf[2] += 1
            if xf[2] >= xf[3]:
                self._install(xf[1])
                self._xfade = None
                self.switch_log.append({"n": self.n, "policy": "xfade-done",
                                        "freqs": list(self._proc.freqs)})
        else:
            y = self._proc.process_sample(x, self, probe=True)
        if self.bypass:
            return x
        return y

    def _install(self, proc):
        self._proc = proc
        self._pending_freqs = None
        if len(self.band_trace) != proc.num_bands:
            self.band_trace = [[] for _ in range(proc.num_bands)]
            self.gain_trace = [[] for _ in range(proc.num_bands)]

    # -------------------------------------------------------- crossover moves
    def _finish_transition(self):
        if self._xfade is not None:
            self._install(self._xfade[1])
            self._xfade = None
        elif self._shadow is not None:
            self._install(self._shadow[0])
            self._shadow = None
        elif self._pending_freqs is not None:
            self._pending_freqs = None

    def _new_processing(self, new_freqs):
        """Build a fresh topology for a double-run (xfade/deferred) move.

        Envelopes are intentionally NOT inherited here: a double-run policy
        warms the new topology on the live stream, so every band's detector
        is rebuilt from the *new* source.  A fresh core can never hold an
        old source's envelope, so this is rebuild-from-new, never leakage.
        """
        cfgs = configs_for_topology(self._proc.configs, self._proc.freqs,
                                    new_freqs, self.fs)
        return BandProcessing(self.fs, new_freqs, cfgs, self.lookahead)

    def move_crossovers(self, new_freqs, policy="auto", tone_hint=None):
        """Move all crossover points. Returns the policy actually used.

        tone_hint = (freq_hz, amplitude, phase_rad) of a sustained tone
        defined as ampl*cos(2*pi*f*k/fs + phase) at switch-relative
        sample k; required by 'map'.
        """
        self._validate(new_freqs)
        new_freqs = [float(f) for f in new_freqs]
        self._finish_transition()
        old_freqs = list(self._proc.freqs)
        fmin = min(min(new_freqs), min(old_freqs))
        settle = self.settle_frames(fmin)
        if policy == "auto":
            if len(self.warmup) >= settle:
                policy = "xfade"
            elif tone_hint is not None:
                policy = "map"
            else:
                policy = "deferred"
        if policy == "map":
            if tone_hint is None:
                raise ValueError("policy 'map' requires tone_hint=(freq, ampl, phase)")
            f, ampl, phase = tone_hint
            w = math.tau * f / self.fs
            X = ampl * cmath.exp(1j * phase)
            old_envs = self._proc.envelope_snapshot()
            if len(new_freqs) == len(old_freqs):
                # same topology count: preserve filter/delay memory, then map
                self._proc.retune_in_place(new_freqs)
                self._proc.map_filter_state(w, X)
                self._proc.recombine_envelopes(old_freqs, old_envs)
            else:
                # count change: filters/delays necessarily rebuilt, but map
                # them to the tone and migrate envelopes by source
                new_proc = BandProcessing(
                    self.fs, new_freqs,
                    configs_for_topology(self._proc.configs, old_freqs,
                                         new_freqs, self.fs),
                    self.lookahead)
                new_proc.map_filter_state(w, X)
                new_proc.recombine_envelopes(old_freqs, old_envs)
                self._install(new_proc)
            self.switch_log.append({"n": self.n, "policy": "map",
                                    "freqs": list(new_freqs)})
        elif policy == "xfade":
            new_proc = self._new_processing(new_freqs)
            for s in self.warmup:
                new_proc.process_sample(s, self, probe=False)
            frames = max(1, min(settle, len(self.warmup)))
            self._xfade = [self._proc, new_proc, 0, frames]
            self._pending_freqs = new_freqs
            self.switch_log.append({"n": self.n, "policy": "xfade",
                                    "frames": frames, "freqs": list(new_freqs)})
        elif policy == "deferred":
            new_proc = self._new_processing(new_freqs)
            self._shadow = [new_proc, settle]
            self._pending_freqs = new_freqs
            self.switch_log.append({"n": self.n, "policy": "deferred",
                                    "frames": settle, "freqs": list(new_freqs)})
        elif policy == "swap":
            # raw baseline: filter coefficients retuned, memories untouched
            self._proc.retune_in_place(new_freqs)
            self.switch_log.append({"n": self.n, "policy": "swap",
                                    "freqs": list(new_freqs)})
        else:
            raise ValueError(f"unknown move policy {policy!r}")
        return policy

    # ---------------------------------------------------------------- analysis
    def reference_allpass(self):
        """The declared unity-gain response as a standalone filter."""
        return AllpassCascade(self._proc.freqs, self.fs)

    def band_ranges(self):
        return band_ranges(self._proc.freqs, self.fs)

    def envelope_state(self):
        """Per-band running state: source interval, envelope, derived gain."""
        p = self._proc
        out = []
        for k, (lo, hi) in enumerate(band_ranges(p.freqs, self.fs)):
            cfg = p.configs[k]
            active = cfg.enabled and self.processing_on
            out.append({
                "index": k,
                "range": (lo, hi),
                "envelope": p.dyns[k].envelope(),
                "gain_db": p.dyns[k].gain_db() if active else 0.0,
                "enabled": cfg.enabled,
            })
        return out
