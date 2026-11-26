"""Multiband dynamics engine.

Split -> per-band detect/gain -> recombine, block-based, state persists
across blocks. Declared unity response (all band gains = 1, processing
on): the cascade of the stage allpasses, i.e. exactly flat magnitude with
the allpass phase of the LR4 stages (see crossover.py).

Switch semantics (each independently verifiable):
  bypass=True          output is bit-exact the input (cores keep running)
  processing_on=False  bands sum at unity gain through the crossover
  band.enabled=False   that band passes at unity gain; its content is NOT
                       removed from the sum
  solo                 only soloed bands reach the output
  mute                 band excluded from the sum

Crossover moves while running use one of three policies:
  xfade     double-run old/new cores and crossfade; needs >= settle frames
            of warmup input history
  map       align every filter's memory to a known sustained tone
            (needs tone_hint); instantaneous
  deferred  keep the old band split while the new core warms up on the
            live stream, then switch (the band switch is delayed)
  swap      raw coefficient swap, states kept (baseline for comparison)
'auto' picks xfade when enough warmup frames exist, else map when a tone
hint is given, else deferred.
"""

import cmath
import math
from collections import deque

from .crossover import AllpassCascade, MultibandCrossover
from .dynamics import BandConfig, BandDynamics


class _Delay:
    __slots__ = ("n", "buf", "i")

    def __init__(self, n):
        self.n = n
        self.buf = [0.0] * n if n > 0 else None
        self.i = 0

    def step(self, x):
        if self.n == 0:
            return x
        y = self.buf[self.i]
        self.buf[self.i] = x
        self.i = (self.i + 1) % self.n
        return y


class _Core:
    """One crossover + per-band dynamics + aligned band delays."""

    def __init__(self, fs, freqs, band_cfgs, lookahead):
        self.crossover = MultibandCrossover(freqs, fs)
        self.dyns = [BandDynamics(fs, c) for c in band_cfgs]
        self.delays = [_Delay(lookahead) for _ in band_cfgs]

    def step(self, x, eng, probe):
        bands = self.crossover.step(x)
        out = 0.0
        solo_any = eng.any_solo
        for k, cfg in enumerate(eng.bands):
            xd = self.delays[k].step(bands[k])
            if eng.processing_on and cfg.enabled:
                g = self.dyns[k].step_gain(bands[k])
            else:
                g = 1.0
                if eng.processing_on:
                    self.dyns[k].track_only(bands[k])
            post = xd * g
            if probe:
                if eng.probe_bands:
                    eng.band_trace[k].append(post)
                if eng.probe_gains:
                    eng.gain_trace[k].append(self.dyns[k].gain_db()
                                             if cfg.enabled and eng.processing_on
                                             else 0.0)
            if (cfg.solo if solo_any else not cfg.mute):
                out += post
        return out


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
        self.freqs = [float(f) for f in crossovers]
        self.bands = [BandConfig() for _ in range(len(self.freqs) + 1)]
        self.lookahead = max(0, int(round(lookahead_ms * self.fs * 1e-3)))
        self.bypass = False
        self.processing_on = True
        self.record_input = False
        self.recorded = []
        self.warmup = deque(maxlen=int(warmup_frames))
        self.probe_bands = False
        self.probe_gains = False
        self.band_trace = [[] for _ in self.bands]
        self.gain_trace = [[] for _ in self.bands]
        self.switch_log = []
        self.n = 0
        self._core = _Core(self.fs, self.freqs, self.bands, self.lookahead)
        self._xfade = None      # [old_core, new_core, pos, frames]
        self._shadow = None     # [new_core, countdown]
        self._pending_freqs = None
        self._preroll = False

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
    def num_bands(self):
        return len(self.bands)

    @property
    def any_solo(self):
        return any(c.solo for c in self.bands)

    @property
    def latency_samples(self):
        return self.lookahead

    def set_band(self, k, **params):
        cfg = self.bands[k]
        for key, val in params.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown band parameter {key!r}")
            setattr(cfg, key, val)

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
        self._core = _Core(self.fs, self.freqs, self.bands, self.lookahead)
        self._xfade = None
        self._shadow = None
        self._pending_freqs = None
        self.warmup.clear()
        self.band_trace = [[] for _ in self.bands]
        self.gain_trace = [[] for _ in self.bands]
        if clear_log:
            self.switch_log.clear()

    def settle_frames(self, fmin=None):
        f = fmin if fmin else min(self.freqs)
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

    def _step(self, x):
        self.n += 1
        if self._shadow is not None:
            sh = self._shadow
            sh[0].step(x, self, False)
            sh[1] -= 1
            if sh[1] <= 0:
                self._core = sh[0]
                self._shadow = None
                self.freqs = self._pending_freqs
                self._pending_freqs = None
                self.switch_log.append({"n": self.n, "policy": "deferred-applied",
                                        "freqs": list(self.freqs)})
        if self._xfade is not None:
            xf = self._xfade
            o = xf[0].step(x, self, False)
            n = xf[1].step(x, self, True)
            w = 0.5 * (1.0 - math.cos(math.pi * (xf[2] + 1) / xf[3]))
            y = (1.0 - w) * o + w * n
            xf[2] += 1
            if xf[2] >= xf[3]:
                self._core = xf[1]
                self._xfade = None
                self.freqs = self._pending_freqs
                self._pending_freqs = None
                self.switch_log.append({"n": self.n, "policy": "xfade-done",
                                        "freqs": list(self.freqs)})
        else:
            y = self._core.step(x, self, not self._preroll)
        if self.bypass:
            return x
        return y

    # -------------------------------------------------------- crossover moves
    def _finish_transition(self):
        if self._xfade is not None:
            self._core = self._xfade[1]
            self._xfade = None
        if self._shadow is not None:
            self._core = self._shadow[0]
            self._shadow = None
        if self._pending_freqs is not None:
            self.freqs = self._pending_freqs
            self._pending_freqs = None

    def move_crossovers(self, new_freqs, policy="auto", tone_hint=None):
        """Move all crossover points. Returns the policy actually used.

        tone_hint = (freq_hz, amplitude, phase_rad) of a sustained tone
        defined as ampl*cos(2*pi*f*k/fs + phase) at switch-relative
        sample k; required by 'map'.
        """
        self._validate(new_freqs)
        new_freqs = [float(f) for f in new_freqs]
        self._finish_transition()
        fmin = min(min(new_freqs), min(self.freqs))
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
            self._core.crossover.set_frequencies(new_freqs)
            self._core.crossover.map_state(w, X)
            self.freqs = new_freqs
            self.switch_log.append({"n": self.n, "policy": "map",
                                    "freqs": list(new_freqs)})
        elif policy == "xfade":
            new_core = _Core(self.fs, new_freqs, self.bands, self.lookahead)
            self._preroll = True
            for s in self.warmup:
                new_core.step(s, self, False)
            self._preroll = False
            frames = max(1, min(settle, len(self.warmup)))
            self._xfade = [self._core, new_core, 0, frames]
            self._pending_freqs = new_freqs
            self.switch_log.append({"n": self.n, "policy": "xfade",
                                    "frames": frames, "freqs": list(new_freqs)})
        elif policy == "deferred":
            new_core = _Core(self.fs, new_freqs, self.bands, self.lookahead)
            self._shadow = [new_core, settle]
            self._pending_freqs = new_freqs
            self.switch_log.append({"n": self.n, "policy": "deferred",
                                    "frames": settle, "freqs": list(new_freqs)})
        elif policy == "swap":
            self._core.crossover.set_frequencies(new_freqs)
            self.freqs = new_freqs
            self.switch_log.append({"n": self.n, "policy": "swap",
                                    "freqs": list(new_freqs)})
        else:
            raise ValueError(f"unknown move policy {policy!r}")
        return policy

    # ---------------------------------------------------------------- analysis
    def reference_allpass(self):
        """The declared unity-gain response as a standalone filter."""
        return AllpassCascade(self.freqs, self.fs)

    def band_ranges(self):
        edges = [0.0] + self.freqs + [self.fs * 0.5]
        return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
