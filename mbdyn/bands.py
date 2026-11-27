"""Band processing with separated state ownership.

Three state domains are kept deliberately distinct:

  filter memory    ``MultibandCrossover`` biquad states      (per topology)
  envelope memory  one ``BandDynamics.env`` per band          (per source)
  band gain        pure function of (env, config)             (no memory)

A band's *detection source* is its frequency interval ``(lo, hi)``.  Only
an envelope whose source is unchanged may follow a band across a topology
change.  If the interval changes the old envelope measured something else
and must be rebuilt (reset), otherwise an unreleased compression gain
leaks onto a signal the detector never saw.

The recombination boundaries are explicit:

  ``retune_in_place``       filter memory: retune same-count topology
  ``map_filter_state``      filter memory: align to a known steady tone
  ``recombine_envelopes``   envelope memory: inherit or rebuild by source
  ``envelope_snapshot``     envelope memory: read before a change

``BandProcessing`` is the unified per-topology processing unit: it takes a
block/sample through split -> detect/gain -> recombine and owns the band
configs so a transition can run old and new topologies side by side.
"""

from .crossover import MultibandCrossover
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


def band_ranges(freqs, fs):
    """Frequency intervals ``(lo, hi)`` of each band for ``freqs``."""
    edges = [0.0] + [float(f) for f in freqs] + [float(fs) * 0.5]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def match_band_intervals(old_freqs, new_freqs, fs):
    """Map ``new_index -> old_index`` for bands with an identical interval.

    Crossover frequencies are strictly increasing, so every interval is
    unique and a new band matches at most one old band.
    """
    old_ranges = band_ranges(old_freqs, fs)
    by_range = {r: i for i, r in enumerate(old_ranges)}
    return {j: by_range[r] for j, r in enumerate(band_ranges(new_freqs, fs))
            if r in by_range}


def configs_for_topology(old_configs, old_freqs, new_freqs, fs):
    """Band configs for ``new_freqs``, aligned to their detection source.

    Identical-interval bands reuse the exact ``BandConfig`` object (only
    the index may move).  A genuinely new interval (a split or merge)
    inherits a copy of the old config whose range contained its centre, so
    user parameters follow the material.
    """
    old_ranges = band_ranges(old_freqs, fs)
    new_ranges = band_ranges(new_freqs, fs)
    if len(new_ranges) == len(old_configs):
        return list(old_configs)
    by_range = {r: cfg for r, cfg in zip(old_ranges, old_configs)}
    out = []
    for r in new_ranges:
        if r in by_range:
            out.append(by_range[r])
            continue
        centre = 0.5 * (r[0] + r[1])
        src = None
        for (lo, hi), cfg in zip(old_ranges, old_configs):
            if lo <= centre < hi:
                src = cfg
                break
        out.append(src.copy() if src is not None else BandConfig())
    return out


class BandProcessing:
    """One topology: filters + per-band envelopes + aligned band delays.

    The object is the unified processing and parameter-update interface;
    the engine owns transitions between topologies.
    """

    def __init__(self, fs, freqs, configs, lookahead):
        self.fs = float(fs)
        self.freqs = [float(f) for f in freqs]
        self.configs = list(configs)
        self.crossover = MultibandCrossover(self.freqs, fs)
        self.dyns = [BandDynamics(fs, c) for c in self.configs]
        self.delays = [_Delay(lookahead) for _ in self.configs]

    @property
    def num_bands(self):
        return len(self.configs)

    @property
    def ranges(self):
        return band_ranges(self.freqs, self.fs)

    # ------------------------------------------------------ processing
    def step(self, x, ctl, probe=True):
        """Compat adapter: the pre-refactor ``_Core.step(x, engine, probe)``."""
        return self.process_sample(x, ctl, probe=probe)

    def process_sample(self, x, ctl, probe=True):
        bands = self.crossover.step(x)
        out = 0.0
        solo_any = any(c.solo for c in self.configs)
        for k, cfg in enumerate(self.configs):
            xd = self.delays[k].step(bands[k])
            if ctl.processing_on and cfg.enabled:
                g = self.dyns[k].step_gain(bands[k])
            else:
                g = 1.0
                if ctl.processing_on:
                    self.dyns[k].track_only(bands[k])
            post = xd * g
            if probe:
                if ctl.probe_bands and k < len(ctl.band_trace):
                    ctl.band_trace[k].append(post)
                if ctl.probe_gains and k < len(ctl.gain_trace):
                    ctl.gain_trace[k].append(
                        self.dyns[k].gain_db()
                        if cfg.enabled and ctl.processing_on else 0.0)
            if (cfg.solo if solo_any else not cfg.mute):
                out += post
        return out

    def update_config(self, k, **params):
        cfg = self.configs[k]
        for key, val in params.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown band parameter {key!r}")
            setattr(cfg, key, val)

    # --------------------------------------------- envelope memory boundary
    def envelope_snapshot(self):
        return [d.envelope() for d in self.dyns]

    def recombine_envelopes(self, old_freqs, old_envs):
        """Inherit envelopes for identical sources; rebuild every other.

        Must be called with ``self.freqs`` already set to the new topology.
        Returns the applied ``new_index -> old_index`` mapping.
        """
        mapping = match_band_intervals(old_freqs, self.freqs, self.fs)
        for new_k, old_k in mapping.items():
            self.dyns[new_k].set_envelope(old_envs[old_k])
        for j, d in enumerate(self.dyns):
            if j not in mapping:
                d.reset_envelope()
        return mapping

    # ----------------------------------------------- filter memory boundary
    def retune_in_place(self, new_freqs):
        """Retune a same-count topology, keeping raw filter memory."""
        if len(new_freqs) != len(self.freqs):
            raise ValueError("in-place retune needs the same crossover count")
        self.crossover.set_frequencies(new_freqs)
        self.freqs = [float(f) for f in new_freqs]

    def map_filter_state(self, w, X):
        self.crossover.map_state(w, X)
