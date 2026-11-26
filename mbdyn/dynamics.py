"""Per-band dynamics: peak envelope detector + soft-knee gain computer.

State (the envelope) persists across audio blocks; the same detector
coefficients are used for the gain path, so gain rides the smoothed
envelope in dB. Detector always tracks the band signal (even when the
band's processing is disabled) so meters and re-engagement stay honest.
"""

import math


class BandConfig:
    def __init__(self):
        self.enabled = True       # False: band passes at unity gain (not removed!)
        self.threshold_db = 0.0   # ratio 1.0 + threshold 0 dB => unity by default
        self.ratio = 1.0
        self.knee_db = 6.0
        self.attack_ms = 10.0
        self.release_ms = 120.0
        self.makeup_db = 0.0
        self.solo = False
        self.mute = False

    def copy(self):
        c = BandConfig()
        c.__dict__.update(self.__dict__)
        return c


def static_gain_db(level_db, threshold_db, ratio, knee_db):
    """Soft-knee compressor characteristic; returns gain in dB (<= 0)."""
    t = threshold_db
    if knee_db > 0.0:
        lo, hi = t - knee_db * 0.5, t + knee_db * 0.5
        if level_db <= lo:
            return 0.0
        if level_db >= hi:
            return (level_db - t) * (1.0 / ratio - 1.0)
        d = level_db - lo
        return (1.0 / ratio - 1.0) * d * d / (2.0 * knee_db)
    if level_db <= t:
        return 0.0
    return (level_db - t) * (1.0 / ratio - 1.0)


class BandDynamics:
    def __init__(self, fs, cfg):
        self.fs = float(fs)
        self.cfg = cfg
        self.env = 0.0
        self._coef_key = None
        self._ca = 0.0
        self._cr = 0.0

    def _coefs(self):
        key = (self.cfg.attack_ms, self.cfg.release_ms, self.fs)
        if key != self._coef_key:
            self._coef_key = key
            self._ca = math.exp(-1.0 / (self.fs * self.cfg.attack_ms * 1e-3))
            self._cr = math.exp(-1.0 / (self.fs * self.cfg.release_ms * 1e-3))
        return self._ca, self._cr

    def _track(self, x):
        xa = abs(x)
        ca, cr = self._coefs()
        c = ca if xa > self.env else cr
        self.env = c * self.env + (1.0 - c) * xa

    def env_db(self):
        return 20.0 * math.log10(self.env + 1e-12)

    def gain_db(self):
        return static_gain_db(self.env_db(), self.cfg.threshold_db,
                              self.cfg.ratio, self.cfg.knee_db) + self.cfg.makeup_db

    def step_gain(self, x):
        """Track envelope and return the linear gain for this sample."""
        self._track(x)
        return 10.0 ** (self.gain_db() * 0.05)

    def track_only(self, x):
        self._track(x)
