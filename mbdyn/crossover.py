"""Linkwitz-Riley 4th-order crossover tree with allpass compensation.

Structure for crossovers f1 < f2 < ... < fM (M stages, M+1 bands):
  rest_0 = x
  stage i:  lp_i = LP4_i(rest_{i-1}),  rest_i = HP4_i(rest_{i-1})
  band k (0<=k<M): LP_{k+1} output passed through compensation allpasses
                   A_j = LP4_j + HP4_j of every higher stage j >= k+2
  band M: rest_M

Reconstruction identity (declared response, all band gains = 1):
  sum_k B_k = A_1 * A_2 * ... * A_M   (cascade of stage allpasses)
so the magnitude is exactly flat and the phase is the sum of the stage
allpass phases. This is NOT a stack of arbitrary LP/HP filters: each
lower band is re-phased by the allpass sums of the stages above it, which
is what keeps the recombined response free of dips/peaks at any spacing.
"""

from .biquad import Biquad, butter_hp2, butter_lp2, biquad_H


class Section:
    """4th-order LR half (two cascaded 2nd-order Butterworth biquads)."""

    def __init__(self, kind, fc, fs):
        self.kind = kind
        self.fc = fc
        self.fs = fs
        self.bq = [Biquad(self._coeffs()), Biquad(self._coeffs())]

    def _coeffs(self):
        if self.kind == "lp":
            return butter_lp2(self.fc, self.fs)
        return butter_hp2(self.fc, self.fs)

    def set_fc(self, fc):
        self.fc = fc
        c = self._coeffs()
        self.bq[0].set_coeffs(c)
        self.bq[1].set_coeffs(c)

    def step(self, x):
        return self.bq[1].step(self.bq[0].step(x))

    def H(self, w):
        h = biquad_H(self.bq[0].c, w)
        return h * h

    def map_state(self, w, X):
        """Align both biquads' memory to steady state for input phasor X."""
        self.bq[0].steady_state(w, X)
        self.bq[1].steady_state(w, X * biquad_H(self.bq[0].c, w))


class AllpassSection:
    """LP4 + HP4 sum of one stage: |H| == 1 by the LR4 identity."""

    def __init__(self, fc, fs):
        self.fc = fc
        self.lp = Section("lp", fc, fs)
        self.hp = Section("hp", fc, fs)

    def set_fc(self, fc):
        self.fc = fc
        self.lp.set_fc(fc)
        self.hp.set_fc(fc)

    def step(self, x):
        return self.lp.step(x) + self.hp.step(x)

    def H(self, w):
        return self.lp.H(w) + self.hp.H(w)

    def map_state(self, w, X):
        self.lp.map_state(w, X)
        self.hp.map_state(w, X)


class AllpassCascade:
    """Standalone cascade of stage allpasses: the declared unity response.

    Independent of the band tree wiring; used as the reconstruction
    reference in tests and in the panel's audition report.
    """

    def __init__(self, freqs, fs):
        self.stages = [AllpassSection(f, fs) for f in freqs]

    def step(self, x):
        for s in self.stages:
            x = s.step(x)
        return x

    def process(self, xs):
        return [self.step(x) for x in xs]

    def H(self, w):
        h = 1 + 0j
        for s in self.stages:
            h *= s.H(w)
        return h


class MultibandCrossover:
    def __init__(self, freqs, fs):
        self.fs = fs
        self.freqs = [float(f) for f in freqs]
        m = len(self.freqs)
        self.lp = [Section("lp", f, fs) for f in self.freqs]
        self.hp = [Section("hp", f, fs) for f in self.freqs]
        # comp[(j, k)]: allpass of stage j applied to band k's path (j > k)
        self.comp = {}
        for j in range(m):
            for k in range(j):
                self.comp[(j, k)] = AllpassSection(self.freqs[j], fs)

    @property
    def num_bands(self):
        return len(self.freqs) + 1

    def set_frequencies(self, freqs):
        """Retune every stage (states kept as-is; caller may map them)."""
        self.freqs = [float(f) for f in freqs]
        for i, f in enumerate(self.freqs):
            self.lp[i].set_fc(f)
            self.hp[i].set_fc(f)
        for (j, _k), a in self.comp.items():
            a.set_fc(self.freqs[j])

    def step(self, x):
        m = len(self.freqs)
        rest = x
        lp_out = []
        for i in range(m):
            lp_out.append(self.lp[i].step(rest))
            rest = self.hp[i].step(rest)
        bands = [0.0] * (m + 1)
        bands[m] = rest
        for k in range(m):
            sig = lp_out[k]
            for j in range(k + 1, m):
                sig = self.comp[(j, k)].step(sig)
            bands[k] = sig
        return bands

    def H_band(self, k, w):
        m = len(self.freqs)
        h = 1 + 0j
        for i in range(k):
            h *= self.hp[i].H(w)
        if k < m:
            h *= self.lp[k].H(w)
        for j in range(k + 1, m):
            h *= self.comp[(j, k)].H(w)
        return h

    def H_sum(self, w):
        return sum(self.H_band(k, w) for k in range(self.num_bands))

    def map_state(self, w, X):
        """Align every section's memory to the steady state of a tone.

        X is the input phasor at the crossover input (x[n]=Re(X e^{jwn})).
        Node phasors are computed with the CURRENT coefficients, so after
        a retune every band filter's memory matches the new steady state.
        """
        m = len(self.freqs)
        p_rest = [X]
        for i in range(m):
            p_rest.append(p_rest[-1] * self.hp[i].H(w))
        for i in range(m):
            self.hp[i].map_state(w, p_rest[i])
            self.lp[i].map_state(w, p_rest[i])
        for k in range(m):
            p = p_rest[k] * self.lp[k].H(w)
            for j in range(k + 1, m):
                self.comp[(j, k)].map_state(w, p)
                p *= self.comp[(j, k)].H(w)
