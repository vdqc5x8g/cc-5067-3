"""Shared helpers for the mbdyn test suite (stdlib only)."""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)


def dft(x, fs, f):
    """Complex amplitude (2/N DFT bin) of real signal x at frequency f."""
    w = math.tau * f / fs
    step = complex(math.cos(-w), math.sin(-w))
    z = 1 + 0j
    acc = 0 + 0j
    for v in x:
        acc += v * z
        z *= step
    return acc * (2.0 / len(x))


def db(x):
    return 20.0 * math.log10(max(abs(x), 1e-15))


def tone(fs, f, amp, dur, phase=0.0):
    n = int(fs * dur)
    return [amp * math.sin(math.tau * f * i / fs + phase) for i in range(n)]


def chirp(fs, f0, f1, amp, dur):
    n = int(fs * dur)
    k = math.log(f1 / f0) / dur
    phase = 0.0
    out = []
    for i in range(n):
        t = i / fs
        phase += math.tau * f0 * math.exp(k * t) / fs
        out.append(amp * math.sin(phase))
    return out


def noise_like(n, seed=12345):
    """Deterministic pseudo-random signal in [-1, 1)."""
    out = []
    s = seed
    for _ in range(n):
        s = (1103515245 * s + 12345) % (1 << 31)
        out.append((s / (1 << 30)) - 1.0)
    return out


def drum(fs, dur=1.0, hits=(0.05,), lf=55.0):
    n = int(fs * dur)
    out = [0.0] * n
    for t0 in hits:
        i0 = int(t0 * fs)
        for i in range(i0, n):
            t = (i - i0) / fs
            out[i] += 0.9 * math.exp(-t / 0.09) * math.sin(math.tau * lf * t)
            if t < 0.008:
                out[i] += 0.5 * math.exp(-t / 0.0012) * math.sin(math.tau * 2500.0 * t)
    return out


def lf_hf_material(fs, dur=1.2, lf=60.0, hf=6000.0):
    """LF and HF step their levels independently (control independence)."""
    n = int(fs * dur)
    out = []
    q = dur / 4.0
    for i in range(n):
        t = i / fs
        lf_amp = 0.10 if t < q or t >= 3 * q else 0.90
        hf_amp = 0.80 if t < 2 * q else 0.08
        out.append(lf_amp * math.sin(math.tau * lf * t)
                   + hf_amp * math.sin(math.tau * hf * t))
    return out


def run(eng, x, block=0):
    if block <= 0:
        return eng.process(x)
    out = []
    for i in range(0, len(x), block):
        out.extend(eng.process(x[i:i + block]))
    return out


def maxdiff(a, b):
    return max(abs(p - q) for p, q in zip(a, b)) if a else 0.0
