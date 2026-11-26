#!/usr/bin/env python3
"""End-to-end demo of the mbdyn multiband dynamics kernel.

Renders the verification materials to demo_out/*.wav and prints the
measured numbers (reconstruction, band boundaries, drum hit, independent
LF/HF control, crossover-move policies, bypass recording vs live path).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mbdyn import MultibandEngine
from mbdyn.crossover import AllpassCascade, MultibandCrossover
from mbdyn.panel import wav16_bytes

FS = 44100.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_out")


def save(name, samples):
    path = os.path.join(OUT, name)
    with open(path, "wb") as f:
        f.write(wav16_bytes(samples, FS))
    return path


def tone(f, amp, dur, phase=0.0):
    n = int(FS * dur)
    return [amp * math.sin(math.tau * f * i / FS + phase) for i in range(n)]


def chirp(f0, f1, amp, dur):
    n = int(FS * dur)
    k = math.log(f1 / f0) / dur
    ph = 0.0
    out = []
    for i in range(n):
        ph += math.tau * f0 * math.exp(k * i / FS) / FS
        out.append(amp * math.sin(ph))
    return out


def drum(dur=1.0):
    n = int(FS * dur)
    out = [0.0] * n
    for t0 in (0.10, 0.55):
        i0 = int(t0 * FS)
        for i in range(i0, n):
            t = (i - i0) / FS
            out[i] += 0.9 * math.exp(-t / 0.09) * math.sin(math.tau * 55.0 * t)
            if t < 0.008:
                out[i] += 0.5 * math.exp(-t / 0.0012) * math.sin(math.tau * 2500.0 * t)
    return out


def lf_hf(dur=1.6):
    n = int(FS * dur)
    out = []
    for i in range(n):
        t = i / FS
        lf = 0.12 if t < 0.4 or t >= 1.2 else 0.9
        hf = 0.75 if t < 0.8 else 0.10
        out.append(lf * math.sin(math.tau * 60.0 * t)
                   + hf * math.sin(math.tau * 6000.0 * t))
    return out


def dft_mag(x, f):
    w = math.tau * f / FS
    step = complex(math.cos(-w), math.sin(-w))
    z = 1 + 0j
    acc = 0 + 0j
    for v in x:
        acc += v * z
        z *= step
    return abs(acc) * 2.0 / len(x)


def db(v):
    return 20.0 * math.log10(max(abs(v), 1e-15))


def maxdiff(a, b):
    return max(abs(p - q) for p, q in zip(a, b))


def main():
    os.makedirs(OUT, exist_ok=True)
    xover = [250.0, 2500.0]

    print("== 1. 重建响应(声明:平坦幅度 + LR4 全通相位)==")
    xo = MultibandCrossover(xover, FS)
    worst = 0.0
    for i in range(600):
        f = 20.0 * (FS / 2 / 20.0) ** (i / 599)
        worst = max(worst, abs(abs(xo.H_sum(math.tau * f / FS)) - 1.0))
    print(f"   |H_sum| 与 1 的最大偏差(4 频带树): {worst:.3e}")
    eng = MultibandEngine(FS, xover)
    sw = chirp(30.0, 16000.0, 0.8, 1.0)
    err = maxdiff(eng.process(sw), AllpassCascade(xover, FS).process(sw))
    print(f"   扫频重组误差 vs 声明全通: {err:.3e}")

    print("== 2. 分频边界双音 == ")
    eng = MultibandEngine(FS, xover)
    eng.probe_bands = True
    x = [0.4 * math.sin(math.tau * xover[0] * i / FS)
         + 0.4 * math.sin(math.tau * xover[1] * i / FS) for i in range(int(FS * 0.3))]
    eng.process(x)
    skip = int(FS * 0.1)
    for k, tr in enumerate(eng.band_trace):
        a1 = db(dft_mag(tr[skip:], xover[0]) / 0.4)
        a2 = db(dft_mag(tr[skip:], xover[1]) / 0.4)
        print(f"   band{k}: @{xover[0]:.0f}Hz {a1:7.2f} dB   @{xover[1]:.0f}Hz {a2:7.2f} dB")

    print("== 3. 鼓击:真实合流后的首峰与低频凹陷 ==")
    x = drum()
    eng = MultibandEngine(FS, xover)
    eng.set_band(0, threshold_db=-18.0, ratio=3.0, knee_db=0.0,
                 attack_ms=25.0, release_ms=200.0)
    eng.probe_gains = True
    y = eng.process(x)
    eng2 = MultibandEngine(FS, xover)
    eng2.set_band(0, threshold_db=-18.0, ratio=3.0, knee_db=0.0,
                  attack_ms=25.0, release_ms=200.0)
    y2 = []
    for i in range(0, len(x), 113):
        y2.extend(eng2.process(x[i:i + 113]))
    print(f"   跨块(113 样本)与整段处理完全一致: {y == y2}")
    i0 = int(0.10 * FS)
    win = slice(i0, i0 + int(0.012 * FS))
    y_unity = MultibandEngine(FS, xover).process(x)
    pin = max(abs(v) for v in x[win])
    pout = max(abs(v) for v in y[win])
    pref = max(abs(v) for v in y_unity[win])
    print(f"   首个峰 in {pin:.3f} -> out {pout:.3f} "
          f"(相对单位增益全通 {pout / pref:.3f})")
    g = eng.gain_trace[0]
    print(f"   低频带最大衰减 {min(g):.1f} dB, 末尾恢复至 {g[-1]:.2f} dB")
    save("drum_in.wav", x)
    save("drum_out.wav", y)

    print("== 4. 低频/高频动态独立控制 ==")
    x = lf_hf()
    eng = MultibandEngine(FS, xover)
    eng.set_band(0, threshold_db=-20.0, ratio=6.0, knee_db=0.0,
                 attack_ms=2.0, release_ms=60.0)
    eng.set_band(2, threshold_db=-6.0, ratio=6.0, knee_db=0.0,
                 attack_ms=2.0, release_ms=60.0)
    eng.probe_gains = True
    y = eng.process(x)
    gl, _, gh = eng.gain_trace
    def avg(tr, t0, t1):
        seg = tr[int(t0 * FS):int(t1 * FS)]
        return sum(seg) / len(seg)
    print(f"   段2(低响/高响): LF GR {avg(gl, 0.5, 0.7):6.1f} dB, HF GR {avg(gh, 0.5, 0.7):6.1f} dB")
    print(f"   段3(低响/高静): LF GR {avg(gl, 0.9, 1.1):6.1f} dB, HF GR {avg(gh, 0.9, 1.1):6.1f} dB")
    save("lfhf_in.wav", x)
    save("lfhf_out.wav", y)

    print("== 5. 持续音中移动分频点:状态映射 vs 双跑淡接 vs 粗糙切换 ==")
    post = tone(300.0, 0.5, 0.15)
    n_move = int(FS * 0.3)
    xo_new = MultibandCrossover([900.0, 2500.0], FS)
    h = xo_new.H_sum(math.tau * 300.0 / FS)
    ideal = [0.5 * math.sin(math.tau * 300.0 * (n_move + k) / FS
                            + math.atan2(h.imag, h.real)) for k in range(len(post))]
    for policy in ("map", "xfade", "swap"):
        eng = MultibandEngine(FS, xover)
        eng.process(tone(300.0, 0.5, 0.3))
        hint = (300.0, 0.5, math.tau * 300.0 * n_move / FS - math.pi / 2)
        eng.move_crossovers([900.0, 2500.0], policy=policy,
                            tone_hint=hint if policy == "map" else None)
        err = maxdiff(eng.process(post), ideal)
        print(f"   {policy:6s} 重组误差: {err:.3e}")

    print("== 6. 同相/反相尾振两组,切换瞬间输入相等 ==")
    for name, sil in (("同相", 96), ("反相", 96 + int(FS / 800.0))):
        eng = MultibandEngine(FS, xover)
        eng.process(tone(250.0, 0.8, 0.03))
        eng.process([0.0] * sil)
        eng.move_crossovers([900.0, 2500.0], policy="map",
                            tone_hint=(300.0, 0.5, -math.pi / 2))
        err = maxdiff(eng.process(post), ideal)
        print(f"   {name}尾振 map 误差: {err:.3e}")

    print("== 7. 同一次扫频:旁路录音 + 实时伴奏(分频扫动)==")
    x = chirp(30.0, 16000.0, 0.8, 1.0)
    rec = MultibandEngine(FS, xover)
    rec.set_bypass(True)
    rec.record_input = True
    live = MultibandEngine(FS, xover)
    y_live = []
    for bi, i in enumerate(range(0, len(x), 256)):
        if bi == 40:
            live.move_crossovers([500.0, 2500.0], policy="deferred")
        if bi == 80:
            live.move_crossovers([900.0, 2500.0], policy="deferred")
        y_live.extend(live.process(x[i:i + 256]))
        rec.process(x[i:i + 256])
    print(f"   旁路录音与输入位精确一致: {rec.recorded == x}")
    print(f"   实时路径峰值 {max(abs(v) for v in y_live):.3f} (输入 {max(abs(v) for v in x):.3f})")
    print(f"   换带策略日志: {[e['policy'] for e in live.switch_log]}")
    save("sweep_bypass_record.wav", rec.recorded)
    save("sweep_live.wav", y_live)

    print(f"\nWAV 已写入 {OUT}/")
    print("启动母带试听面板: python3 -m mbdyn.panel --port 8137")


if __name__ == "__main__":
    main()
