#!/usr/bin/env python3
"""Original score + sound design for the Japan explainer.

Hit-point driven rather than grid-locked: taiko accents land exactly on the
edit beats, over a sustained drone/pad bed. Melodic material uses the Japanese
'in' scale on D, and plucks are Karplus-Strong so they read as koto, not sine.
"""
import sys, os, math, wave, numpy as np
from scipy import signal
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from story import DUR, BEATS

SR = 48000
N = int(DUR * SR) + SR
rng = np.random.default_rng(11)
L = np.zeros(N); R = np.zeros(N)

def place(x, t, g=1.0, pan=0.0):
    i = int(t * SR)
    if i < 0: x, i = x[-i:], 0
    n = min(len(x), N - i)
    if n <= 0: return
    L[i:i+n] += x[:n] * g * math.cos((pan + 1) * math.pi / 4) * 1.414
    R[i:i+n] += x[:n] * g * math.sin((pan + 1) * math.pi / 4) * 1.414

def fade(x, ms=8):
    k = int(SR * ms / 1000)
    if len(x) > 2 * k:
        x[:k] *= np.linspace(0, 1, k); x[-k:] *= np.linspace(1, 0, k)
    return x

# ---------------------------------------------------------------- scale
# 'In' scale on D  ·  D  Eb  G  A  Bb
def nt(semi, octv=0): return 293.66 * (2 ** ((semi + 12 * octv) / 12.0))
IN = {"D": nt(0), "Eb": nt(1), "G": nt(5), "A": nt(7), "Bb": nt(8),
      "D5": nt(12), "Eb5": nt(13), "G5": nt(17), "A5": nt(19),
      "D3": nt(-12), "A3": nt(-5), "Bb3": nt(-4), "G3": nt(-7),
      "D2": nt(-24), "A2": nt(-17), "G2": nt(-19), "Bb2": nt(-16), "D1": nt(-36)}

# ---------------------------------------------------------------- voices
_pluck_cache = {}
def koto(freq, dur=1.8, damp=0.9965, bright=0.55):
    key = (round(freq, 2), round(dur, 2), damp, bright)
    if key in _pluck_cache: return _pluck_cache[key].copy()
    P = max(4, int(SR / freq))
    buf = rng.uniform(-1, 1, P)
    sos = signal.butter(2, min(0.95, bright), output="sos")
    buf = signal.sosfilt(sos, buf)
    n = int(dur * SR)
    out = np.empty(n)
    b = buf.copy(); idx = 0
    for i in range(n):
        out[i] = b[idx]
        nxt = (idx + 1) % P
        b[idx] = damp * 0.5 * (b[idx] + b[nxt])
        idx = nxt
    out *= np.exp(-np.arange(n) / (dur * SR * 0.55))
    out /= (np.max(np.abs(out)) + 1e-9)
    _pluck_cache[key] = out
    return out.copy()

def taiko(dur=1.5, f0=132, f1=52, drive=1.9, body=0.30):
    n = int(dur * SR); t = np.arange(n) / SR
    k = np.log(f1 / f0) / dur
    ph = 2 * np.pi * f0 * (np.exp(k * t) - 1) / k
    x = np.sin(ph) * np.exp(-t / body)
    skin = rng.normal(0, 1, n) * np.exp(-t / 0.030)
    skin = signal.sosfilt(signal.butter(2, [220 / (SR/2), 2600 / (SR/2)], btype="band", output="sos"), skin)
    stick = rng.normal(0, 1, n) * np.exp(-t / 0.0035)
    x = np.tanh((x + 0.42 * skin + 0.18 * stick) * drive) / np.tanh(drive)
    return fade(x, 3)

def pad(freqs, dur, bright=0.55):
    n = int(dur * SR); t = np.arange(n) / SR
    x = np.zeros(n)
    for f in freqs:
        for h, amp in ((1, 1.0), (2, 0.32), (3, 0.16), (4, 0.08), (6, 0.04)):
            det = 1.0 + rng.uniform(-0.0016, 0.0016)
            x += amp * np.sin(2 * np.pi * f * h * det * t + rng.uniform(0, 6.28))
    x /= len(freqs) * 1.6
    cut = 480 + 900 * bright * (0.5 + 0.5 * np.sin(2 * np.pi * t / max(dur, 1) - 1.5))
    x = signal.sosfilt(signal.butter(2, np.clip(cut.mean(), 200, 6000) / (SR/2), output="sos"), x)
    a = np.minimum(1.0, t / (dur * 0.30)) * np.minimum(1.0, (dur - t) / (dur * 0.34))
    x *= a * (0.86 + 0.14 * np.sin(2 * np.pi * t * 0.13))
    return fade(x, 40)

def drone(freq, dur):
    n = int(dur * SR); t = np.arange(n) / SR
    x = np.sin(2 * np.pi * freq * t) + 0.30 * np.sin(2 * np.pi * freq * 2 * t)
    x *= 0.86 + 0.14 * np.sin(2 * np.pi * t * 0.09)
    a = np.minimum(1.0, t / 1.6) * np.minimum(1.0, (dur - t) / 1.8)
    return fade(x * a, 60)

def shaku(freq, dur=2.6):
    """Breathy end-blown-flute tone: vibrato sine plus filtered breath noise."""
    n = int(dur * SR); t = np.arange(n) / SR
    vib = 1 + 0.006 * np.sin(2 * np.pi * 4.6 * t) * np.minimum(1, t / 0.7)
    x = np.sin(2 * np.pi * freq * np.cumsum(vib) / SR) + 0.20 * np.sin(4 * np.pi * freq * t)
    br = rng.normal(0, 1, n)
    br = signal.sosfilt(signal.butter(2, [freq * 0.8 / (SR/2), min(0.95, freq * 6 / (SR/2))],
                                      btype="band", output="sos"), br)
    x = 0.92 * x + 0.19 * br
    a = np.minimum(1.0, t / 0.42) * np.minimum(1.0, (dur - t) / 0.9)
    return fade(x * a, 30)

def whoosh(dur=0.55, bright=1.0, rev=False):
    n = int(dur * SR); src = rng.normal(0, 1, n); out = np.zeros(n)
    steps = 22
    for i in range(steps):
        u = (i + .5) / steps
        fc = float(np.clip(240 * (5200 * bright / 240) ** (u ** 0.75), 60, SR/2 * 0.9))
        lo, hi = max(20, fc * 0.55) / (SR/2), min(SR/2 * 0.95, fc * 1.8) / (SR/2)
        seg = signal.sosfilt(signal.butter(2, [lo, hi], btype="band", output="sos"), src)
        w = np.zeros(n); a0, b0 = int(n*i/steps), int(n*(i+1)/steps)
        pad_ = int((b0-a0)*0.7); s0, e0 = max(0,a0-pad_), min(n,b0+pad_)
        ramp = np.concatenate([np.linspace(0,1,a0-s0), np.linspace(1,0,e0-a0)])[:e0-s0]
        w[s0:e0] = ramp
        out += seg * w
    out /= (np.max(np.abs(out)) + 1e-9)
    out *= np.sin(np.linspace(0, np.pi, n)) ** 1.6
    if rev: out = out[::-1].copy()
    return fade(out, 10)

def riser(dur=1.5):
    n = int(dur * SR); t = np.linspace(0, 1, n)
    src = rng.normal(0, 1, n); out = np.zeros(n); steps = 20
    for i in range(steps):
        u = (i + .5) / steps
        fc = float(np.clip(300 * (9000/300) ** (u ** 1.9), 60, SR/2*0.9))
        seg = signal.sosfilt(signal.butter(2, [max(20, fc*.6)/(SR/2), min(SR/2*.95, fc*1.7)/(SR/2)],
                                           btype="band", output="sos"), src)
        w = np.zeros(n); a0, b0 = int(n*i/steps), int(n*(i+1)/steps); w[a0:b0] = 1
        out += seg * w
    tone = np.sin(2 * np.pi * np.cumsum(220 * (1 + 6 * t**2)) / SR) * 0.4
    return fade((out / (np.abs(out).max()+1e-9) * .9 + tone) * t**2.2, 8)

# ================================================================ arrangement
SECTIONS = [                        # (start, end, chord, level)
    (0.0,  11.2, ["D2", "A2", "D3"],        0.30),
    (11.2, 24.0, ["Bb2", "D3", "A3"],       0.40),
    (24.0, 31.0, ["D2", "A2", "D3"],        0.44),
    (31.0, 42.0, ["G2", "Bb2", "D3"],       0.46),
    (42.0, 55.2, ["Bb2", "D3", "G3"],       0.48),
    (55.2, 62.0, ["D2", "A2", "D3"],        0.52),
    (62.0, DUR,  ["Bb2", "D3", "A3"],       0.56),
]
for (s, e, ch, lv) in SECTIONS:
    place(pad([IN[c] for c in ch], e - s + 1.4, bright=0.5 + lv * 0.5), s - 0.5, lv * 0.30)
place(drone(IN["D1"], DUR + 1.0), 0.0, 0.30)

# taiko: a hit on every edit beat, plus a softer answer between beats
for i, b in enumerate(BEATS):
    place(taiko(1.7, 138, 50, 2.0, 0.34), b, 0.62 if i in (0, 5, 11) else 0.46)
    if i + 1 < len(BEATS):
        mid = (b + BEATS[i + 1]) / 2
        place(taiko(1.1, 120, 54, 1.6, 0.22), mid, 0.24, pan=0.18 if i % 2 else -0.18)

# koto phrases
PHRASES = [
    (12.0, ["D5", "Bb", "A", "G"],       0.62, 0.30),
    (18.2, ["A", "Bb", "D5", "A"],       0.58, 0.26),
    (24.8, ["D5", "A", "G", "Eb5"],      0.60, 0.28),
    (43.0, ["Bb", "D5", "G5", "A5"],     0.55, 0.30),
    (49.2, ["D5", "A5", "G5", "D5"],     0.52, 0.28),
    (56.0, ["A", "Bb", "D5", "A5"],      0.48, 0.32),
    (63.0, ["Bb", "D5", "A5", "D5"],     0.55, 0.34),
    (68.8, ["D5", "A", "D"],             0.90, 0.36),
]
for start, notes, step, gain in PHRASES:
    for k, nm in enumerate(notes):
        place(koto(IN[nm], 2.0), start + k * step, gain * (1.0 - 0.10 * k),
              pan=-0.22 + 0.44 * (k % 2))

# the emotional low: strip back to a lone flute over the chart
place(shaku(IN["A"], 3.0),  33.4, 0.26, pan=-0.12)
place(shaku(IN["G"], 2.6),  36.6, 0.22, pan=0.14)
place(shaku(IN["D"], 3.4),  39.0, 0.20, pan=-0.08)

# ---------------------------------------------------------------- sfx
place(whoosh(0.9, 1.2, rev=True), 0.00, 0.34, -0.15)
for i, b in enumerate(BEATS):
    place(whoosh(0.55, 1.0 + 0.12 * (i % 3)), b - 0.30, 0.26, 0.32 if i % 2 else -0.32)
for ct in (17.85, 24.45, 42.05, 48.65, 55.65):        # stat cards
    place(whoosh(0.34, 1.5), ct - 0.16, 0.20, 0.22)
    place(taiko(0.9, 150, 60, 1.5, 0.16), ct, 0.30)
for bt, *_ in [(1.75,), (2.65,)]:                      # the two blasts
    place(taiko(2.4, 160, 34, 2.4, 0.55), bt, 0.72)
    place(whoosh(0.7, 0.8), bt - 0.18, 0.30)
place(riser(1.6), 60.5, 0.26)
place(riser(1.4), 67.2, 0.24)
place(taiko(2.6, 150, 42, 2.2, 0.50), 68.5, 0.70)

# ================================================================ master
st = np.stack([L, R], 1)[:int(DUR * SR)]
# gentle bus compression, then peak normalise
env = np.abs(st).max(axis=1)
w = int(SR * 0.02)
env = np.convolve(env, np.ones(w) / w, mode="same")
thr, ratio = 0.42, 3.2
gain = np.where(env > thr, (thr + (env - thr) / ratio) / np.maximum(env, 1e-9), 1.0)
gain = np.convolve(gain, np.ones(w) / w, mode="same")
st *= gain[:, None]
st = signal.sosfilt(signal.butter(2, 26 / (SR/2), btype="high", output="sos"), st, axis=0)
st /= (np.max(np.abs(st)) + 1e-9)
st *= 0.90
out = os.path.join(HERE, "score.wav")
w_ = wave.open(out, "w"); w_.setnchannels(2); w_.setsampwidth(2); w_.setframerate(SR)
w_.writeframes((np.clip(st, -1, 1) * 32767).astype("<i2").tobytes()); w_.close()
print("score.wav  %.2f s" % (len(st) / SR))
