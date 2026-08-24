#!/usr/bin/env python3
"""Synthesises the sound-design bed (whooshes, impacts, risers, ticks) as a WAV."""
import sys, os, numpy as np
from scipy import signal
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from timeline import DUR, BEATS

SR = 48000
N  = int(DUR * SR) + SR // 2
rng = np.random.default_rng(7)

def env_exp(n, tau):        return np.exp(-np.arange(n) / (tau * SR))
def fade(x, ms=6):
    k = int(SR * ms / 1000)
    if len(x) > 2 * k:
        x[:k]  *= np.linspace(0, 1, k)
        x[-k:] *= np.linspace(1, 0, k)
    return x

def band_noise_sweep(dur, f0, f1, q=1.2, steps=28, curve=1.0):
    """Noise pushed through a band-pass whose centre glides f0 -> f1."""
    n = int(dur * SR)
    src = rng.normal(0, 1, n)
    out = np.zeros(n)
    edges = np.linspace(0, n, steps + 1).astype(int)
    for i in range(steps):
        u = (i + 0.5) / steps
        fc = f0 * (f1 / f0) ** (u ** curve)
        fc = float(np.clip(fc, 40, SR / 2 * 0.92))
        bw = fc / q
        lo = max(20.0, fc - bw / 2) / (SR / 2)
        hi = min(SR / 2 * 0.95, fc + bw / 2) / (SR / 2)
        sos = signal.butter(2, [lo, hi], btype="band", output="sos")
        seg = signal.sosfilt(sos, src)
        w = np.zeros(n)
        a, b = edges[i], edges[i + 1]
        pad = int((b - a) * 0.6)
        s, e = max(0, a - pad), min(n, b + pad)
        ramp = np.zeros(e - s)
        mid = (a - s)
        ramp[:mid] = np.linspace(0, 1, mid) if mid > 0 else 0
        ramp[mid:] = np.linspace(1, 0, len(ramp) - mid)
        w[s:e] = ramp
        out += seg * w
    return out / (np.max(np.abs(out)) + 1e-9)

def whoosh(dur=0.60, reverse=False, bright=1.0):
    x = band_noise_sweep(dur, 220, 5200 * bright, q=1.1, curve=0.75)
    n = len(x)
    a = np.sin(np.linspace(0, np.pi, n)) ** 1.6
    x = x * a
    if reverse:
        x = x[::-1].copy()
    return fade(x, 10)

def impact(dur=0.95, f0=95, f1=34, drive=1.6):
    n = int(dur * SR)
    t = np.arange(n) / SR
    k = np.log(f1 / f0) / dur
    ph = 2 * np.pi * f0 * (np.exp(k * t) - 1) / k
    body = np.sin(ph) * env_exp(n, 0.22)
    thud = np.sin(2 * np.pi * 150 * t) * env_exp(n, 0.05) * 0.35
    click = rng.normal(0, 1, n) * env_exp(n, 0.006) * 0.30
    lo = signal.sosfilt(signal.butter(2, 4000 / (SR / 2), output="sos"), click)
    x = np.tanh((body + thud + lo) * drive) / np.tanh(drive)
    return fade(x, 4)

def riser(dur=1.15):
    x = band_noise_sweep(dur, 260, 8200, q=2.2, curve=1.9)
    n = len(x)
    t = np.linspace(0, 1, n)
    tone = np.sin(2 * np.pi * np.cumsum(320 * (1 + 5.5 * t ** 2)) / SR) * 0.35
    x = (x * 0.9 + tone) * (t ** 2.1)
    return fade(x, 8)

def tick(freq=1250, dur=0.09):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * freq * t) * env_exp(n, 0.012)
    x += rng.normal(0, 1, n) * env_exp(n, 0.0025) * 0.25
    return fade(x, 2)

def subdrop(dur=1.4, f0=70, f1=26):
    n = int(dur * SR)
    t = np.arange(n) / SR
    k = np.log(f1 / f0) / dur
    ph = 2 * np.pi * f0 * (np.exp(k * t) - 1) / k
    return fade(np.sin(ph) * np.exp(-t / 0.7), 8)

# ------------------------------------------------------------------ arrange
L = np.zeros(N); R = np.zeros(N)
def place(x, t, gain=1.0, pan=0.0):
    i = int(t * SR)
    if i < 0:
        x = x[-i:]; i = 0
    n = min(len(x), N - i)
    if n <= 0: return
    gl = gain * np.sqrt((1 - pan) / 2 + 0.5 * (1 - abs(pan)) * 0)
    gl = gain * np.cos((pan + 1) * np.pi / 4)
    gr = gain * np.sin((pan + 1) * np.pi / 4)
    L[i:i+n] += x[:n] * gl * 1.414
    R[i:i+n] += x[:n] * gr * 1.414

SECTION_HITS = [4.80, 9.50, 12.60, 16.70, 19.90, 27.50, 33.90,
                41.30, 44.90, 49.70, 59.20, 66.30, 70.50]
CARD_HITS    = [9.75, 13.05, 39.45, 45.35, 50.30]
TITLE_SWISH  = [5.30, 17.15, 20.35, 24.10, 28.30, 34.40, 56.35, 59.90, 66.80]

# intro
place(whoosh(0.85, reverse=True, bright=1.2), 0.00, 0.42, -0.15)
place(impact(1.30, 105, 30, 1.9),             0.33, 0.72,  0.00)
place(subdrop(1.6),                           0.33, 0.40,  0.00)

for i, t in enumerate(SECTION_HITS):
    pan = -0.35 if i % 2 == 0 else 0.35
    place(whoosh(0.58, bright=1.0 + 0.1 * (i % 3)), t - 0.30, 0.34, pan)
    place(impact(0.80, 88, 33, 1.4),                t,        0.40, 0.0)

for t in CARD_HITS:
    place(impact(0.95, 100, 36, 1.7), t, 0.50, 0.0)
    place(whoosh(0.34, bright=1.4),   t - 0.16, 0.22, 0.20)
    place(tick(1500, 0.07),           t + 0.78, 0.16, -0.20)   # count-up lands

for i, t in enumerate(TITLE_SWISH):
    place(whoosh(0.40, bright=1.5), t - 0.18, 0.20, 0.30 if i % 2 else -0.30)

for t in [4.90, 9.60, 12.70, 16.80, 19.95, 27.60, 33.95, 41.35, 44.95, 49.80, 59.30, 66.40]:
    place(tick(950, 0.06), t + 0.14, 0.11, 0.0)                # chapter chip

# outro
place(riser(1.25),                69.65, 0.30,  0.0)
place(impact(1.20, 100, 28, 1.8), 70.85, 0.55,  0.0)
place(whoosh(0.50, bright=1.3),   70.55, 0.28, -0.25)
place(tick(1700, 0.08),           72.70, 0.16,  0.15)

st = np.stack([L, R], 1)[:int(DUR * SR)]
pk = np.max(np.abs(st))
st = st / pk * 0.60 if pk > 0 else st
import wave
w = wave.open(os.path.join(os.path.dirname(HERE), "sfx.wav"), "w")
w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((np.clip(st, -1, 1) * 32767).astype("<i2").tobytes())
w.close()
print("sfx.wav  %.2fs  peak %.3f" % (len(st) / SR, pk))
