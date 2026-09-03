"""Offline music bed for the video.

The bed itself is synthesised additively with numpy - a soft pad progression, a
quiet arpeggio, and a bell on each word reveal. Pre-rendered voiceover clips are
mixed on top, and the bed ducks under them so the narration stays clear.
"""
import subprocess

import imageio_ffmpeg
import numpy as np

RATE = 44100
BPM = 82
CHORDS = [  # A minor - F - C - G, one bar each
    (57, 60, 64, 69),
    (53, 57, 60, 65),
    (48, 52, 55, 60),
    (55, 59, 62, 67),
]


def _hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def _saw(t, freq, harmonics=9):
    """Band-limited saw: harmonics summed with 1/n rolloff, which is also the
    lowpass character we want without needing a filter."""
    out = np.zeros_like(t)
    for n in range(1, harmonics + 1):
        out += np.sin(2 * np.pi * freq * n * t) / n
    return out / np.log(harmonics + 1)


def _add(out, start, seg):
    """Accumulate `seg` into `out` at `start`, clipping anything past the end."""
    end = min(len(out), start + len(seg))
    if end > start:
        out[start:end] += seg[:end - start]


def _env(n, attack, release, rate=RATE):
    e = np.ones(n)
    a, r = int(attack * rate), int(release * rate)
    if a:
        e[:a] = np.linspace(0, 1, a) ** 1.6
    if r:
        e[-r:] *= np.linspace(1, 0, r) ** 1.4
    return e


def _pad(duration):
    bar = 4 * 60.0 / BPM
    out = np.zeros(int(duration * RATE) + RATE)
    for i in range(int(duration / bar) + 2):
        chord = CHORDS[i % len(CHORDS)]
        start = int(i * bar * RATE)
        n = int(bar * 1.25 * RATE)
        t = np.arange(n) / RATE
        voice = np.zeros(n)
        for m in chord:
            for detune in (-0.12, 0.0, 0.12):  # slight spread keeps it warm
                voice += _saw(t, _hz(m) + detune, harmonics=7)
        voice *= _env(n, bar * 0.35, bar * 0.55) / (len(chord) * 3)
        _add(out, start, voice)
    return out[:int(duration * RATE)]


def _arp(duration):
    step = 60.0 / BPM / 2  # eighth notes
    bar = 4 * 60.0 / BPM
    out = np.zeros(int(duration * RATE) + RATE)
    for i in range(int(duration / step) + 1):
        chord = CHORDS[int(i * step / bar) % len(CHORDS)]
        midi = chord[[0, 2, 3, 2][i % 4]] + 12
        n = int(step * 2.4 * RATE)
        t = np.arange(n) / RATE
        f = _hz(midi)
        note = np.sin(2 * np.pi * f * t) + 0.28 * np.sin(2 * np.pi * f * 2 * t)
        note *= np.exp(-t * 6.5) * _env(n, 0.004, step * 0.4)
        start = int(i * step * RATE)
        _add(out, start, note * 0.30)
    return out[:int(duration * RATE)]


def _bell(duration, times):
    """A short two-partial chime marking each reveal."""
    out = np.zeros(int(duration * RATE) + RATE)
    n = int(1.5 * RATE)
    t = np.arange(n) / RATE
    for k, when in enumerate(times):
        f = _hz(81 + [0, 4, 7, 12, 7][k % 5])
        tone = (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 2.76 * t)) * np.exp(-t * 4.2)
        start = int(when * RATE)
        _add(out, start, tone * 0.34)
    return out[:int(duration * RATE)]


def load_clip(path):
    """Decode any audio file to a mono float array at RATE, via ffmpeg."""
    raw = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", str(path),
         "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)


def _voice_bus(duration, clips):
    """Lay the voiceover clips onto one track at their scheduled times."""
    bus = np.zeros(int(duration * RATE))
    for clip in clips:
        audio = load_clip(clip["file"])
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * clip.get("gain", 0.92)
        _add(bus, int(clip["at"] * RATE), audio)
    return bus


def _duck(bus, depth=0.72, window=0.30):
    """Gain curve for the music: dips wherever the voice bus has energy."""
    win = int(window * RATE)
    energy = np.convolve(np.abs(bus), np.ones(win) / win, mode="same")
    if energy.max() > 0:
        energy /= energy.max()
    return 1.0 - depth * np.clip(energy * 2.2, 0, 1)


def build(duration, reveal_times, path, voice=()):
    """Render the bed - plus any voiceover - to a 16-bit stereo WAV at `path`."""
    bed = _pad(duration) * 0.52 + _arp(duration) * 0.22 + _bell(duration, reveal_times) * 0.26

    if voice:
        bus = _voice_bus(duration, voice)
        mix = bed * _duck(bus) * 0.85 + bus
    else:
        mix = bed

    fade = int(1.2 * RATE)
    mix[:fade] *= np.linspace(0, 1, fade)
    mix[-fade:] *= np.linspace(1, 0, fade)
    mix /= max(1e-6, np.abs(mix).max())
    mix *= 0.88

    # Widen: delay one channel by a few milliseconds.
    d = int(0.012 * RATE)
    left, right = mix, np.concatenate([np.zeros(d), mix[:-d]])
    stereo = np.stack([left, right * 0.94], axis=1)

    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((stereo * 32767).astype("<i2").tobytes())
    return path


if __name__ == "__main__":
    build(23.4, [5.4, 9.0, 12.6, 16.2, 19.8], "out/_music.wav")

