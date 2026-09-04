#!/usr/bin/env python3
"""
Build output/german_daily_phrases.mp4 end to end.

    python3 generate_video.py

Stages: assets -> timeline (from measured clip durations) -> frames -> audio
mix -> mux -> validate. Everything downstream of the ElevenLabs generations is
deterministic, so re-running reproduces the same video byte-for-byte apart from
encoder timestamps.

The ElevenLabs voice clips and music bed live in assets/ and are committed, so
this runs offline. Pass --fetch-audio to re-pull them (see scripts/fetch_audio.py).
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import timeline as tl_mod          # noqa: E402
import audio as audio_mod          # noqa: E402
from render import Renderer, W, H, FPS   # noqa: E402

OUT_DIR = os.path.join(ROOT, "output")
BUILD = os.path.join(ROOT, "build")
VIDEO = os.path.join(OUT_DIR, "german_daily_phrases.mp4")
THUMB = os.path.join(OUT_DIR, "thumbnail.png")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_assets():
    need = [
        ("assets/character/teacher.png", "character cut-out"),
        ("assets/background/plate.png", "background plate"),
        ("assets/background/flag.png", "flag badge"),
    ]
    if all(os.path.exists(os.path.join(ROOT, p)) for p, _ in need):
        return
    log("preparing visual assets")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "prepare_assets.py")],
                   check=True)


def render_frames(timeline, silent_video):
    r = Renderer(ROOT, timeline)
    n = timeline["n_frames"]
    log(f"rendering {n} frames at {W}x{H}@{FPS}")

    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-",
         "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
         "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
         "-x264-params", "keyint=60:min-keyint=30:scenecut=0",
         silent_video],
        stdin=subprocess.PIPE)

    t0 = time.time()
    for i in range(n):
        ff.stdin.write(r.frame(i).tobytes())
        if i and i % 200 == 0:
            done = i / n
            eta = (time.time() - t0) / done * (1 - done)
            log(f"  {i}/{n} ({done * 100:.0f}%)  eta {eta:.0f}s")
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg video encode failed")
    return r


def validate(path, timeline):
    """Refuse to declare success on a file that does not meet the spec."""
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        capture_output=True, text=True, check=True).stdout)
    v = next(s for s in probe["streams"] if s["codec_type"] == "video")
    a = next(s for s in probe["streams"] if s["codec_type"] == "audio")

    num, den = v["r_frame_rate"].split("/")
    fps = int(num) / int(den)
    dur = float(probe["format"]["duration"])

    checks = [
        ("resolution 1080x1920", (v["width"], v["height"]) == (W, H)),
        ("video codec h264", v["codec_name"] == "h264"),
        ("pixel format yuv420p", v["pix_fmt"] == "yuv420p"),
        ("frame rate 30", abs(fps - 30) < 0.01),
        ("audio codec aac", a["codec_name"] == "aac"),
        ("audio stereo", a["channels"] == 2),
        ("duration 60-75s", 60.0 <= dur <= 75.0),
        ("duration matches timeline", abs(dur - timeline["duration"]) < 0.5),
    ]

    # No black frames and no clipped audio.
    black = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", path,
         "-vf", "blackdetect=d=0.12:pic_th=0.98", "-an", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    # The deliberate open/close dips are excluded: only report black in the body.
    strays = [ln for ln in black.splitlines() if "black_start" in ln
              and not (float(ln.split("black_start:")[1].split()[0]) < 0.7
                       or float(ln.split("black_start:")[1].split()[0]) > dur - 1.2)]
    checks.append(("no black frames mid-programme", not strays))

    vol = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", path, "-af", "volumedetect",
         "-f", "null", "-"], capture_output=True, text=True).stderr
    peak = next((float(ln.split("max_volume:")[1].replace("dB", ""))
                 for ln in vol.splitlines() if "max_volume" in ln), None)
    mean = next((float(ln.split("mean_volume:")[1].replace("dB", ""))
                 for ln in vol.splitlines() if "mean_volume" in ln), None)
    checks.append(("audio peak measurable", peak is not None))
    checks.append((f"audio peak {peak} dB below clipping", peak is not None and peak < -0.2))
    checks.append((f"audio present (mean {mean} dB)", mean is not None and mean > -40))

    lufs = audio_mod.measure_lufs(path)
    checks.append((f"loudness {lufs:.1f} LUFS in -17..-11", -17.0 <= lufs <= -11.0))

    # Speech must sit clearly above the music bed during the repeat pauses.
    def seg_mean(a, b):
        o = subprocess.run(["ffmpeg", "-hide_banner", "-ss", f"{a}", "-to", f"{b}",
                            "-i", path, "-af", "volumedetect", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
        return next(float(ln.split("mean_volume:")[1].replace("dB", ""))
                    for ln in o.splitlines() if "mean_volume" in ln)

    ph = timeline["phrases"]
    speech = [seg_mean(p["de_start"], p["de_end"]) for p in ph[:6]]
    gaps = [seg_mean(ph[i]["de_end"] + 0.35, ph[i + 1]["en_start"] - 0.35)
            for i in range(5)]
    headroom = sum(speech) / len(speech) - sum(gaps) / len(gaps)
    checks.append((f"voice sits {headroom:.1f} dB above the bed (>=12)", headroom >= 12.0))

    ok = True
    for name, passed in checks:
        print(f"   {'PASS' if passed else 'FAIL'}  {name}")
        ok &= passed
    return ok, dur, fps, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-only", action="store_true")
    ap.add_argument("--skip-frames", action="store_true",
                    help="reuse build/silent.mp4 (audio/mux iteration only)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)

    ensure_assets()

    log("building timeline from measured clip durations")
    timeline, cues = tl_mod.build(ROOT)
    log(f"  {timeline['duration']:.2f}s / {timeline['n_frames']} frames, "
        f"{len(cues)} voice cues")
    with open(os.path.join(BUILD, "timeline.json"), "w") as f:
        json.dump(timeline, f, indent=2)

    silent = os.path.join(BUILD, "silent.mp4")
    if args.skip_frames and os.path.exists(silent):
        log(f"reusing {silent}")
        renderer = Renderer(ROOT, timeline)
    else:
        renderer = render_frames(timeline, silent)

    if args.frames_only:
        log(f"frames only -> {silent}")
        return 0

    log("assembling voice track")
    voice = audio_mod.build_voice_track(ROOT, cues, timeline["duration"],
                                        os.path.join(BUILD, "voice.wav"))
    music, kind = audio_mod.build_music_bed(ROOT, timeline["duration"],
                                            os.path.join(BUILD, "music.wav"))
    log(f"  music bed: {kind}")
    mixed, pre_lufs, gain_db = audio_mod.mix(
        voice, music, os.path.join(BUILD, "mix.wav"), timeline["duration"], BUILD)
    log(f"  premaster {pre_lufs:.1f} LUFS, static master gain {gain_db:+.2f} dB")

    log("muxing")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", mixed,
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
         "-movflags", "+faststart", "-shortest", VIDEO], check=True)

    log("thumbnail")
    renderer.frame(int(2.0 * FPS)).save(THUMB)

    log("validating")
    ok, dur, fps, peak = validate(VIDEO, timeline)
    size_mb = os.path.getsize(VIDEO) / 1e6

    print()
    print(f"  file      {VIDEO}")
    print(f"  duration  {dur:.2f}s")
    print(f"  size      {size_mb:.1f} MB")
    print(f"  thumbnail {THUMB}")
    if not ok:
        print("\n  VALIDATION FAILED")
        return 1
    print("\n  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
