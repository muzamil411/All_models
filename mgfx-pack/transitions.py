#!/usr/bin/env python3
"""
Transition builders for the pack.

Both transitions are taken from measurements of the reference video:

  flash  - a 1-2 frame white or black solid at a cut. The reference uses 23 of
           these and every one sits on a section boundary. This is the single
           highest-leverage stylistic tic in that video.

  prism  - a 4-6 frame zoom with RGB channel separation and directional blur,
           verified on a native-resolution frame at 00:44.40 where the red and
           blue channels are visibly offset.

Usage:
  python3 transitions.py flash  A.mp4 B.mp4 out.mp4 [--frames 2] [--color white]
  python3 transitions.py prism  A.mp4 B.mp4 out.mp4 [--frames 6] [--zoom 0.40]
                                                    [--shift 16] [--blur 6]

The prism ramp is applied frame by frame (it is only a handful of frames), so
the zoom, channel offset and blur all escalate smoothly instead of sitting at a
constant value the way a single ffmpeg filter pass would give you.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
W, H, FPS = 1920, 1080, 30


def run(args):
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{proc.stderr[-1500:]}")


def probe_duration(src):
    """Seconds, parsed from ffmpeg's own report (no ffprobe dependency)."""
    proc = subprocess.run([FFMPEG, "-hide_banner", "-i", str(src)],
                          capture_output=True, text=True)
    for line in proc.stderr.splitlines():
        if "Duration:" in line:
            clock = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = clock.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    sys.exit(f"could not read duration of {src}")


def normalise(src, dst):
    """Force a clip to the pack's resolution and frame rate so concat is safe."""
    run([
        "-i", str(src),
        "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
               f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p",
        "-c:v", "libx264", "-crf", "16", "-an", str(dst),
    ])


def concat(parts, out):
    listing = out.parent / "_concat.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    run(["-f", "concat", "-safe", "0", "-i", str(listing),
         "-c:v", "libx264", "-crf", "16", str(out)])
    listing.unlink()


def build_flash(a, b, out, frames, color):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        na, nb, solid = tmp / "a.mp4", tmp / "b.mp4", tmp / "flash.mp4"
        normalise(a, na)
        normalise(b, nb)
        hexes = {"white": "0xFFFFFF", "black": "0x000000"}
        run([
            "-f", "lavfi",
            "-i", f"color=c={hexes.get(color, color)}:s={W}x{H}:r={FPS}:d={frames / FPS:.4f}",
            "-frames:v", str(frames),
            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(solid),
        ])
        concat([na, nb], out) if frames <= 0 else concat([na, solid, nb], out)


def ramp_frames(src_dir, dst_dir, count, zoom, shift, blur, reverse):
    """Apply an escalating zoom / channel-split / blur across a run of frames."""
    for i in range(count):
        # p goes 0 -> 1 across the outgoing tail, and 1 -> 0 across the incoming head.
        p = (i + 1) / count
        if reverse:
            p = 1 - p
        z = 1 + zoom * p
        px = max(0, int(round(shift * p)))
        sigma = max(0.01, blur * p)
        chain = (
            f"crop=iw/{z:.5f}:ih/{z:.5f}:(iw-iw/{z:.5f})/2:(ih-ih/{z:.5f})/2,"
            f"scale={W}:{H},"
            f"rgbashift=rh=-{px}:bh={px},"
            f"gblur=sigma={sigma:.3f}"
        )
        run(["-i", str(src_dir / f"f{i:04d}.png"), "-vf", chain,
             str(dst_dir / f"g{i:04d}.png")])


def build_prism(a, b, out, frames, zoom, shift, blur):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        na, nb = tmp / "a.mp4", tmp / "b.mp4"
        normalise(a, na)
        normalise(b, nb)

        tail, head = tmp / "tail", tmp / "head"
        tailfx, headfx = tmp / "tailfx", tmp / "headfx"
        for d in (tail, head, tailfx, headfx):
            d.mkdir()

        dur = frames / FPS
        # Last `frames` of A, and first `frames` of B.
        # -start_number 0 so the written names match the reader below; the image
        # muxer counts from 1 otherwise.
        run(["-sseof", f"-{dur:.4f}", "-i", str(na), "-frames:v", str(frames),
             "-start_number", "0", str(tail / "f%04d.png")])
        run(["-i", str(nb), "-frames:v", str(frames),
             "-start_number", "0", str(head / "f%04d.png")])

        ramp_frames(tail, tailfx, frames, zoom, shift, blur, reverse=False)
        ramp_frames(head, headfx, frames, zoom, shift, blur, reverse=True)

        # Rebuild: A minus its tail, the two ramped runs, then B minus its head.
        a_body, b_body = tmp / "abody.mp4", tmp / "bbody.mp4"
        keep = max(0.0, probe_duration(na) - dur)
        run(["-i", str(na), "-t", f"{keep:.4f}",
             "-c:v", "libx264", "-crf", "16", str(a_body)])
        run(["-ss", f"{dur:.4f}", "-i", str(nb),
             "-c:v", "libx264", "-crf", "16", str(b_body)])

        t_out, h_out = tmp / "tail.mp4", tmp / "head.mp4"
        run(["-framerate", str(FPS), "-i", str(tailfx / "g%04d.png"),
             "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(t_out)])
        run(["-framerate", str(FPS), "-i", str(headfx / "g%04d.png"),
             "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(h_out)])

        concat([a_body, t_out, h_out, b_body], out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)

    f = sub.add_parser("flash", help="1-2 frame solid at the cut")
    f.add_argument("a"); f.add_argument("b"); f.add_argument("out")
    f.add_argument("--frames", type=int, default=2)
    f.add_argument("--color", default="white")

    p = sub.add_parser("prism", help="zoom + chromatic aberration + blur")
    p.add_argument("a"); p.add_argument("b"); p.add_argument("out")
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--zoom", type=float, default=0.40)
    p.add_argument("--shift", type=int, default=16)
    p.add_argument("--blur", type=float, default=6.0)

    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.kind == "flash":
        build_flash(Path(args.a), Path(args.b), out, args.frames, args.color)
    else:
        build_prism(Path(args.a), Path(args.b), out,
                    args.frames, args.zoom, args.shift, args.blur)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
