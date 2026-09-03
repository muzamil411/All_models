"""Build a vertical "<language> in seconds" short from a content file.

Everything is generated locally: the backdrop and the avatar are painted with
Pillow, the music bed is synthesised with numpy, and ffmpeg only muxes the
result. Swap the JSON in content/ to make a video for another language.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import audio
import background
import character

W, H = 1080, 1920
FPS = 30
FONT_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

TEXT_X = 62
BLOCK_TOP = 606
PAIR_GAP = 214
KNOWN_SIZE = 80
TAUGHT_SIZE = 66
KNOWN_COLOUR = (255, 205, 60)
TAUGHT_COLOUR = (255, 255, 255)

BADGE_W, BADGE_H = 300, 200
BADGE_TOP = 210
BAR_W, BAR_H = 300, 18
BAR_TOP = BADGE_TOP + BADGE_H + 26


def ease_out(x):
    return 1 - (1 - max(0.0, min(1.0, x))) ** 3


# --------------------------------------------------------------------------- text

def text_layer(text, size, fill, glow=(0, 0, 0), glow_radius=14, glow_alpha=190):
    """Render one line of text with a soft shadow so it stays legible on any frame."""
    font = ImageFont.truetype(FONT_BOLD, size)
    pad = glow_radius * 3
    box = font.getbbox(text)
    w, h = box[2] - box[0] + pad * 2, box[3] - box[1] + pad * 2
    origin = (pad - box[0], pad - box[1])

    shadow = Image.new("L", (w, h), 0)
    ImageDraw.Draw(shadow).text(origin, text, font=font, fill=glow_alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(glow_radius))

    layer = Image.new("RGBA", (w, h), glow + (0,))
    layer.putalpha(shadow)
    ImageDraw.Draw(layer).text(origin, text, font=font, fill=fill + (255,))
    return layer


def paste_faded(canvas, layer, xy, alpha):
    """Alpha-composite `layer` onto `canvas`, scaled by `alpha` in [0,1]."""
    if alpha <= 0.004:
        return
    if alpha < 0.996:
        layer = layer.copy()
        layer.putalpha(layer.getchannel("A").point(lambda v: int(v * alpha)))
    canvas.alpha_composite(layer, (int(xy[0]), int(xy[1])))


# --------------------------------------------------------------------------- chrome

def union_jack(size):
    """A simplified Union Flag - close enough to read correctly at badge size."""
    ss = 4
    w, h = size[0] * ss, size[1] * ss
    img = Image.new("RGB", (w, h), (1, 33, 105))
    d = ImageDraw.Draw(img)
    for a, b in (((0, 0), (w, h)), ((w, 0), (0, h))):
        d.line([a, b], fill=(255, 255, 255), width=int(h * 0.30))
    for a, b in (((0, 0), (w, h)), ((w, 0), (0, h))):
        d.line([a, b], fill=(200, 16, 46), width=int(h * 0.12))
    d.rectangle([w / 2 - h * 0.17, 0, w / 2 + h * 0.17, h], fill=(255, 255, 255))
    d.rectangle([0, h / 2 - h * 0.17, w, h / 2 + h * 0.17], fill=(255, 255, 255))
    d.rectangle([w / 2 - h * 0.10, 0, w / 2 + h * 0.10, h], fill=(200, 16, 46))
    d.rectangle([0, h / 2 - h * 0.10, w, h / 2 + h * 0.10], fill=(200, 16, 46))
    return img.resize(size, Image.LANCZOS)


def badge(size, radius=34):
    """The flag on a rounded card with a drop shadow."""
    pad = 30
    canvas = Image.new("RGBA", (size[0] + pad * 2, size[1] + pad * 2), (0, 0, 0, 0))

    shadow = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(shadow).rounded_rectangle(
        [pad, pad + 10, pad + size[0], pad + size[1] + 14], radius=radius, fill=170
    )
    canvas.putalpha(shadow.filter(ImageFilter.GaussianBlur(16)))

    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=255)
    flag = union_jack(size)
    flag.putalpha(mask)
    canvas.alpha_composite(flag, (pad, pad))

    ring = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(ring).rounded_rectangle(
        [pad, pad, pad + size[0] - 1, pad + size[1] - 1], radius=radius,
        outline=(255, 255, 255, 210), width=5,
    )
    canvas.alpha_composite(ring)
    return canvas, pad


def progress_bar(fraction):
    layer = Image.new("RGBA", (BAR_W, BAR_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    r = BAR_H // 2
    d.rounded_rectangle([0, 0, BAR_W - 1, BAR_H - 1], radius=r, fill=(255, 255, 255, 70))
    fill_w = max(BAR_H, int(BAR_W * max(0.0, min(1.0, fraction))))
    d.rounded_rectangle([0, 0, fill_w, BAR_H - 1], radius=r, fill=(255, 255, 255, 240))
    return layer


def scrim():
    """Darken the left column and the base of the frame so text always reads."""
    x = np.arange(W)[None, :]
    y = np.arange(H)[:, None]
    side = np.clip(1.0 - x / (W * 0.82), 0, None) ** 1.5
    bottom = np.clip((y - H * 0.42) / (H * 0.58), 0, None) ** 1.6
    alpha = np.clip(200 * side + 120 * bottom, 0, 215).astype("uint8")

    rgba = np.zeros((H, W, 4), dtype="uint8")
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 6, 4, 14
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, "RGBA")


# --------------------------------------------------------------------------- build

def load_content(path):
    data = json.loads(Path(path).read_text())
    pairs = data["pairs"]
    reveal_at = data.get("first_reveal", 5.4)
    gap = data.get("reveal_gap", 3.6)
    data["reveals"] = [reveal_at + i * gap for i in range(len(pairs))]
    data["duration"] = data.get("duration", data["reveals"][-1] + gap)
    data["voice_track"] = voice_track(data)
    return data


def voice_track(data):
    """Schedule the voiceover clips: the intro on its own beat, then one clip
    per reveal, nudged slightly late so the word lands before it is spoken."""
    spec = data.get("voice")
    if not spec:
        return []

    here = Path(__file__).parent
    track = []
    if spec.get("intro"):
        track.append({"file": here / spec["intro"]["file"], "at": spec["intro"]["at"]})

    offset = spec.get("reveal_offset", 0.12)
    for reveal, name in zip(data["reveals"], spec.get("files", [])):
        track.append({"file": here / name, "at": reveal + offset})

    missing = [str(c["file"]) for c in track if not c["file"].exists()]
    if missing:
        raise SystemExit("voiceover clips not found:\n  " + "\n  ".join(missing))
    return track


def render(content, out_path, preview_only=None):
    duration = content["duration"]
    total = int(duration * FPS)
    reveals = content["reveals"]

    print(f"painting backdrop ...", flush=True)
    base_bg = background.paint(seed=content.get("seed", 7))
    veil = scrim()
    badge_img, badge_pad = badge((BADGE_W, BADGE_H))
    badge_xy = ((W - BADGE_W) // 2 - badge_pad, BADGE_TOP - badge_pad)

    print("drawing character poses ...", flush=True)
    character.loop_frames()
    char_w = 520
    char_h = int(char_w * character.SIZE[1] / character.SIZE[0])
    char_xy = (W - char_w - 44, H - char_h + 26)

    print("laying out text ...", flush=True)
    known = [text_layer(p[0], KNOWN_SIZE, KNOWN_COLOUR) for p in content["pairs"]]
    taught = [text_layer(p[1], TAUGHT_SIZE, TAUGHT_COLOUR) for p in content["pairs"]]

    frames = range(total) if preview_only is None else preview_only
    if preview_only is None:
        proc = subprocess.Popen(
            [FFMPEG, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
             "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", str(out_path)],
            stdin=subprocess.PIPE,
        )
    else:
        proc = None

    for i in frames:
        t = i / FPS
        canvas = background.ken_burns(base_bg, (W, H), t / duration).convert("RGBA")
        canvas.alpha_composite(veil)

        # Badge, and the loading bar that fills over the opening beat.
        paste_faded(canvas, badge_img, badge_xy, ease_out(0.35 + t / 0.3))
        bar_life = content.get("bar_seconds", 4.2)
        if t < bar_life:
            fade = min(1.0, (bar_life - t) / 0.5)
            paste_faded(canvas, progress_bar(t / bar_life), ((W - BAR_W) // 2, BAR_TOP), fade)

        # Word list: the known-language column slides in during the first beat.
        for idx, layer in enumerate(known):
            # Staggered but front-loaded, so frame 0 already works as a cover.
            k = ease_out((0.30 + t - idx * 0.06) / 0.40)
            y = BLOCK_TOP + idx * PAIR_GAP - 30 + 30 * k
            paste_faded(canvas, layer, (TEXT_X - 26, y - 26), k)

        # Each translation fades and slides up on its own beat.
        for idx, layer in enumerate(taught):
            k = ease_out((t - reveals[idx]) / 0.5)
            y = BLOCK_TOP + idx * PAIR_GAP + 92 + 26 * (1 - k)
            paste_faded(canvas, layer, (TEXT_X + 12, y - 26), k)

        char = character.frame(i).resize((char_w, char_h), Image.LANCZOS)
        paste_faded(canvas, char, char_xy, ease_out(0.4 + t / 0.5))

        if t < 0.4:  # lift out of black, but never to a fully black first frame
            k = 0.58 + 0.42 * ease_out(t / 0.4)
            canvas = Image.blend(Image.new("RGBA", (W, H), (0, 0, 0, 255)), canvas, k)

        rgb = canvas.convert("RGB")
        if proc:
            proc.stdin.write(rgb.tobytes())
            if i % 60 == 0:
                print(f"  frame {i}/{total}", flush=True)
        else:
            rgb.save(f"out/_preview_{i:04d}.jpg", quality=90)

    if proc:
        proc.stdin.close()
        proc.wait()
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", default="content/english_in_seconds.json")
    ap.add_argument("--out", default="out/english_in_seconds.mp4")
    ap.add_argument("--preview", help="comma-separated frame numbers; writes stills instead of video")
    args = ap.parse_args()

    content = load_content(args.content)
    Path("out").mkdir(exist_ok=True)

    if args.preview:
        render(content, None, [int(x) for x in args.preview.split(",")])
        return

    silent = Path("out/_silent.mp4")
    render(content, silent)

    track = content["voice_track"]
    print(f"mixing audio ({len(track)} voice clips) ...", flush=True)
    wav = audio.build(content["duration"], content["reveals"], "out/_music.wav", voice=track)

    print("muxing ...", flush=True)
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", str(silent), "-i", wav,
         "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", args.out],
        check=True,
    )
    silent.unlink()
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
