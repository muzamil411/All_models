#!/usr/bin/env python3
"""
Prepare the static visual assets for the German Daily Phrases short.

Inputs  (assets/source/): char_raw.png, bg_16x9_a.png  -- raw ElevenLabs renders
Outputs (assets/): character/teacher.png         -- RGBA cut-out, tight bbox
                   background/plate.png          -- 1296x2304 portrait plate (zoom headroom)
                   background/flag.png           -- German flag badge, drawn from scratch
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
SOURCE = os.path.join(ASSETS, "source")

PLATE_W, PLATE_H = 1296, 2304  # 9:16, 1.2x of 1080x1920 -> headroom for the slow push-in


# --------------------------------------------------------------------------- character
def cut_out_character(src_path, dst_path):
    """Key the flat white studio background using a border flood fill.

    A plain luminance key would punch holes in the white t-shirt and sneakers,
    so instead we flood from the image border: interior whites are sealed off
    by the illustration's bold black outlines and survive.
    """
    im = Image.open(src_path).convert("RGB")
    w, h = im.size

    # Pad by 2px so a figure touching an edge still has an outside to flood from.
    pad = 2
    work = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (255, 255, 255))
    work.paste(im, (pad, pad))

    MARK = (255, 0, 255)
    for xy in [(0, 0), (work.width - 1, 0), (0, work.height - 1),
               (work.width - 1, work.height - 1), (work.width // 2, 0),
               (work.width // 2, work.height - 1)]:
        ImageDraw.floodfill(work, xy, MARK, thresh=42)

    a = np.array(work)
    background = (a[:, :, 0] == 255) & (a[:, :, 1] == 0) & (a[:, :, 2] == 255)
    mask = np.where(background, 0, 255).astype(np.uint8)
    mask = Image.fromarray(mask[pad:pad + h, pad:pad + w], "L")

    # Erode 1px to swallow the white fringe left on the outline, then soften
    # the edge back so it composites without stair-stepping.
    mask = mask.filter(ImageFilter.MinFilter(3))
    mask = mask.filter(ImageFilter.GaussianBlur(0.7))

    out = im.convert("RGBA")
    out.putalpha(mask)
    bbox = out.getbbox()
    out = out.crop(bbox)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    out.save(dst_path)
    print(f"character: {im.size} -> bbox {bbox} -> {out.size}")
    return out


# -------------------------------------------------------------------------- background
def build_plate(src_path, dst_path):
    """Turn the 16:9 render into a 9:16 plate by extending its blue-hour sky upward."""
    src = Image.open(src_path).convert("RGB")

    # Drop the right-hand block whose rooftops run all the way to the top edge,
    # so the seam with the synthesised sky lands in clean sky across full width.
    crop = src.crop((110, 0, 1740, src.height))
    scaled_h = round(crop.height * PLATE_W / crop.width)
    scaled = crop.resize((PLATE_W, scaled_h), Image.LANCZOS)
    y0 = PLATE_H - scaled_h
    feather = 300

    # Grow the sky from a patch left of the church spire (nothing but cloud in
    # there), stretched and blurred into a soft high-altitude gradient. Blurring
    # is what keeps the stretch from reading as streaks.
    sky = src.crop((0, 0, 1350, 500)).resize((PLATE_W, y0 + feather), Image.LANCZOS)
    ext = np.array(sky.filter(ImageFilter.GaussianBlur(26)), float)

    # Match the extension's brightness to the photograph across the seam band,
    # so the feather has nothing to hide.
    band_top = np.array(scaled.crop((0, 0, PLATE_W, feather)), float)
    scale = (band_top.mean(axis=(0, 1)) + 1e-6) / (ext[-feather:].mean(axis=(0, 1)) + 1e-6)
    ext = ext * scale

    # Deepen toward the top of frame, easing back to 1.0 by the seam.
    g = np.clip(np.linspace(0.40, 1.12, y0 + feather), 0, 1.0)[:, None, None]
    ext = np.clip(ext * g, 0, 255)

    plate = Image.new("RGB", (PLATE_W, PLATE_H))
    plate.paste(Image.fromarray(ext.astype(np.uint8)), (0, 0))

    # Feather the photograph in over the seam.
    ramp = np.concatenate([np.linspace(0.0, 1.0, feather),
                           np.ones(scaled_h - feather)])[:, None, None]
    under = np.array(plate.crop((0, y0, PLATE_W, PLATE_H)), float)
    merged = np.array(scaled, float) * ramp + under * (1.0 - ramp)
    plate.paste(Image.fromarray(merged.astype(np.uint8)), (0, y0))

    # Grade: slightly cooler and darker so yellow/white type sits cleanly on top,
    # plus a vignette that pulls the eye to the centre column.
    plate = ImageEnhance.Color(plate).enhance(0.92)
    plate = ImageEnhance.Brightness(plate).enhance(0.86)

    yy, xx = np.mgrid[0:PLATE_H, 0:PLATE_W]
    nx = (xx - PLATE_W / 2) / (PLATE_W / 2)
    ny = (yy - PLATE_H / 2) / (PLATE_H / 2)
    r = np.sqrt(nx ** 2 + (ny * 0.72) ** 2)
    vig = np.clip(1.0 - 0.52 * np.clip((r - 0.42) / 0.95, 0, 1) ** 1.6, 0, 1)[:, :, None]
    plate = Image.fromarray(np.clip(np.array(plate, float) * vig, 0, 255).astype(np.uint8))

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    plate.save(dst_path)
    print(f"plate: {src.size} -> {plate.size} (photo seam at y={y0})")
    return plate


# -------------------------------------------------------------------------------- flag
def draw_flag(dst_path, w=300, h=200, radius=30):
    """German tricolour badge, drawn here rather than sourced, at 3x for clean downscale."""
    s = 3
    W, H, R = w * s, h * s, radius * s
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    bands = [(0, 0, 0), (221, 0, 0), (255, 206, 0)]  # black / red / gold
    band_h = H / 3
    stripes = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stripes)
    for i, c in enumerate(bands):
        sd.rectangle([0, round(i * band_h), W, round((i + 1) * band_h)], fill=c + (255,))

    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W - 1, H - 1], radius=R, fill=255)
    img.paste(stripes, (0, 0), mask)

    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=R, outline=(18, 18, 22, 255), width=5 * s)

    img = img.resize((w, h), Image.LANCZOS)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    img.save(dst_path)
    print(f"flag: {img.size}")
    return img


if __name__ == "__main__":
    cut_out_character(os.path.join(SOURCE, "char_raw.png"),
                      os.path.join(ASSETS, "character", "teacher.png"))
    build_plate(os.path.join(SOURCE, "bg_16x9_a.png"),
                os.path.join(ASSETS, "background", "plate.png"))
    draw_flag(os.path.join(ASSETS, "background", "flag.png"))
    print("assets ready")
