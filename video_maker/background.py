"""Procedurally painted city-street backdrop.

Rendered once, larger than the final frame, so the video can pan and zoom into
it (Ken Burns) without ever reaching the edge of the canvas.
"""
import math
import random

from PIL import Image, ImageChops, ImageDraw, ImageFilter

# Painted larger than 1080x1920 so a 16% zoom still has pixels to spare.
CANVAS = (1500, 2600)
HORIZON_T = 0.56


def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _sky(size, top, mid, horizon, horizon_y):
    """Vertical three-stop gradient, warmest right at the horizon."""
    w, h = size
    strip = Image.new("RGB", (1, h))
    px = strip.load()
    for y in range(h):
        t = min(1.0, y / horizon_y)
        px[0, y] = _lerp(top, mid, t / 0.6) if t < 0.6 else _lerp(mid, horizon, ((t - 0.6) / 0.4) ** 1.6)
    return strip.resize((w, h), Image.BILINEAR)


def _skyline(draw, width, y_base, height, colour, seed, lit=0.0, window=(255, 176, 92)):
    """One depth layer of blocky buildings standing on `y_base`."""
    rng = random.Random(seed)
    x = -rng.randint(0, 80)
    while x < width + 60:
        bw = rng.randint(70, 190)
        bh = int(height * rng.uniform(0.45, 1.0))
        top = y_base - bh
        draw.rectangle([x, top, x + bw, y_base], fill=colour)

        if rng.random() < 0.3:  # roof detail: a pitched top or a chimney
            cx = x + bw // 2
            if rng.random() < 0.5:
                draw.polygon([(x, top), (x + bw, top), (cx, top - rng.randint(30, 70))], fill=colour)
            else:
                draw.rectangle([cx - 8, top - rng.randint(20, 50), cx + 8, top], fill=colour)

        if lit and bw > 80:
            for wy in range(top + 26, y_base - 30, 46):
                for wx in range(x + 18, x + bw - 24, 34):
                    if rng.random() < lit:
                        glow = window if rng.random() < 0.75 else (255, 226, 168)
                        draw.rectangle([wx, wy, wx + 12, wy + 18], fill=glow)
        x += bw + rng.randint(4, 22)


def _clock_tower(draw, cx, y_base, height, colour, face):
    """The landmark silhouette that anchors the composition."""
    w = height * 0.135
    top = y_base - height
    draw.rectangle([cx - w / 2, top + height * 0.30, cx + w / 2, y_base], fill=colour)
    draw.rectangle([cx - w * 0.62, top + height * 0.22, cx + w * 0.62, top + height * 0.32], fill=colour)
    draw.polygon(
        [(cx - w * 0.55, top + height * 0.22), (cx + w * 0.55, top + height * 0.22), (cx, top)],
        fill=colour,
    )
    fr = w * 0.36
    fy = top + height * 0.42
    draw.ellipse([cx - fr, fy - fr, cx + fr, fy + fr], fill=face, outline=colour, width=5)
    draw.line([cx, fy, cx, fy - fr * 0.62], fill=colour, width=5)
    draw.line([cx, fy, cx + fr * 0.48, fy + fr * 0.18], fill=colour, width=5)
    for i in range(5):
        yy = top + height * (0.55 + i * 0.075)
        draw.rectangle([cx - w * 0.16, yy, cx + w * 0.16, yy + height * 0.035], fill=(255, 198, 128))


def _road(draw, size, horizon_y):
    """Wet cobbled street receding to a vanishing point on the horizon."""
    w, h = size
    draw.polygon(
        [(w * 0.02, h), (w * 0.98, h), (w * 0.58, horizon_y), (w * 0.42, horizon_y)],
        fill=(58, 40, 46),
    )
    # Cobble courses: closer together near the horizon, wider at our feet.
    for i in range(1, 30):
        t = (i / 30) ** 2.1
        y = horizon_y + (h - horizon_y) * t
        left = w * 0.42 - (w * 0.40) * t
        right = w * 0.58 + (w * 0.40) * t
        shade = int(74 + 34 * t)
        draw.line([(left, y), (right, y)], fill=(shade, shade - 14, shade - 4), width=2)
    # Reflections of the street lamps, smeared along the road's depth axis.
    for i in range(10):
        s = 0.18 + i * 0.075
        x0 = w * 0.5 + (s - 0.5) * w * 0.24
        x1 = w * 0.5 + (s - 0.5) * w * 1.05
        draw.line([(x0, horizon_y + 20), (x1, h)], fill=(126, 78, 52), width=3 + (i % 3) * 3)


def _street_wall(draw, size, horizon_y, side, seed):
    """Buildings lining one kerb, drawn far-to-near so nearer ones overlap."""
    rng = random.Random(seed)
    w, h = size
    for i in range(9):
        t = ((i + 1) / 9) ** 1.7
        # Follow the kerb line from the vanishing point out to the frame edge.
        edge_x = w * 0.5 + side * (w * 0.08 + w * 0.36 * t)
        edge_y = horizon_y + (h - horizon_y) * t
        top = edge_y - (260 + 1500 * t)
        x0, x1 = (edge_x, w + 80) if side > 0 else (-80, edge_x)
        lo, hi = min(x0, x1), max(x0, x1)
        # Facades run past the bottom edge so the road never shows through.
        shade = 60 - 42 * t
        draw.rectangle([lo, top, hi, h + 100], fill=(int(shade + 10), int(shade), int(shade + 22)))
        if i == 8:  # kerb lip, only on the nearest facade
            draw.line([(edge_x, edge_y), (edge_x, h + 100)], fill=(96, 78, 92), width=14)

        # Lit windows, sized by depth so the perspective reads correctly.
        step = int(38 + 120 * t)
        wsz = max(3, int(step * 0.30))
        for wy in range(int(top) + step, int(h), step):
            for wx in range(int(lo) + step, int(hi) - wsz, step):
                r = rng.random()
                if r < 0.30:
                    glow = (255, 186, 104) if r < 0.24 else (176, 208, 255)
                elif r < 0.55:
                    glow = (int(shade + 30), int(shade + 18), int(shade + 40))  # dark pane
                else:
                    continue
                draw.rectangle([wx, wy, wx + wsz, wy + int(wsz * 1.5)], fill=glow)

    # A handful of street lamps, near ones only, so they read as objects.
    for i in (3, 5, 7):
        t = ((i + 1) / 9) ** 1.7
        edge_x = w * 0.5 + side * (w * 0.08 + w * 0.36 * t)
        edge_y = horizon_y + (h - horizon_y) * t
        post_h = 170 + 380 * t
        draw.line([(edge_x, edge_y), (edge_x, edge_y - post_h)], fill=(20, 16, 26), width=int(4 + 9 * t))
        arm = side * -(14 + 34 * t)
        ly = edge_y - post_h
        draw.line([(edge_x, ly), (edge_x + arm, ly)], fill=(20, 16, 26), width=int(3 + 7 * t))
        lr = 9 + 24 * t
        draw.ellipse([edge_x + arm - lr, ly - lr * 0.7, edge_x + arm + lr, ly + lr * 1.1], fill=(255, 216, 150))


def _bokeh(size, horizon_y, seed):
    """Soft out-of-focus lights, added on top rather than blended in."""
    rng = random.Random(seed)
    layer = Image.new("RGB", size, (0, 0, 0))
    d = ImageDraw.Draw(layer)
    for _ in range(55):
        x = rng.randint(0, size[0])
        y = rng.randint(int(horizon_y * 0.7), int(size[1] * 0.92))
        r = rng.randint(5, 18)
        k = rng.uniform(0.25, 0.7)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(int(255 * k), int(190 * k), int(112 * k)))
    return layer.filter(ImageFilter.GaussianBlur(11))


def _vignette(img, strength=0.72):
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    inset = int(min(w, h) * 0.22)
    ImageDraw.Draw(mask).ellipse([-inset, -inset, w + inset, h + inset], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(min(w, h) * 0.14))
    mask = mask.point(lambda v: int(255 - (255 - v) * strength))
    return Image.composite(img, Image.new("RGB", (w, h), (10, 8, 20)), mask)


def paint(seed=7):
    """Return the full-size backdrop image."""
    w, h = CANVAS
    horizon = int(h * HORIZON_T)
    rng = random.Random(seed)

    img = _sky(CANVAS, top=(20, 24, 62), mid=(86, 60, 108), horizon=(232, 142, 92), horizon_y=horizon)
    d = ImageDraw.Draw(img)

    for _ in range(180):  # stars, only where the sky is still dark
        sx, sy = rng.randint(0, w), rng.randint(0, int(h * 0.30))
        b = rng.randint(160, 240)
        d.point((sx, sy), fill=(b, b, b))

    mx, my, mr = w * 0.76, h * 0.11, 44
    d.ellipse([mx - mr, my - mr, mx + mr, my + mr], fill=(240, 236, 250))
    img = img.filter(ImageFilter.GaussianBlur(1.4))
    d = ImageDraw.Draw(img)

    # Far to near: haze layer, landmark, mid skyline, road, then the street walls.
    _skyline(d, w, horizon - 24, 420, (108, 76, 116), seed + 1)
    _clock_tower(d, w * 0.31, horizon - 6, 880, (52, 36, 68), (214, 180, 138))
    _skyline(d, w, horizon + 6, 620, (52, 36, 68), seed + 2, lit=0.26)
    _road(d, CANVAS, horizon + 6)
    _street_wall(d, CANVAS, horizon + 6, -1, seed + 3)
    _street_wall(d, CANVAS, horizon + 6, +1, seed + 4)

    img = Image.blend(img, img.filter(ImageFilter.GaussianBlur(2.2)), 0.22)
    img = ImageChops.add(img, _bokeh(CANVAS, horizon, seed + 5))
    return _vignette(img)


def ken_burns(base, frame_size, t):
    """Crop `base` for progress `t` in [0,1]: a slow zoom-in with a gentle drift."""
    bw, bh = base.size
    fw, fh = frame_size
    zoom = 1.0 + 0.16 * t
    scale = min(bw / fw, bh / fh) / zoom  # widest crop matching the frame aspect
    cw, ch = fw * scale, fh * scale
    cx = bw / 2 + (bw - cw) * 0.18 * math.sin(t * math.pi * 0.5)
    cy = bh / 2 - (bh - ch) * 0.22 * t
    return base.resize(frame_size, Image.LANCZOS, box=(cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2))


if __name__ == "__main__":
    paint().save("out/_bg_preview.jpg", quality=90)
