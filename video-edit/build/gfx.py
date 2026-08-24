"""Drawing helpers: easing, letterspaced text, cards, gradients."""
import math
from PIL import Image, ImageDraw, ImageFont

FONTS_DIR = None  # set by caller

def load(name, size):
    return ImageFont.truetype("%s/%s" % (FONTS_DIR, name), size)

# ---------- easing ----------
def clamp01(x):        return 0.0 if x < 0 else (1.0 if x > 1 else x)
def ease_out_cubic(u): u = clamp01(u); return 1 - (1 - u) ** 3
def ease_out_quint(u): u = clamp01(u); return 1 - (1 - u) ** 5
def ease_in_cubic(u):  u = clamp01(u); return u ** 3
def ease_out_back(u, s=1.70158):
    u = clamp01(u); return 1 + (s + 1) * (u - 1) ** 3 + s * (u - 1) ** 2
def ease_in_out(u):
    u = clamp01(u); return 3 * u * u - 2 * u ** 3

def env(t, t_in, t_out, d_in=0.40, d_out=0.30):
    """Returns (progress_in 0..1, alpha 0..1). alpha==0 means don't draw."""
    if t < t_in - 0.001 or t > t_out + d_out:
        return 0.0, 0.0
    pin = ease_out_cubic((t - t_in) / d_in)
    if t <= t_out:
        return pin, min(1.0, (t - t_in) / max(d_in * 0.75, .001))
    return 1.0, 1.0 - ease_in_cubic((t - t_out) / d_out)

# ---------- text ----------
def ls_width(draw, text, font, ls):
    w = 0
    for ch in text:
        w += draw.textlength(ch, font=font) + ls
    return max(0, w - ls)

def ls_text(draw, xy, text, font, fill, ls=0, anchor="la", shadow=None):
    """Letterspaced text. anchor: 'la' left-ascender, 'ma' centered, 'ra' right."""
    x, y = xy
    total = ls_width(draw, text, font, ls)
    if anchor[0] == "m": x -= total / 2
    elif anchor[0] == "r": x -= total
    if shadow:
        sc, sox, soy = shadow
        cx = x
        for ch in text:
            draw.text((cx + sox, y + soy), ch, font=font, fill=sc, anchor="l" + anchor[1])
            cx += draw.textlength(ch, font=font) + ls
    cx = x
    for ch in text:
        draw.text((cx, y), ch, font=font, fill=fill, anchor="l" + anchor[1])
        cx += draw.textlength(ch, font=font) + ls
    return total

def text_sh(draw, xy, text, font, fill, anchor="la", shadow=(0, 0, 0, 150), off=(0, 4)):
    draw.text((xy[0] + off[0], xy[1] + off[1]), text, font=font, fill=shadow, anchor=anchor)
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)

# ---------- shapes ----------
def rrect(draw, box, r, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)

def vgrad(size, top_rgba, bot_rgba):
    w, h = size
    img = Image.new("RGBA", (1, h))
    px = img.load()
    for y in range(h):
        u = y / max(h - 1, 1)
        px[0, y] = tuple(int(top_rgba[i] + (bot_rgba[i] - top_rgba[i]) * u) for i in range(4))
    return img.resize((w, h))

def fmt_int(n):
    return "{:,}".format(int(round(n)))
