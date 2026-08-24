#!/usr/bin/env python3
"""Renders the animated graphics layer as raw RGBA frames on stdout."""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gfx
gfx.FONTS_DIR = os.path.join(os.path.dirname(HERE), "fonts")
from gfx import *
from timeline import *
from PIL import Image, ImageDraw

# ---------------------------------------------------------------- palette
INK   = (10, 15, 24)
GOLD  = (255, 199, 44)
RED   = (233, 58, 62)
GRN   = (34, 197, 122)
WHT   = (255, 255, 255)
def a(c, al): return (c[0], c[1], c[2], int(al * 255))

# ---------------------------------------------------------------- fonts
F_HERO   = load("anton.ttf", 232)
F_BIG    = load("anton.ttf", 132)
F_MID    = load("anton.ttf", 96)
F_UNIT   = load("montserrat900.ttf", 50)
F_SUB    = load("montserrat700.ttf", 38)
F_LBL    = load("inter600.ttf", 32)
F_CHIP   = load("inter600.ttf", 28)
F_CHIPNO = load("anton.ttf", 34)
F_TINY   = load("inter600.ttf", 24)

# ---------------------------------------------------------------- scrims
SCRIM_TOP = vgrad((W, 420), (0, 0, 0, 120), (0, 0, 0, 0))
SCRIM_BOT = vgrad((W, 560), (0, 0, 0, 0), (0, 0, 0, 135))

MARGIN = 62

# ================================================================ elements

def draw_chip(d, t):
    """Top-left chapter chip, one per section."""
    for (t0, t1, no, lab, _s) in SECTIONS:
        if no is None or not (t0 - 0.05 <= t <= t1 + 0.30):
            continue
        p, al = env(t, t0 + 0.10, t1 - 0.15, 0.38, 0.28)
        if al <= 0.01:
            continue
        x = MARGIN - 46 * (1 - p)
        y = 196
        wno = d.textlength(no, font=F_CHIPNO)
        wlb = ls_width(d, lab, F_CHIP, 5)
        wid = 30 + wno + 20 + 8 + 20 + wlb + 30
        rrect(d, (x, y, x + wid, y + 66), 33, fill=a(INK, 0.62 * al))
        rrect(d, (x, y, x + wid, y + 66), 33, outline=a(WHT, 0.13 * al), width=2)
        d.text((x + 30, y + 33), no, font=F_CHIPNO, fill=a(GOLD, al), anchor="lm")
        d.ellipse((x + 30 + wno + 18, y + 30, x + 30 + wno + 24, y + 36), fill=a(WHT, 0.5 * al))
        ls_text(d, (x + 30 + wno + 46, y + 33), lab, F_CHIP, a(WHT, 0.92 * al), ls=5, anchor="lm")
        break

def draw_hero(d, t):
    """Opening title card."""
    t0, t1 = 0.35, 4.05
    p, al = env(t, t0, t1, 0.55, 0.35)
    if al <= 0.01:
        return
    cx, y = W // 2, 470
    # reveal mask via progressive character alpha
    word = "ITALY"
    n = len(word)
    total = ls_width(d, word, F_HERO, 14)
    x = cx - total / 2
    for i, ch in enumerate(word):
        cp = ease_out_quint(clamp01((p - i * 0.055) / 0.5))
        dy = 70 * (1 - cp)
        ca = al * cp
        d.text((x + 5, y + dy + 8), ch, font=F_HERO, fill=a((0, 0, 0), 0.45 * ca), anchor="ls")
        d.text((x, y + dy), ch, font=F_HERO, fill=a(WHT, ca), anchor="ls")
        x += d.textlength(ch, font=F_HERO) + 14
    # gold rule wipe
    rw = 470 * ease_out_cubic((p - 0.30) / 0.6)
    if rw > 2:
        d.rounded_rectangle((cx - rw / 2, y + 40, cx + rw / 2, y + 49), 5, fill=a(GOLD, al))
    # subtitle
    sp = ease_out_cubic((p - 0.45) / 0.5)
    if sp > 0.01:
        ls_text(d, (cx, y + 108 - 18 * (1 - sp)), "SHAPED LIKE A BOOT", F_SUB,
                a(WHT, 0.88 * al * sp), ls=9, anchor="ma",
                shadow=(a((0, 0, 0), 0.45 * al * sp), 3, 4))

def stat_card(d, t, spec):
    """Bottom stat card with count-up value."""
    t0, t1 = spec["t0"], spec["t1"]
    p, al = env(t, t0, t1, 0.42, 0.30)
    if al <= 0.01:
        return
    has_sub = bool(spec.get("sub"))
    ch  = 348 if has_sub else 258
    top = 1312 - (58 if has_sub else 0)
    slide = 68 * (1 - ease_out_cubic(p))
    top += slide
    x0, x1 = MARGIN, W - MARGIN
    rrect(d, (x0, top, x1, top + ch), 30, fill=a(INK, 0.80 * al))
    rrect(d, (x0, top, x1, top + ch), 30, outline=a(WHT, 0.14 * al), width=2)
    # accent bar grows vertically
    bp = ease_out_cubic((p - 0.10) / 0.55)
    if bp > 0.01:
        d.rounded_rectangle((x0 + 22, top + 26, x0 + 32, top + 26 + (ch - 52) * bp), 5,
                            fill=a(spec.get("accent", GOLD), al))
    tx = x0 + 62
    ls_text(d, (tx, top + 40), spec["label"], F_LBL, a(spec.get("accent", GOLD), 0.95 * al), ls=6)
    # value (count-up)
    cu = ease_out_cubic((p - 0.08) / 0.75)
    val = spec["value"](cu) if callable(spec["value"]) else spec["value"]
    vy = top + 96
    d.text((tx + 4, vy + 6), val, font=F_BIG, fill=a((0, 0, 0), 0.45 * al), anchor="la")
    d.text((tx, vy), val, font=F_BIG, fill=a(WHT, al), anchor="la")
    if spec.get("unit"):
        uw = d.textlength(val, font=F_BIG)
        d.text((tx + uw + 22, vy + 66), spec["unit"], font=F_UNIT,
               fill=a(spec.get("accent", GOLD), al), anchor="la")
    if has_sub:
        sy = top + 272
        sw = ls_width(d, spec["sub"], F_TINY, 3)
        rrect(d, (tx - 12, sy - 8, tx + sw + 20, sy + 40), 12,
              fill=a(spec.get("subbg", GRN), 0.85 * al))
        ls_text(d, (tx + 4, sy + 16), spec["sub"], F_TINY, a(WHT, al), ls=3, anchor="lm")

def title_block(d, t, spec):
    """Big lower-left title for sections without a number."""
    t0, t1 = spec["t0"], spec["t1"]
    p, al = env(t, t0, t1, 0.42, 0.28)
    if al <= 0.01:
        return
    y = spec.get("y", 1300)
    x = MARGIN + 8
    dx = -48 * (1 - ease_out_cubic(p))
    bh = 120 * ease_out_cubic((p - 0.05) / 0.5)
    if bh > 2:
        d.rounded_rectangle((x + dx, y - 6, x + dx + 10, y - 6 + bh), 5,
                            fill=a(spec.get("accent", GOLD), al))
    kx, ky = x + dx + 34, y + 4
    kw = ls_width(d, spec["kicker"], F_TINY, 6)
    rrect(d, (kx - 16, ky - 9, kx + kw + 18, ky + 41), 13, fill=a(INK, 0.90 * al))
    rrect(d, (kx - 16, ky - 9, kx + kw + 18, ky + 41), 13, outline=a(WHT, 0.12 * al), width=2)
    ls_text(d, (kx, ky + 16), spec["kicker"], F_TINY,
            a(spec.get("accent", GOLD), al), ls=6, anchor="lm")
    tp = ease_out_cubic((p - 0.12) / 0.6)
    for ox, oy, sa in ((5, 7, 0.55), (3, 4, 0.35)):
        d.text((x + dx + 34 + ox, y + 46 + oy + 14 * (1 - tp)), spec["title"], font=F_MID,
               fill=a((0, 0, 0), sa * al * tp), anchor="la")
    d.text((x + dx + 34, y + 46 + 14 * (1 - tp)), spec["title"], font=F_MID,
           fill=a(WHT, al * tp), anchor="la")

def draw_progress(d, t):
    y = 1874
    x0, x1 = MARGIN, W - MARGIN
    d.rounded_rectangle((x0, y, x1, y + 7), 4, fill=a(WHT, 0.20))
    for (bt, _e, _n, _l, _s) in SECTIONS[1:]:
        tx = x0 + (x1 - x0) * (bt / DUR)
        d.rectangle((tx - 1, y - 4, tx + 1, y + 11), fill=a(WHT, 0.30))
    fw = (x1 - x0) * clamp01(t / DUR)
    if fw > 4:
        d.rounded_rectangle((x0, y, x0 + fw, y + 7), 4, fill=a(GOLD, 0.95))
        d.ellipse((x0 + fw - 9, y - 6, x0 + fw + 9, y + 13), fill=a(WHT, 0.95))

def draw_mark(d, t):
    al = 0.34 if t < 69.5 else 0.34 * max(0.0, 1 - (t - 69.5) / 0.8)
    if al <= 0.01:
        return
    ls_text(d, (W - MARGIN, 214), "TRUTH SEEKER", F_TINY, a(WHT, al), ls=6, anchor="ra",
            shadow=(a((0, 0, 0), al * 0.7), 2, 2))

FLASHES = [4.80, 9.50, 12.60, 16.70, 19.90, 27.50, 33.90, 41.30, 44.90, 49.70, 59.20, 66.30, 70.50]
def draw_fx(d, t):
    """Cut flashes + a gold sweep line on section changes."""
    for f in FLASHES:
        dt = t - f
        if -0.02 <= dt < 0.20:
            d.rectangle((0, 0, W, H), fill=a(WHT, 0.20 * math.exp(-dt / 0.06)))
        if 0 <= dt < 0.42:
            u = ease_out_cubic(dt / 0.42)
            sx = -180 + (W + 360) * u
            al = 0.55 * (1 - u) ** 0.7
            for k in range(9):
                d.rectangle((sx - k * 16, 0, sx - k * 16 + 8, H),
                            fill=a(GOLD, al * (1 - k / 9) * 0.55))
            d.rectangle((sx, 0, sx + 5, H), fill=a(WHT, al))

# ================================================================ content
CARDS = [
    dict(t0=9.75,  t1=12.35, label="TOTAL LAND AREA",
         value=lambda u: fmt_int(301340 * u), unit="km²", accent=GOLD),
    dict(t0=13.05, t1=16.35, label="POPULATION",
         value=lambda u: fmt_int(59 * u), unit="MILLION", accent=GOLD),
    dict(t0=39.45, t1=44.55, label="GDP  ·  NOMINAL",
         value=lambda u: "$" + fmt_int(2 * u), unit="TRILLION", accent=GRN),
    dict(t0=45.35, t1=49.35, label="AVERAGE SALARY",
         value=lambda u: "$" + fmt_int(50000 * u), unit="/ YEAR", accent=GRN,
         sub="≈  1.40 CRORE PKR PER YEAR", subbg=GRN),
    dict(t0=50.30, t1=55.85, label="TIME TO CITIZENSHIP",
         value=lambda u: fmt_int(10 * u), unit="YEARS", accent=RED,
         sub="ITALIAN LANGUAGE TEST REQUIRED  ·  B1", subbg=RED),
]
TITLES = [
    dict(t0=5.30,  t1=9.10,  kicker="WHERE IT IS", title="SOUTHERN EUROPE", y=1390),
    dict(t0=17.15, t1=19.65, kicker="CAPITAL CITY", title="ROME", y=560, accent=RED),
    dict(t0=20.35, t1=23.20, kicker="27 BC  —  476 AD", title="ROMAN EMPIRE", y=1390, accent=GOLD),
    dict(t0=24.10, t1=27.20, kicker="1939  —  1945", title="WORLD WAR II", y=1390, accent=RED),
    dict(t0=28.30, t1=33.55, kicker="FULL STORY ON THE CHANNEL", title="WATCH NEXT", y=1390, accent=GOLD),
    dict(t0=34.40, t1=39.05, kicker="8TH LARGEST IN THE WORLD", title="THE ECONOMY", y=1390, accent=GRN),
    dict(t0=56.35, t1=58.85, kicker="THEN YOU CAN APPLY", title="ITALIAN PASSPORT", y=1390, accent=RED),
    dict(t0=59.90, t1=63.05, kicker="FASHION & FINANCE CAPITAL", title="MILAN", y=500, accent=GOLD),
    dict(t0=66.80, t1=70.05, kicker="THINKING OF MOVING?", title="START HERE", y=1390, accent=GOLD),
]

def draw_cta(d, t):
    t0, t1 = 70.85, 74.20
    p, al = env(t, t0, t1, 0.45, 0.40)
    if al <= 0.01:
        return
    cx = W // 2
    y = 1430 + 40 * (1 - ease_out_cubic(p))
    txt = "FOLLOW FOR PART 2"
    wid = ls_width(d, txt, F_SUB, 7) + 96
    rrect(d, (cx - wid / 2, y, cx + wid / 2, y + 96), 48, fill=a(GOLD, 0.95 * al))
    ls_text(d, (cx, y + 48), txt, F_SUB, a(INK, al), ls=7, anchor="mm")
    sp = ease_out_cubic((p - 0.35) / 0.5)
    if sp > 0.01:
        msg = "COMMENT  \"ITALY\"  BELOW"
        mw = ls_width(d, msg, F_TINY, 5)
        rrect(d, (cx - mw / 2 - 26, y + 142, cx + mw / 2 + 26, y + 196), 27,
              fill=a(INK, 0.80 * al * sp))
        ls_text(d, (cx, y + 169), msg, F_TINY, a(WHT, 0.95 * al * sp), ls=5, anchor="mm")

# ================================================================ main
def render_frame(t):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    img.alpha_composite(SCRIM_TOP, (0, 0))
    img.alpha_composite(SCRIM_BOT, (0, H - 560))
    d = ImageDraw.Draw(img)
    draw_chip(d, t)
    draw_hero(d, t)
    for c in CARDS:
        stat_card(d, t, c)
    for s in TITLES:
        title_block(d, t, s)
    draw_cta(d, t)
    draw_mark(d, t)
    draw_progress(d, t)
    draw_fx(d, t)
    return img

if __name__ == "__main__":
    if "--still" in sys.argv:
        for ts in [float(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")]:
            render_frame(ts).save("%s/still_%05.1f.png" % (sys.argv[-1], ts))
        sys.exit()
    n = int(round(DUR * FPS))
    out = sys.stdout.buffer
    for i in range(n):
        out.write(render_frame(i / FPS).tobytes())
    out.flush()
