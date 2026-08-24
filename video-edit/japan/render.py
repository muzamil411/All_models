#!/usr/bin/env python3
"""Renders the Japan explainer frame by frame (RGB24 to stdout, or stills)."""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "build"))
import gfx
gfx.FONTS_DIR = os.path.join(ROOT, "fonts")
from gfx import (load, clamp01, ease_out_cubic, ease_out_quint, ease_in_cubic,
                 ease_in_out, env, ls_text, ls_width, rrect, vgrad)
from story import *
import worldmap as wm
from PIL import Image, ImageDraw, ImageFilter

def a(c, al): return (c[0], c[1], c[2], int(max(0, min(1, al)) * 255))

F_HERO   = load("anton.ttf", 118)
F_BIG    = load("anton.ttf", 132)
F_JP     = load("notosansjp900.ttf", 250)
F_WORD   = load("anton.ttf", 82)
F_UNIT   = load("montserrat900.ttf", 50)
F_LBL    = load("inter600.ttf", 32)
F_CHIP   = load("inter600.ttf", 28)
F_CHIPNO = load("anton.ttf", 34)
F_TINY   = load("inter600.ttf", 24)
F_MICRO  = load("inter600.ttf", 21)
F_AXIS   = load("inter500.ttf", 22)
F_ANNO   = load("anton.ttf", 58)

SCRIM_TOP = vgrad((W, 460), (*INK, 150), (*INK, 0))
SCRIM_BOT = vgrad((W, 620), (*INK, 0), (*INK, 165))

# ---------------------------------------------------------------- camera
def cam_at(t):
    ks = CAM_KEYS
    lon, lat, z = ks[-1][1], ks[-1][2], ks[-1][3]
    for i in range(len(ks) - 1):
        t0, lo0, la0, z0 = ks[i]
        t1, lo1, la1, z1 = ks[i + 1]
        if t0 <= t <= t1:
            u = gfx.ease_in_out((t - t0) / max(t1 - t0, 1e-6))
            lon, lat, z = lo0 + (lo1 - lo0) * u, la0 + (la1 - la0) * u, z0 + (z1 - z0) * u
            break
    for b in BEATS:                                   # short punch on each beat
        if t >= b:
            d = t - b
            z += 0.055 * (1 - math.exp(-d / 0.05)) * math.exp(-d / 0.30)
    lon += 0.06 * math.sin(t * 0.31)
    lat += 0.04 * math.sin(t * 0.24 + 1.1)
    return wm.Camera(lon, lat, z, W, H)

# ---------------------------------------------------------------- map layer
_depth = None
def sea_depth():
    global _depth
    if _depth is None:
        m = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(m)
        d.ellipse((-W * 0.45, H * 0.10, W * 1.45, H * 0.90), fill=210)
        m = m.filter(ImageFilter.GaussianBlur(150))
        _depth = Image.merge("RGBA", (Image.new("L", (W, H), 0),) * 3 +
                             (m.point(lambda v: 112 - int(v * 0.60)),))
    return _depth

def blast_rings(d, cam, t):
    for ft, lon, lat, name in BLASTS:
        dt = t - ft
        if dt < 0 or dt > 3.4:
            continue
        x, y = cam.project(lon, lat)
        x, y = float(x), float(y)
        for k in range(3):
            u = clamp01((dt - k * 0.18) / 1.5)
            if u <= 0 or u >= 1:
                continue
            r = 26 + 420 * ease_out_cubic(u)
            al = (1 - u) ** 1.7 * 0.75
            d.ellipse((x - r, y - r, x + r, y + r), outline=a(GLOW, al), width=max(2, int(7 * (1 - u)) + 2))
        core = clamp01(1 - dt / 1.0)
        if core > 0:
            r = 9 + 26 * (1 - core)
            d.ellipse((x - r, y - r, x + r, y + r), fill=a(WASHI, core * 0.95))
        lab = clamp01((dt - 0.35) / 0.5) * clamp01((3.4 - dt) / 0.8)
        if lab > 0.02:
            ls_text(d, (x, y + 42), name, F_MICRO, a(WASHI, lab * 0.92), ls=4, anchor="ma",
                    shadow=(a(INK, lab * 0.8), 2, 2))

def tokyo_marker(d, cam, t):
    p, al = env(t, 55.4, 61.6, 0.5, 0.4)
    if al <= 0.01:
        return
    x, y = cam.project(*TOKYO)
    x, y = float(x), float(y)
    ph = (t - 55.4) % 1.6 / 1.6
    r = 16 + 54 * ease_out_cubic(ph)
    d.ellipse((x - r, y - r, x + r, y + r), outline=a(WASHI, (1 - ph) * 0.55 * al), width=3)
    d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=a(WASHI, al))
    ls_text(d, (x + 24, y - 10), "TOKYO", F_MICRO, a(WASHI, 0.95 * al), ls=4,
            shadow=(a(INK, 0.85 * al), 2, 2))

def arrows(d, cam, t):
    tx, ty = cam.project(*JP_CENTRE)
    tx, ty = float(tx), float(ty)
    for i, (lon, lat) in enumerate(ARROWS):
        ft = 62.6 + i * 0.28
        dt = t - ft
        if dt < 0:
            continue
        u = ease_out_cubic(clamp01(dt / 1.15))
        fade = clamp01((70.6 - t) / 0.8)
        if fade <= 0:
            continue
        sx, sy = cam.project(lon, lat)
        sx, sy = float(sx), float(sy)
        mx, my = (sx + tx) / 2, (sy + ty) / 2 - abs(tx - sx) * 0.22
        pts = []
        for s in range(29):
            q = (s / 28) * u
            px = (1 - q) ** 2 * sx + 2 * (1 - q) * q * mx + q * q * tx
            py = (1 - q) ** 2 * sy + 2 * (1 - q) * q * my + q * q * ty
            pts.append((px, py))
        if len(pts) > 1:
            d.line(pts, fill=a(GOLD, 0.80 * fade), width=4, joint="curve")
            hx, hy = pts[-1]
            d.ellipse((hx - 6, hy - 6, hx + 6, hy + 6), fill=a(GOLD, 0.95 * fade))

# ---------------------------------------------------------------- graphics
def chapter_chip(d, t):
    for (t0, t1, no, lab) in CHAPTERS:
        if not (t0 - 0.05 <= t <= t1 + 0.35):
            continue
        p, al = env(t, t0 + 0.12, t1 - 0.15, 0.38, 0.28)
        if al <= 0.01:
            continue
        x, y = MARGIN - 46 * (1 - p), 196
        wno = d.textlength(no, font=F_CHIPNO)
        wlb = ls_width(d, lab, F_CHIP, 5)
        wid = 30 + wno + 38 + 20 + wlb + 30
        rrect(d, (x, y, x + wid, y + 66), 33, fill=a(INK, 0.72 * al))
        rrect(d, (x, y, x + wid, y + 66), 33, outline=a(WASHI, 0.16 * al), width=2)
        d.text((x + 30, y + 33), no, font=F_CHIPNO, fill=a(GOLD, al), anchor="lm")
        d.ellipse((x + 30 + wno + 16, y + 30, x + 30 + wno + 22, y + 36), fill=a(WASHI, 0.5 * al))
        ls_text(d, (x + 30 + wno + 44, y + 33), lab, F_CHIP, a(WASHI, 0.94 * al), ls=5, anchor="lm")
        break

def headline(d, t, h):
    p, al = env(t, h["t0"], h["t1"], 0.55, 0.34)
    if al <= 0.01:
        return
    x, y = MARGIN + 6, h["y"]
    kp = ease_out_cubic(p / 0.45)
    if kp > 0.01:
        kw = ls_width(d, h["kicker"], F_TINY, 6)
        rrect(d, (x - 14, y - 46, x + kw + 18, y + 4), 13, fill=a(h["accent"], 0.92 * al * kp))
        ls_text(d, (x + 2, y - 21), h["kicker"], F_TINY, a(INK, al * kp), ls=6, anchor="lm")
    for i, line in enumerate(h["lines"]):
        lp = ease_out_quint(clamp01((p - 0.10 - i * 0.09) / 0.55))
        if lp <= 0.01:
            continue
        ly = y + 52 + i * 116 + 34 * (1 - lp)
        for ox, oy, sa in ((6, 8, 0.55), (3, 4, 0.32)):
            d.text((x + ox, ly + oy), line, font=F_HERO, fill=a((0, 0, 0), sa * al * lp), anchor="la")
        d.text((x, ly), line, font=F_HERO, fill=a(WASHI, al * lp), anchor="la")

def reveal(d, t):
    p, al = env(t, REVEAL["t0"], REVEAL["t1"], 0.62, 0.42)
    if al <= 0.01:
        return
    cx = W // 2
    jp_p = ease_out_quint(clamp01(p / 0.55))
    sc = 0.86 + 0.14 * jp_p
    f = load("notosansjp900.ttf", max(10, int(250 * sc)))
    y = 640
    d.text((cx + 8, y + 12), REVEAL["jp"], font=f, fill=a((0, 0, 0), 0.55 * al * jp_p), anchor="mm")
    d.text((cx, y), REVEAL["jp"], font=f, fill=a(WASHI, al * jp_p), anchor="mm")
    ep = ease_out_cubic(clamp01((p - 0.30) / 0.55))
    if ep > 0.01:
        rw = 300 * ep
        d.rounded_rectangle((cx - rw / 2, y + 152, cx + rw / 2, y + 161), 5, fill=a(JP_EDGE, al))
        ls_text(d, (cx, y + 232 - 18 * (1 - ep)), REVEAL["en"], F_WORD, a(WASHI, 0.95 * al * ep),
                ls=18, anchor="mm", shadow=(a((0, 0, 0), 0.5 * al), 3, 5))

def stat_card(d, t, spec):
    p, al = env(t, spec["t0"], spec["t1"], 0.44, 0.30)
    if al <= 0.01:
        return
    ch, top = 348, 1254
    top += 68 * (1 - ease_out_cubic(p))
    x0, x1 = MARGIN, W - MARGIN
    rrect(d, (x0, top, x1, top + ch), 30, fill=a(INK, 0.86 * al))
    rrect(d, (x0, top, x1, top + ch), 30, outline=a(WASHI, 0.16 * al), width=2)
    bp = ease_out_cubic((p - 0.10) / 0.55)
    if bp > 0.01:
        d.rounded_rectangle((x0 + 22, top + 26, x0 + 32, top + 26 + (ch - 52) * bp), 5,
                            fill=a(spec["accent"], al))
    tx = x0 + 62
    ls_text(d, (tx, top + 40), spec["label"], F_LBL, a(spec["accent"], 0.96 * al), ls=6)
    cu = ease_out_cubic((p - 0.08) / 0.72)
    val = spec["value"](cu)
    vy = top + 96
    d.text((tx + 4, vy + 6), val, font=F_BIG, fill=a((0, 0, 0), 0.5 * al), anchor="la")
    d.text((tx, vy), val, font=F_BIG, fill=a(WASHI, al), anchor="la")
    uw = d.textlength(val, font=F_BIG)
    d.text((tx + uw + 22, vy + 66), spec["unit"], font=F_UNIT, fill=a(spec["accent"], al), anchor="la")
    sy = top + 272
    sw = ls_width(d, spec["sub"], F_TINY, 3)
    rrect(d, (tx - 12, sy - 8, tx + sw + 20, sy + 40), 12, fill=a(spec["accent"], 0.88 * al))
    ls_text(d, (tx + 4, sy + 16), spec["sub"], F_TINY, a(INK, al), ls=3, anchor="lm")


def dashed(d, pts, color, width, dash=20, gap=14):
    """Uniform dashes measured along the polyline, not per data segment."""
    carry, on = 0.0, True
    for i in range(len(pts) - 1):
        (ax, ay), (bx, by) = pts[i], pts[i + 1]
        seg = math.hypot(bx - ax, by - ay)
        pos = 0.0
        while pos < seg:
            span = (dash if on else gap) - carry
            end = min(seg, pos + span)
            if on:
                u0, u1 = pos / seg, end / seg
                d.line([(ax + (bx - ax) * u0, ay + (by - ay) * u0),
                        (ax + (bx - ax) * u1, ay + (by - ay) * u1)], fill=color, width=width)
            if end - pos >= span:
                on, carry = not on, 0.0
            else:
                carry += end - pos
            pos = end

def pop_chart(ui, d, t):
    p, al = env(t, CHART["t0"], CHART["t1"], 0.5, 0.36)
    if al <= 0.01:
        return
    x0, y0, x1, y1 = MARGIN, 1080, W - MARGIN, 1592
    rrect(d, (x0, y0, x1, y1), 28, fill=a(INK, 0.88 * al))
    rrect(d, (x0, y0, x1, y1), 28, outline=a(WASHI, 0.15 * al), width=2)
    ls_text(d, (x0 + 46, y0 + 44), "POPULATION  ·  MILLIONS", F_TINY, a(GOLD, 0.95 * al), ls=6)
    px0, px1 = x0 + 52, x1 - 52
    py0, py1 = y0 + 108, y1 - 92
    ymin, ymax = 78.0, 133.0
    def P(year, val):
        u = (year - 1950) / (2070 - 1950)
        v = (val - ymin) / (ymax - ymin)
        return px0 + (px1 - px0) * u, py1 - (py1 - py0) * v
    for gv in (90, 110, 130):
        gy = P(1950, gv)[1]
        d.line([(px0, gy), (px1, gy)], fill=a(WASHI, 0.10 * al), width=2)
        ls_text(d, (px0 - 8, gy), str(gv), F_AXIS, a(FAINT, 0.85 * al), ls=1, anchor="rm")
    draw_u = ease_out_cubic(clamp01((p - 0.05) / 0.62))
    pts = [P(y, v) for (y, v, _f) in POP]
    n_hist = sum(1 for _y, _v, f in POP if not f)
    total = len(pts) - 1
    shown = draw_u * total
    def upto(lo, hi):
        out = []
        for i in range(lo, hi + 1):
            if i <= shown:
                out.append(pts[i])
            elif i - 1 <= shown:
                q = shown - (i - 1)
                out.append((pts[i-1][0] + (pts[i][0]-pts[i-1][0]) * q,
                            pts[i-1][1] + (pts[i][1]-pts[i-1][1]) * q))
                break
            else:
                break
        return out
    hist = upto(0, n_hist - 1)
    if len(hist) > 1:
        from PIL import ImageChops
        poly = hist + [(hist[-1][0], py1), (hist[0][0], py1)]
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).polygon(poly, fill=255)
        g = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        g.paste(vgrad((W, int(py1 - py0)), (*WASHI, int(58 * al)), (*WASHI, 0)), (0, int(py0)))
        xe = hist[-1][0]
        ramp = Image.new("L", (W, 1), 0)
        rp = ramp.load()
        for px in range(W):
            rp[px, 0] = 255 if px < xe - 150 else (0 if px >= xe else int(255 * (xe - px) / 150))
        m = ImageChops.multiply(m, ramp.resize((W, H)))
        g.putalpha(ImageChops.multiply(g.split()[3], m))
        ui.alpha_composite(g)
        d.line(hist, fill=a(WASHI, 0.95 * al), width=6, joint="curve")
    proj = upto(n_hist - 1, total)
    if len(proj) > 1:
        fine = []
        for i in range(len(proj) - 1):
            for k in range(6):
                q = k / 6.0
                fine.append((proj[i][0] + (proj[i+1][0]-proj[i][0]) * q,
                             proj[i][1] + (proj[i+1][1]-proj[i][1]) * q))
        fine.append(proj[-1])
        dashed(d, fine, a(RED, 0.95 * al), 6, dash=22, gap=15)
    peak = P(2008, 128.1)
    pk = clamp01((draw_u - 0.52) / 0.2)
    if pk > 0.02:
        d.ellipse((peak[0]-9, peak[1]-9, peak[0]+9, peak[1]+9), fill=a(WASHI, al * pk))
        pl = "PEAK 2008  ·  128M"
        pw = ls_width(d, pl, F_MICRO, 3)
        rrect(d, (peak[0]-pw/2-14, peak[1]-56, peak[0]+pw/2+14, peak[1]-18), 11,
              fill=a(INK, 0.90 * al * pk))
        ls_text(d, (peak[0], peak[1] - 37), pl, F_MICRO, a(WASHI, 0.97 * al * pk), ls=3, anchor="mm")
    end = P(2070, 87.0)
    ek = clamp01((draw_u - 0.93) / 0.07)
    if ek > 0.02:
        d.ellipse((end[0]-9, end[1]-9, end[0]+9, end[1]+9), fill=a(RED, al * ek))
        el = "87 MILLION"
        ew = ls_width(d, el, F_MICRO, 3)
        rrect(d, (end[0]-ew-40, end[1]-19, end[0]-18, end[1]+19), 11, fill=a(INK, 0.90 * al * ek))
        ls_text(d, (end[0]-29, end[1]), el, F_MICRO, a(RED, al * ek), ls=3, anchor="rm")
    dp = ease_out_cubic(clamp01((p - 0.70) / 0.30))
    if dp > 0.02:
        d.text((px1 - 6, py0 + 4), "−30%", font=F_ANNO, fill=a(RED, al * dp), anchor="ra")
    ls_text(d, (px0, py1 + 26), "1950", F_AXIS, a(FAINT, 0.9 * al), ls=1)
    ls_text(d, (px1, py1 + 26), "2070", F_AXIS, a(FAINT, 0.9 * al), ls=1, anchor="ra")

def cta(d, t):
    p, al = env(t, CTA["t0"], CTA["t1"], 0.5, 0.4)
    if al <= 0.01:
        return
    cx = W // 2
    y = 1330 + 44 * (1 - ease_out_cubic(p))
    wid = ls_width(d, CTA["main"], F_LBL, 7) + 104
    rrect(d, (cx - wid / 2, y, cx + wid / 2, y + 100), 50, fill=a(GOLD, 0.96 * al))
    ls_text(d, (cx, y + 50), CTA["main"], F_LBL, a(INK, al), ls=7, anchor="mm")
    sp = ease_out_cubic((p - 0.35) / 0.5)
    if sp > 0.02:
        sw = ls_width(d, CTA["sub"], F_TINY, 5)
        rrect(d, (cx - sw/2 - 26, y + 144, cx + sw/2 + 26, y + 198), 27, fill=a(INK, 0.82 * al * sp))
        ls_text(d, (cx, y + 171), CTA["sub"], F_TINY, a(WASHI, 0.95 * al * sp), ls=5, anchor="mm")

def chrome(d, t):
    y = 1874
    x0, x1 = MARGIN, W - MARGIN
    d.rounded_rectangle((x0, y, x1, y + 7), 4, fill=a(WASHI, 0.20))
    for (bt, _e, _n, _l) in CHAPTERS:
        tx = x0 + (x1 - x0) * (bt / DUR)
        d.rectangle((tx - 1, y - 4, tx + 1, y + 11), fill=a(WASHI, 0.30))
    fw = (x1 - x0) * clamp01(t / DUR)
    if fw > 4:
        d.rounded_rectangle((x0, y, x0 + fw, y + 7), 4, fill=a(GOLD, 0.95))
        d.ellipse((x0 + fw - 9, y - 6, x0 + fw + 9, y + 13), fill=a(WASHI, 0.95))
    ml = 0.34 if t < 67.5 else 0.34 * max(0.0, 1 - (t - 67.5) / 0.8)
    if ml > 0.01:
        ls_text(d, (W - MARGIN, 214), "TRUTH SEEKER", F_TINY, a(WASHI, ml), ls=6, anchor="ra",
                shadow=(a(INK, ml * 0.8), 2, 2))

FLASH = [5.6, 11.2, 17.4, 24.0, 31.0, 41.6, 48.2, 55.2, 62.0]
def fx(d, t):
    for f in FLASH:
        dt = t - f
        if -0.02 <= dt < 0.20:
            d.rectangle((0, 0, W, H), fill=a(WASHI, 0.16 * math.exp(-dt / 0.06)))
        if 0 <= dt < 0.42:
            u = ease_out_cubic(dt / 0.42)
            sx = -180 + (W + 360) * u
            al = 0.5 * (1 - u) ** 0.7
            for k in range(9):
                d.rectangle((sx - k * 16, 0, sx - k * 16 + 8, H), fill=a(GOLD, al * (1 - k / 9) * 0.5))
            d.rectangle((sx, 0, sx + 5, H), fill=a(WASHI, al))

# ---------------------------------------------------------------- frame
MAPPER = None
def frame(t):
    global MAPPER
    if MAPPER is None:
        MAPPER = wm.MapRenderer(W, H, ss=2, palette=MAP_PALETTE)
    cam = cam_at(t)
    glow = 0.16 + 0.30 * clamp01((t - 5.6) / 1.2)
    img = MAPPER.render(cam, highlight=("Japan",), glow=glow).convert("RGBA")
    img.alpha_composite(sea_depth())
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    blast_rings(d, cam, t)
    tokyo_marker(d, cam, t)
    arrows(d, cam, t)
    img.alpha_composite(ov)
    img.alpha_composite(SCRIM_TOP, (0, 0))
    img.alpha_composite(SCRIM_BOT, (0, H - 620))
    ui = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ui)
    chapter_chip(d, t)
    for h in HEADLINES:
        headline(d, t, h)
    reveal(d, t)
    for c in CARDS:
        stat_card(d, t, c)
    pop_chart(ui, d, t)
    cta(d, t)
    chrome(d, t)
    fx(d, t)
    img.alpha_composite(ui)
    return img.convert("RGB")

if __name__ == "__main__":
    if "--still" in sys.argv:
        outdir = sys.argv[-1]
        for ts in [float(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")]:
            frame(ts).save("%s/s_%06.2f.png" % (outdir, ts))
        sys.exit()
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else int(round(DUR * FPS))
    out = sys.stdout.buffer
    for i in range(lo, hi):
        out.write(frame(i / FPS).tobytes())
    out.flush()
