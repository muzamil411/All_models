"""A flat-illustration avatar, drawn from scratch and animated as a short loop.

The character is a looping sticker rather than a lip-synced puppet, so a fixed
number of poses is rendered once and then cycled for the whole video.
"""
import math

from PIL import Image, ImageDraw

SIZE = (560, 1060)      # character canvas before it is scaled into the frame
SS = 3                  # supersampling factor for smooth edges
LOOP_FRAMES = 90        # a 3-second cycle at 30fps

SKIN = (247, 214, 189)
SKIN_SHADE = (228, 189, 162)
HAIR = (36, 29, 40)
BAND = (232, 92, 122)
SHIRT = (250, 250, 252)
SHIRT_SHADE = (219, 221, 231)
JEANS = (110, 152, 202)
JEANS_SHADE = (86, 124, 174)
BELT = (56, 60, 76)
SHOE = (240, 122, 62)
LINE = (28, 24, 32)


def _capsule(d, p0, p1, r, fill):
    """A thick line with rounded ends - the workhorse for limbs."""
    d.line([p0, p1], fill=fill, width=int(r * 2))
    for p in (p0, p1):
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=fill)


def _limb(d, root, side, upper_deg, fore_deg, length, r, fill):
    """Two-segment limb. Angles are degrees from straight-down; `side` is -1/+1
    for left/right, so a positive angle always swings away from the body."""
    def step(origin, deg, dist):
        a = math.radians(deg)
        return (origin[0] + side * math.sin(a) * dist, origin[1] + math.cos(a) * dist)

    elbow = step(root, upper_deg, length * 0.52)
    hand = step(elbow, fore_deg, length * 0.48)
    _capsule(d, root, elbow, r, fill)
    _capsule(d, elbow, hand, r * 0.86, fill)
    d.ellipse([hand[0] - r * 1.05, hand[1] - r * 1.05, hand[0] + r * 1.05, hand[1] + r * 1.05], fill=fill)


def _head(d, cx, cy, blink, mouth, s):
    hw, hh = s(152), s(170)

    # Hair: the mass behind the head plus the locks falling past the shoulders.
    d.ellipse([cx - hw - s(24), cy - hh - s(26), cx + hw + s(24), cy + hh * 0.55], fill=HAIR)
    for side in (-1, 1):
        d.rounded_rectangle(
            [cx + side * (hw - s(16)) - s(50), cy - s(40),
             cx + side * (hw - s(16)) + s(50), cy + hh + s(140)],
            radius=s(50), fill=HAIR,
        )

    d.ellipse([cx - hw, cy - hh, cx + hw, cy + hh], fill=SKIN)                       # face
    d.ellipse([cx + hw - s(30), cy + s(6), cx + hw + s(20), cy + s(76)], fill=SKIN_SHADE)  # ear

    # Fringe sweeping across the brow, heavier on the left.
    d.chord([cx - hw - s(6), cy - hh - s(8), cx + hw + s(6), cy + hh * 0.30], 178, 362, fill=HAIR)
    d.ellipse([cx - hw - s(10), cy - hh * 0.55, cx - hw + s(86), cy + hh * 0.34], fill=HAIR)
    # Headband over the fringe.
    d.arc([cx - hw - s(14), cy - hh - s(18), cx + hw + s(14), cy + hh * 0.16], 190, 350,
          fill=BAND, width=s(36))

    ey, ex = cy + s(30), s(58)
    for side in (-1, 1):
        px = cx + side * ex
        d.arc([px - s(36), ey - s(86), px + s(36), ey - s(24)], 202, 338, fill=LINE, width=s(9))
        if blink:
            d.arc([px - s(30), ey - s(26), px + s(30), ey + s(30)], 200, 340, fill=LINE, width=s(9))
        else:
            d.ellipse([px - s(29), ey - s(36), px + s(29), ey + s(36)], fill=LINE)
            d.ellipse([px - s(6), ey - s(26), px + s(14), ey - s(4)], fill=(255, 255, 255))

    for side in (-1, 1):
        d.ellipse([cx + side * s(104) - s(26), ey + s(38), cx + side * s(104) + s(26), ey + s(66)],
                  fill=(245, 172, 172))

    mh = s(10) + mouth * s(26)
    d.ellipse([cx - s(21), ey + s(78), cx + s(21), ey + s(78) + mh], fill=(150, 62, 68))


def _pose(frame):
    """Animation curves for one frame of the loop."""
    p = frame / LOOP_FRAMES
    bob = math.sin(p * math.tau) * 10
    sway = math.sin(p * math.tau) * 6
    # The right arm lifts into a wave over roughly a third of the cycle.
    wave = max(0.0, math.sin((p - 0.32) / 0.30 * math.pi)) if 0.32 < p < 0.62 else 0.0
    blink = 0.02 < (p % 0.5) < 0.055
    mouth = max(0.0, math.sin(p * math.tau * 3)) * 0.6
    return bob, sway, wave, blink, mouth


def _render_pose(frame):
    img = Image.new("RGBA", (SIZE[0] * SS, SIZE[1] * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bob, sway, wave, blink, mouth = _pose(frame)

    def s(v):
        return v * SS

    cx = s(SIZE[0] / 2)
    sh_y = s(500 + bob)          # shoulder line
    hip_y = s(672 + bob * 0.4)

    # Legs and shoes
    for side in (-1, 1):
        hip = (cx + side * s(46), hip_y)
        knee = (cx + side * s(52), s(826))
        ankle = (cx + side * s(56), s(960))
        _capsule(d, hip, knee, s(42), JEANS_SHADE if side < 0 else JEANS)
        _capsule(d, knee, ankle, s(35), JEANS_SHADE if side < 0 else JEANS)
        d.rounded_rectangle(
            [ankle[0] - s(46), ankle[1] - s(8), ankle[0] + s(50), ankle[1] + s(42)],
            radius=s(21), fill=SHOE,
        )

    # Torso: a slightly tapered tee with a shaded right side.
    d.polygon(
        [(cx - s(100), sh_y), (cx + s(100), sh_y),
         (cx + s(92), hip_y + s(18)), (cx - s(92), hip_y + s(18))],
        fill=SHIRT,
    )
    d.polygon(
        [(cx + s(42), sh_y), (cx + s(100), sh_y),
         (cx + s(92), hip_y + s(18)), (cx + s(42), hip_y + s(18))],
        fill=SHIRT_SHADE,
    )
    d.rounded_rectangle([cx - s(94), hip_y + s(2), cx + s(94), hip_y + s(34)], radius=s(14), fill=BELT)

    # Arms hang close to the body; the right one swings up to wave.
    _limb(d, (cx - s(96), sh_y + s(22)), -1, 14 + sway * 0.5, 8 + sway, s(250), s(29), SKIN)
    _capsule(d, (cx, sh_y - s(34)), (cx, sh_y + s(6)), s(36), SKIN_SHADE)   # neck
    _head(d, cx, s(292 + bob), blink, mouth, s)

    # Drawn last so the raised wave passes in front of the hair, not behind it.
    _limb(d, (cx + s(96), sh_y + s(22)), +1,
          14 + sway * 0.5 + wave * 128, 8 + sway + wave * 150, s(250), s(29), SKIN)

    return img.resize(SIZE, Image.LANCZOS)


_cache = None


def loop_frames():
    """Render (once) and return the full pose cycle."""
    global _cache
    if _cache is None:
        _cache = [_render_pose(i) for i in range(LOOP_FRAMES)]
    return _cache


def frame(index):
    return loop_frames()[index % LOOP_FRAMES]


if __name__ == "__main__":
    picks = [0, 20, 42, 50, 68]
    tw, th = SIZE[0] // 2, SIZE[1] // 2
    sheet = Image.new("RGB", (tw * len(picks), th), (26, 22, 36))
    for i, f in enumerate(picks):
        thumb = frame(f).resize((tw, th), Image.LANCZOS)
        sheet.paste(thumb, (i * tw, 0), thumb)
    sheet.save("out/_char_sheet.jpg", quality=92)
