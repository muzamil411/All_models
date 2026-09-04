"""
Frame renderer for the German Daily Phrases short.

Layout follows the reference video's visual language: German flag badge and a
progress bar pinned to the top, a cartoon presenter standing bottom-right, and
a left-aligned column of phrases where a bold yellow English line is answered
by a white German line underneath.

Because these phrases are much longer than the reference's two-word verbs, the
column scrolls through a focus window instead of listing all ten at once. That
keeps the type large enough to read on a phone while preserving the same
"checklist" feel.
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1080, 1920
FPS = 30

YELLOW = (255, 214, 61)
WHITE = (255, 255, 255)
# Inactive lines get their own muted inks rather than a lowered alpha: fading
# yellow towards the dark blue plate turns it olive, which reads as a mistake.
YELLOW_DIM = (208, 176, 76)
WHITE_DIM = (206, 212, 222)

MARGIN_L = 76
SAFE_R = 44              # right-hand safe margin the longest phrase must respect

FLAG_W, FLAG_H = 196, 131
FLAG_Y = 116
BAR_Y = FLAG_Y + FLAG_H + 40
BAR_W, BAR_H = 372, 12

VIEW_TOP, VIEW_BOTTOM = 372, 1214     # scrolling window for the phrase column
FOCUS_Y = 712                         # where the active phrase settles
FADE = 128                            # soft edges on the window

EN_SIZE, DE_SIZE = 64, 54
EN_DE_GAP = 12
ENTRY_GAP = 60

CHAR_H = 636
CHAR_CX, CHAR_BOTTOM = 892, 1872

SCROLL_LEAD = 0.50       # scroll begins this long before the English line
SCROLL_DUR = 0.62

# Eye boxes in the 234x695 character source, used for the blink.
EYE_BOXES = [(65, 125, 111, 160), (125, 125, 171, 160)]
BLINK_LEVELS = 5
BLINK_DUR = 0.17
# Irregular spacing reads as natural; a fixed period looks mechanical.
BLINK_TIMES = [2.1, 5.6, 9.0, 13.4, 17.1, 21.9, 25.4, 29.8, 33.2, 37.9,
               41.4, 45.0, 49.7, 53.1, 57.4, 60.8]


def ease(t):
    """Smootherstep — no visible acceleration kink at either end."""
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (t * (t * 6 - 15) + 10)


def fade_in(t, start, dur):
    return ease((t - start) / dur) if dur > 0 else float(t >= start)


class Renderer:
    def __init__(self, project_root, timeline):
        self.root = project_root
        self.tl = timeline
        a = os.path.join(project_root, "assets")

        self.plate = Image.open(os.path.join(a, "background", "plate.png")).convert("RGB")
        self.flag = Image.open(os.path.join(a, "background", "flag.png")).convert("RGBA")
        self.flag = self.flag.resize((FLAG_W, FLAG_H), Image.LANCZOS)

        char = Image.open(os.path.join(a, "character", "teacher.png")).convert("RGBA")
        cw = round(char.width * CHAR_H / char.height)
        # Build the blink frames at source resolution, then scale each once.
        self.char_frames = [
            self._blink_variant(char, i / (BLINK_LEVELS - 1)).resize(
                (cw, CHAR_H), Image.LANCZOS)
            for i in range(BLINK_LEVELS)
        ]
        self.char = self.char_frames[0]

        f = os.path.join(a, "fonts")
        self.f_en = ImageFont.truetype(os.path.join(f, "montserrat-latin-800.ttf"), EN_SIZE)
        self.f_de = ImageFont.truetype(os.path.join(f, "montserrat-latin-700.ttf"), DE_SIZE)
        self.f_title = ImageFont.truetype(os.path.join(f, "montserrat-latin-800.ttf"), 92)
        self.f_sub = ImageFont.truetype(os.path.join(f, "montserrat-latin-700.ttf"), 56)
        self.f_num = ImageFont.truetype(os.path.join(f, "montserrat-latin-800.ttf"), 30)

        self._build_entries()
        self._build_window_mask()
        self._build_intro_outro()

    @staticmethod
    def _blink_variant(img, k):
        """Squash each eye down toward the lower lid — the classic 2D blink.

        The vacated space is filled with skin sampled from the eyelid directly
        above the eye, so the patch always matches the shading around it.
        """
        if k <= 0.001:
            return img.copy()
        out = img.copy()
        for x0, y0, x1, y1 in EYE_BOXES:
            w, h = x1 - x0, y1 - y0
            # Sample below the eye (cheek), never above: the brows sit
            # immediately over the lid and would tint the patch dark.
            skin = img.getpixel(((x0 + x1) // 2, y1 + 8))
            eye = img.crop((x0, y0, x1, y1))
            nh = max(2, int(round(h * (1.0 - 0.86 * k))))
            # Feather the patch edges: a flat fill against the face's soft
            # shading otherwise leaves a faint rectangle around the eye.
            mask = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask).rectangle([3, 3, w - 4, h - 4], fill=255)
            out.paste(Image.new("RGBA", (w, h), skin), (x0, y0),
                      mask.filter(ImageFilter.GaussianBlur(2.2)))
            out.paste(eye.resize((w, nh), Image.LANCZOS), (x0, y1 - nh))
        return out

    def _blink_level(self, t):
        for bt in BLINK_TIMES:
            dt = t - bt
            if 0.0 <= dt <= BLINK_DUR:
                # close for the first half, open for the second
                u = dt / BLINK_DUR
                k = u / 0.45 if u < 0.45 else (1.0 - u) / 0.55
                return min(int(round(min(max(k, 0.0), 1.0) * (BLINK_LEVELS - 1))),
                           BLINK_LEVELS - 1)
        return 0

    # ------------------------------------------------------------------ layout
    def _text_sprite(self, text, font, fill, pad=26, glow=None):
        """Render text once onto its own RGBA tile, with a drop shadow.

        The shadow is what keeps white and yellow legible over the brighter
        parts of the photograph.
        """
        tmp = Image.new("RGBA", (1, 1))
        box = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
        w, h = box[2] - box[0], box[3] - box[1]
        img = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        ox, oy = pad - box[0], pad - box[1]

        shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(shadow).text((ox, oy + 4), text, font=font, fill=(0, 0, 0, 205))
        img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(7)))
        img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(2)))

        if glow:
            g = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(g).text((ox, oy), text, font=font, fill=glow + (150,))
            img.alpha_composite(g.filter(ImageFilter.GaussianBlur(16)))

        d.text((ox, oy), text, font=font, fill=fill + (255,))
        return img, (ox, oy), (w, h)

    def _build_entries(self):
        """Pre-render every phrase line and stack them into a virtual column."""
        self.entries = []
        y = 0
        for p in self.tl["phrases"]:
            en_dim, _, en_wh = self._text_sprite(p["en_text"], self.f_en, YELLOW_DIM)
            en_hot, _, _ = self._text_sprite(p["en_text"], self.f_en, YELLOW, glow=YELLOW)
            de_dim, _, de_wh = self._text_sprite(p["de_text"], self.f_de, WHITE_DIM)
            de_hot, _, _ = self._text_sprite(p["de_text"], self.f_de, WHITE, glow=(190, 220, 255))

            en_h, de_h = en_wh[1], de_wh[1]
            block_h = en_h + EN_DE_GAP + de_h
            self.entries.append({
                "p": p,
                "en_dim": en_dim, "en_hot": en_hot,
                "de_dim": de_dim, "de_hot": de_hot,
                "en_wh": en_wh, "de_wh": de_wh,
                "pad_w": max(en_wh[0], de_wh[0]) + 92,
                "top": y, "h": block_h,
                "en_y": y, "de_y": y + en_h + EN_DE_GAP,
                "centre": y + block_h / 2,
            })
            y += block_h + ENTRY_GAP
        self.column_h = y

        # The phrase window (VIEW_BOTTOM) closes above where the presenter starts,
        # so long lines cannot run into her; they only have to stay inside the
        # frame's safe area.
        widest = MARGIN_L + max(max(e["en_wh"][0], e["de_wh"][0]) for e in self.entries)
        if widest > W - SAFE_R:
            raise ValueError(
                f"longest phrase reaches x={widest}px, past the {W - SAFE_R}px "
                f"safe edge — shorten it or drop EN_SIZE")
        char_top = CHAR_BOTTOM - CHAR_H
        if VIEW_BOTTOM > char_top:
            raise ValueError(
                f"phrase window ({VIEW_BOTTOM}) overlaps the presenter ({char_top})")

    def _build_window_mask(self):
        """Soft top/bottom edges so the column dissolves rather than cuts."""
        m = np.zeros((H, 1), dtype=np.float32)
        m[VIEW_TOP:VIEW_BOTTOM] = 1.0
        m[VIEW_TOP:VIEW_TOP + FADE, 0] = np.linspace(0, 1, FADE)
        m[VIEW_BOTTOM - FADE:VIEW_BOTTOM, 0] = np.linspace(1, 0, FADE)
        self.window_mask = np.repeat(m, W, axis=1)

    def _build_intro_outro(self):
        self.intro_lines = [
            (self._text_sprite("10 GERMAN", self.f_title, YELLOW, glow=YELLOW), 0),
            (self._text_sprite("PHRASES", self.f_title, YELLOW, glow=YELLOW), 1),
            (self._text_sprite("you'll actually use", self.f_sub, WHITE), 2),
            (self._text_sprite("EVERY DAY", self.f_title, WHITE, glow=(190, 220, 255)), 3),
        ]
        self.outro_lines = [
            (self._text_sprite("SAVE THIS VIDEO", self.f_title, YELLOW, glow=YELLOW), 0),
            (self._text_sprite("& practice every day", self.f_sub, WHITE), 1),
            (self._text_sprite("FOLLOW FOR MORE", self.f_sub, WHITE), 2),
            (self._text_sprite("GERMAN!", self.f_title, YELLOW, glow=YELLOW), 3),
        ]

    # ------------------------------------------------------------- backgrounds
    def background(self, t):
        """Slow push-in with a gentle drift, so the frame is never static."""
        d = self.tl["duration"]
        u = t / d
        zoom = 1.0 + 0.105 * ease(min(u * 1.06, 1.0))
        pw, ph = self.plate.size
        cw, ch = pw / zoom, ph / zoom
        # Drift down-left over the run: the town rises slightly into frame.
        cx = pw / 2 + (pw - cw) * 0.5 * (0.30 * math.sin(u * math.pi * 0.9 - 0.4))
        cy = ph / 2 + (ph - ch) * 0.5 * (0.34 - 0.72 * u)
        left = min(max(cx - cw / 2, 0), pw - cw)
        top = min(max(cy - ch / 2, 0), ph - ch)
        crop = self.plate.resize((W, H), Image.BILINEAR,
                                 box=(left, top, left + cw, top + ch))
        return crop.convert("RGBA")

    def character(self, frame, t):
        """Breathing, a slow weight shift, and a small pulse on each new phrase."""
        breathe = math.sin(t * 2 * math.pi / 3.4)
        sway = math.sin(t * 2 * math.pi / 5.7)
        lean = math.sin(t * 2 * math.pi / 8.3)

        pulse = 0.0
        for p in self.tl["phrases"]:
            dt = t - p["en_start"]
            if 0 <= dt < 0.85:
                pulse = max(pulse, math.sin(math.pi * dt / 0.85) * math.exp(-dt * 2.2))
            dt = t - p["de_start"]
            if 0 <= dt < 0.7:
                pulse = max(pulse, 0.7 * math.sin(math.pi * dt / 0.7) * math.exp(-dt * 2.4))

        scale = 1.0 + 0.006 * breathe + 0.028 * pulse
        img = self.char_frames[self._blink_level(t)]
        cw, ch = round(img.width * scale), round(img.height * scale)
        img = img.resize((cw, ch), Image.BILINEAR)
        img = img.rotate(1.15 * sway, resample=Image.BICUBIC, expand=True)

        x = round(CHAR_CX - img.width / 2 + 9 * lean)
        y = round(CHAR_BOTTOM - img.height + 5 * breathe - 16 * pulse)
        frame.alpha_composite(img, (x, y))

    def chrome(self, frame, t):
        """Flag badge and the progress bar that tracks how far the list has got."""
        fx = (W - FLAG_W) // 2
        bob = round(3 * math.sin(t * 2 * math.pi / 4.6))
        frame.alpha_composite(self.flag, (fx, FLAG_Y + bob))

        ph = self.tl["phrases"]
        if t < ph[0]["en_start"]:
            prog = 0.0
        elif t >= self.tl["phrases_end"]:
            prog = 1.0
        else:
            span = self.tl["phrases_end"] - ph[0]["en_start"]
            prog = (t - ph[0]["en_start"]) / span

        bar = Image.new("RGBA", (BAR_W, BAR_H), (0, 0, 0, 0))
        d = ImageDraw.Draw(bar)
        r = BAR_H // 2
        d.rounded_rectangle([0, 0, BAR_W - 1, BAR_H - 1], radius=r, fill=(255, 255, 255, 62))
        fill_w = int(BAR_W * prog)
        if fill_w > BAR_H:
            d.rounded_rectangle([0, 0, fill_w, BAR_H - 1], radius=r, fill=(255, 214, 61, 240))
        frame.alpha_composite(bar, ((W - BAR_W) // 2, BAR_Y + bob))

    # ------------------------------------------------------------------ phrases
    def _scroll_offset(self, t):
        """Column position: eased steps that land each phrase on the focus line."""
        ents = self.entries
        target = ents[0]["centre"]
        for i, e in enumerate(ents):
            trigger = e["p"]["en_start"] - SCROLL_LEAD
            if t >= trigger:
                if i == 0:
                    target = e["centre"]
                else:
                    prev = ents[i - 1]["centre"]
                    k = ease((t - trigger) / SCROLL_DUR)
                    target = prev + (e["centre"] - prev) * k
        return FOCUS_Y - target

    def phrases(self, frame, t):
        intro = self.tl["intro"]
        appear = fade_in(t, intro["list_in"], 0.85)
        leave = 1.0 - fade_in(t, self.tl["outro"]["start"] + 0.15, 0.55)
        overall = appear * leave
        if overall <= 0.004:
            return

        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        off = self._scroll_offset(t)

        # Which phrase currently owns the focus line.
        active = 0
        for i, e in enumerate(self.entries):
            if t >= e["p"]["en_start"] - SCROLL_LEAD * 0.5:
                active = i

        for i, e in enumerate(self.entries):
            base_y = e["top"] + off
            if base_y > VIEW_BOTTOM + 90 or base_y + e["h"] < VIEW_TOP - 90:
                continue

            is_active = (i == active)
            # Focus ramps in with the scroll so nothing pops.
            k = ease((t - (e["p"]["en_start"] - SCROLL_LEAD)) / SCROLL_DUR) if is_active else 0.0
            seen = t >= e["p"]["en_start"] - SCROLL_LEAD
            alpha = (0.88 + 0.12 * k) if is_active else (0.90 if seen else 0.72)

            if is_active and k > 0.02:
                # Soft highlight pad behind the active phrase.
                pad = Image.new("RGBA", (e["pad_w"], e["h"] + 74), (0, 0, 0, 0))
                ImageDraw.Draw(pad).rounded_rectangle(
                    [0, 0, pad.width - 1, pad.height - 1], radius=30,
                    fill=(12, 20, 38, int(96 * k)))
                pad = pad.filter(ImageFilter.GaussianBlur(11))
                layer.alpha_composite(pad, (MARGIN_L - 46, round(base_y - 34)))

            en = e["en_hot"] if is_active else e["en_dim"]
            self._blit(layer, en, MARGIN_L, e["en_y"] + off, alpha)

            # The German answer only exists once it has been spoken.
            de_a = fade_in(t, e["p"]["de_start"] - 0.16, 0.34)
            if de_a > 0.01:
                de = e["de_hot"] if is_active else e["de_dim"]
                slide = round(16 * (1.0 - de_a))
                self._blit(layer, de, MARGIN_L + 4, e["de_y"] + off + slide,
                           alpha * de_a)

        arr = np.array(layer, dtype=np.float32)
        arr[:, :, 3] *= self.window_mask * overall
        frame.alpha_composite(Image.fromarray(arr.astype(np.uint8)))

    @staticmethod
    def _blit(layer, sprite, x, y, alpha):
        if alpha >= 0.995:
            layer.alpha_composite(sprite, (round(x) - 26, round(y) - 26))
            return
        if alpha <= 0.004:
            return
        a = sprite.getchannel("A").point(lambda v: int(v * alpha))
        s = sprite.copy()
        s.putalpha(a)
        layer.alpha_composite(s, (round(x) - 26, round(y) - 26))

    # ------------------------------------------------------------ intro / outro
    def _card(self, frame, lines, t, t0, centre_y, stagger=0.13, hold_out=None):
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        heights = [ln[0][2][1] for ln in lines]
        gaps = [26, 30, 26]
        total = sum(heights) + sum(gaps[:len(lines) - 1])
        y = centre_y - total / 2
        drawn = False
        for idx, ((sprite, _, (tw, th)), order) in enumerate(lines):
            a = fade_in(t, t0 + order * stagger, 0.42)
            if hold_out is not None:
                a *= 1.0 - fade_in(t, hold_out, 0.42)
            if a > 0.01:
                rise = round(30 * (1.0 - a))
                self._blit(layer, sprite, (W - tw) // 2, y + rise, a)
                drawn = True
            y += th + (gaps[idx] if idx < len(gaps) else 26)
        if drawn:
            frame.alpha_composite(layer)

    def intro(self, frame, t):
        i = self.tl["intro"]
        if t > i["list_in"] + 0.9:
            return
        self._card(frame, self.intro_lines, t, 0.20, 800, hold_out=i["list_in"] - 0.05)

    def outro(self, frame, t):
        o = self.tl["outro"]
        if t < o["start"]:
            return
        self._card(frame, self.outro_lines, t, o["start"] + 0.35, 820, stagger=0.16)

    # ----------------------------------------------------------------- compose
    def frame(self, i):
        t = i / FPS
        f = self.background(t)
        self.character(f, t)
        self.chrome(f, t)
        self.phrases(f, t)
        self.intro(f, t)
        self.outro(f, t)
        # Lift in quickly and close on black. The opening starts part-lit rather
        # than at zero: a fully black first frame is what some platforms grab as
        # the cover image, and it wastes the front of the hook.
        d = self.tl["duration"]
        k = min(0.22 + 0.78 * fade_in(t, 0.0, 0.30),
                1.0 - fade_in(t, d - 0.55, 0.55))
        if k < 0.999:
            f = Image.blend(Image.new("RGBA", (W, H), (0, 0, 0, 255)), f, k)
        return f.convert("RGB")
