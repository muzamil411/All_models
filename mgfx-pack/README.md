# Motion Graphics Pack

A code-rendered graphics kit for faceless, English, long-form YouTube videos in
the style analysed in the reference teardown: dark ground, violet/magenta accent,
word-by-word captions, italic bold cards, flash-frame punctuation.

Every element renders to a **real video file with a real alpha channel** — drop
the `.mov` straight onto a timeline in Premiere, Resolve or Final Cut.

Nothing is fetched at render time. Fonts (Poppins, Montserrat) are local files.

## Setup

```bash
cd mgfx-pack
npm install
```

Requires Node 18+, ffmpeg on `PATH`, and a Chromium binary. Point at a specific
browser or ffmpeg with `CHROMIUM_PATH` / `FFMPEG_PATH` if the defaults miss.

## Rendering

```bash
node render.mjs <scene> [--params '<json>'] [--out name] [--fps 30] [--preview]
```

Each render writes to `out/<name>/`:

| File | Use |
|---|---|
| `frames/*.png` | Alpha sequence — the universal import, works everywhere |
| `<name>.mov` | ProRes 4444 with alpha — the file you actually drop on a timeline |
| `<name>.webm` | VP9 with alpha — quick review, web use |
| `<name>.mp4` | Flattened preview over a dark ground (`--preview` only) |

## Scenes

| Scene | What it is | Key params |
|---|---|---|
| `caption` | Word-by-word captions. Words **snap** in with no fade — that hard cut is what makes the pace read as fast. | `cues[{start,end,text,emphasis,color,sticker}]`, `sizePx`, `baselinePct` |
| `card-text` | Full-frame text card, italic headline over drifting blurred blobs, bullets staggering in 0.15 s apart. | `headline`, `bullets[]`, `bulletColors[]`, `accent`, `opaque` |
| `card-icons` | Neon icon tiles that draw their stroke then scale in, one per beat. | `title`, `icons[]` (`ebook` `templates` `video` `app` `chart` `rocket`) |
| `lower-third` | Mask-wipe headline plus character type-on body and an accent bar. | `head`, `body`, `accent` |
| `title-ghost` | Oversized ghosted title with slow tracking expansion. | `text`, `sizePx`, `opacity` |

All scenes take `duration` (seconds).

### Examples

```bash
# A stats card
node render.mjs card-text --out stats --preview --params '{
  "headline": "Some Other Key Stats:",
  "bullets": ["- 270 Website Views", "- 4 Buyers"],
  "duration": 3.2
}'

# The challenge lower third, and its magenta reprise later in the video
node render.mjs lower-third --out challenge --params '{
  "head": "This Video'"'"'s Challenge:",
  "body": "Make a sale on a brand new\nAI digital product",
  "accent": "#22D3EE"
}'
node render.mjs lower-third --out challenge-recap --params '{
  "head": "My Challenge:", "body": "Build a digital product\nand launch it",
  "accent": "#E040FB"
}'

# Captions from your own cue list
node render.mjs caption --out hook-caps --params '{
  "cues": [
    {"start":0.0,"end":0.38,"text":"AI CAN DO"},
    {"start":0.38,"end":0.92,"text":"90%","emphasis":true,"color":"#E23A45"},
    {"start":0.92,"end":1.6,"text":"OF THE WORK","sticker":"⚡"}
  ],
  "duration": 1.8
}'
```

Render everything at once:

```bash
./render-all.sh
```

## Transitions

```bash
# 1-2 frame solid at the cut — the reference uses 23 of these, all on
# section boundaries. Use them there and nowhere else.
python3 transitions.py flash A.mp4 B.mp4 out.mp4 --frames 2 --color white

# 4-6 frame zoom with RGB channel separation and directional blur
python3 transitions.py prism A.mp4 B.mp4 out.mp4 --frames 6 --zoom 0.40 --shift 16
```

The prism ramp is applied frame by frame, so zoom, channel offset and blur all
escalate smoothly rather than sitting at one constant value.

## Brand

Edit `brand.json` for colours, fonts and caption defaults, or override per
render with `--params`. The palette defaults to the reference's violet/magenta
on near-black; change `violet` and `magenta` to re-skin the whole pack.

## Adding a scene

Copy any file in `scenes/`. The contract is simple: no CSS animation, no
`requestAnimationFrame` — build the DOM as a pure function of `t` inside
`seek(t)` and call `defineScene({fps, duration, seek})`. That determinism is
what makes the offline render frame-accurate.

`lib/engine.js` provides `anim`, `tween`, `span` and `params`. The `expoOut`
curve matches the settle measured in the reference (displacement decaying
`(-22,-26) → (-12,-15) → (-3,-3) → 0` over roughly 0.8 s).
