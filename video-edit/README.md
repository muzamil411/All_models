# Italy re-edit pipeline

Post-production pass over a 576×1024 / 74.7 s vertical explainer ("Truth Seeker" Italy facts).
The original voice-over track is kept — everything around it is rebuilt.

## What it does

| Stage | File | Output |
|---|---|---|
| 1. Camera + grade | `build/make_base.py` | `base.mp4` — 3× lanczos upscale to 1080×1920, continuous keyframed zoom with beat punches, filmic grade, vignette, grain |
| 2. Sound design | `build/sfx.py` | `sfx.wav` — synthesised whooshes, sub impacts, risers, UI ticks placed on the edit beats |
| 3. Audio | `render.sh` | `audio_final.wav` — original track de-mudded / presence-lifted / compressed, mixed with SFX, normalised to −14 LUFS, −1 dBTP |
| 4. Graphics | `build/overlay.py` | RGBA frames piped to ffmpeg — chapter chips, hero title, count-up stat cards, section titles, progress bar, cut flashes and sweeps |

`build/timeline.py` holds the single source of truth: section boundaries, beat list and
hard cuts. Beats were derived from speech phrase gaps (RMS envelope minima) plus ffmpeg
scene detection, so the graphics and camera land on the narration rather than on a grid.

## Run

    ./render.sh path/to/source.mp4 outdir

Needs `ffmpeg`, and `python3` with `pillow`, `numpy`, `scipy`. Fonts (Anton, Montserrat,
Inter) are fetched from Google Fonts on first run.

## Notes

- The source's own burnt-in labels stay visible; the new stat cards are positioned to clear
  them rather than to cover them.
- Camera keyframes alternate lo/hi across section boundaries so the move is continuous —
  no snap at a cut, with a short exponential punch layered on each beat.
- Output is 1080×1920 / 30 fps / ~2.75 Mbps H.264 + 128 kbps AAC, which matches what
  TikTok, Reels and Shorts re-encode to anyway.
