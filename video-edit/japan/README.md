# Japan explainer — built from scratch

A 72.5 s vertical explainer with no source footage and no voice-over: every
frame is generated. Companion to the re-edit pipeline in the parent folder,
which reuses the same typography helpers (`../build/gfx.py`).

## Story

Catastrophe → recovery → the hidden crisis → the opening.

1. Two atomic bombs, August 1945
2. By 1968 the world's #2 economy
3. The land — 377,975 km², 14,125 islands
4. The people — 123 million
5. **The problem** — population peaked in 2008 and is projected to fall ~30% by 2070
6. The economy — $4.2 trillion, 4th largest
7. The money — ~$30,000/year average salary
8. The engineering — 320 km/h, zero passenger deaths since 1964
9. The opening — 820,000 foreign workers wanted by 2029

## Pieces

| File | Role |
|---|---|
| `story.py` | Beat sheet, camera keyframes, palette, chart data — the single source of truth |
| `worldmap.py` | Slippy-map projection over Natural Earth country polygons, animated camera, 2× supersampled PIL rendering |
| `render.py` | Frame renderer: map, blast rings, Tokyo marker, migration arrows, headlines, stat cards, the population chart, chrome |
| `music.py` | The score — Karplus-Strong koto, synthesised taiko, breathy flute, drone and pad, all in the Japanese *in* scale on D |
| `render_all.sh` | Four parallel render workers → concat → grain and grade → two-pass encode |

## Notes on the score

It is hit-point driven rather than grid-locked: a taiko lands on every edit
beat from `story.BEATS`, with softer answers between them, over a sustained
drone and per-section pad chords. That keeps the music locked to the cut
without forcing the edit onto a fixed tempo grid.

## Run

    ./render_all.sh

Needs `ffmpeg`, and `python3` with `pillow`, `numpy`, `scipy`, `geopandas`
(0.14.x, for the bundled `naturalearth_lowres` polygons). Fonts come from the
parent `../fonts` directory.
