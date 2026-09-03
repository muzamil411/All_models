# video_maker

Generates a vertical "language in seconds" short of the kind that does well on
TikTok/Reels: a flag badge, a two-colour word list where the translations reveal
one at a time, an animated avatar, and a music bed.

Output: `out/english_in_seconds.mp4` — 1080x1920, 30fps, 23.4s, H.264 + AAC.

## Running it

```bash
pip install pillow numpy imageio-ffmpeg
python3 make_video.py --content content/english_in_seconds.json --out out/my_video.mp4

# Stills instead of a full render, for checking layout quickly:
python3 make_video.py --preview 0,90,300,690
```

## Making a different video

Everything the video says lives in the JSON. Copy `content/english_in_seconds.json`,
change the pairs, and render:

```json
{
  "seed": 7,               // reshuffles the backdrop's buildings and lights
  "first_reveal": 5.4,     // seconds before the first translation appears
  "reveal_gap": 3.6,       // seconds between reveals
  "bar_seconds": 4.2,      // how long the loading bar under the badge runs
  "pairs": [["Salam", "Hello"], ["Shukriya", "Thank You"]]
}
```

The first item of each pair is the language the viewer already speaks (shown in
gold from the first frame); the second is the phrase being taught (white, revealed
on its beat). `duration` is derived from the reveal schedule unless you set it.

## Files

| File | What it does |
|---|---|
| `background.py` | Paints the dusk city street, then crops it per frame for the Ken Burns move |
| `character.py` | Draws the avatar and renders a 90-frame pose loop once, reused for the whole video |
| `audio.py` | Synthesises the pad, arpeggio and reveal chimes with numpy |
| `make_video.py` | Lays out the text and chrome, composites every frame, pipes to ffmpeg |

Nothing is downloaded at render time — no stock footage, no fonts to install
beyond Liberation Sans, no API keys.

## Swapping the flag

`union_jack()` in `make_video.py` paints the badge. For another language, replace
that function with one that draws the flag you want (or load a PNG and skip the
drawing); `badge()` handles the rounded corners, ring and drop shadow either way.

## Adding a voiceover

The renderer has no TTS — this environment has no speech engine and no network.
To add narration, generate one clip per phrase and mix them in at the reveal
times, which `load_content()` already computes as `content["reveals"]`:

```python
# after audio.build(...), before the mux
# overlay each voice clip at reveals[i] and write a combined wav
```

ElevenLabs gives the most natural result; TikTok's own text-to-speech is free and
fast if you would rather add it in the app after uploading.
