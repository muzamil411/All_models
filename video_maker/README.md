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
| `audio.py` | Synthesises the pad, arpeggio and reveal chimes, then mixes the voiceover over them |
| `make_video.py` | Lays out the text and chrome, composites every frame, pipes to ffmpeg |

Nothing is downloaded at render time. The voiceover clips in `voice/` are
committed, so a render needs no network and no API key — see below for
regenerating them.

## Swapping the flag

`union_jack()` in `make_video.py` paints the badge. For another language, replace
that function with one that draws the flag you want (or load a PNG and skip the
drawing); `badge()` handles the rounded corners, ring and drop shadow either way.

## Voiceover

The clips in `voice/` were generated with ElevenLabs (voice *Alice - Clear,
Engaging Educator*, model `eleven_multilingual_v2`) and are committed so renders
stay reproducible. The content file schedules them:

```json
"voice": {
  "intro": {"file": "voice/intro.mp3", "at": 1.5},
  "reveal_offset": 0.12,
  "files": ["voice/p1.mp3", "voice/p2.mp3", "voice/p3.mp3", "voice/p4.mp3", "voice/p5.mp3"]
}
```

`files` lines up with `pairs`, and each clip plays `reveal_offset` seconds after
its word appears, so the viewer reads it a beat before hearing it. `audio.py`
ducks the music bed under the narration automatically, so no manual level
balancing is needed.

To voice a new set of phrases, render one clip per phrase, drop them in `voice/`,
and point `files` at them. Drop the whole `"voice"` key and the video renders
with music only.
