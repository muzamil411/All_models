# video_maker

Generates a vertical "language in seconds" short of the kind that does well on
TikTok/Reels: a flag badge, a two-colour word list where the translations reveal
one at a time, an animated avatar, a voiceover and a music bed.

Everything a video says and shows lives in a JSON file under `content/`.

| Video | Content file | Length |
|---|---|---|
| `out/english_in_seconds.mp4` | `content/english_in_seconds.json` | 23.4s, 5 words, Roman Urdu → English |
| `out/german_question_words.mp4` | `content/german_question_words.json` | 70s, 10 words, English → German, with a spoken outro |

Both are 1080x1920, 30fps, H.264 + AAC.

## Running it

```bash
pip install pillow numpy imageio-ffmpeg
python3 make_video.py --content content/german_question_words.json --out out/my_video.mp4

# Stills instead of a full render, for checking layout quickly:
python3 make_video.py --content content/german_question_words.json --preview 60,900,1980
```

## Making a different video

Everything the video says lives in the JSON. Copy `content/english_in_seconds.json`,
change the pairs, and render:

```json
{
  "flag": "de",            // "uk" or "de" - the badge at the top
  "style": "canal",        // backdrop: "street" or "canal"
  "seed": 21,              // reshuffles the backdrop's buildings and lights
  "first_reveal": 6.0,     // seconds before the first translation appears
  "reveal_gap": 5.3,       // seconds between reveals
  "beat_bar": true,        // pacing bar refills once per word (else "bar_seconds": 4.2 once)
  "outro_start": 56.4,     // list clears, character walks to centre stage
  "duration": 70.0,
  "pairs": [["Where", "Wo"], ["When", "Wann"]]
}
```

The first item of each pair is the language the viewer already speaks (gold, on
screen from the first frame); the second is the phrase being taught (white,
revealed on its beat). Type size and line spacing are chosen from the number of
pairs — five words get a large scale, ten a compact one — and a `layout` object
overrides any of `top`, `gap`, `known`, `taught`, `drop` if you need to.

Leave `outro_start` out and there is no outro: the list stays up and the
character stays in its corner to the last frame.

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

## Adding a flag

`FLAGS` in `make_video.py` maps the content file's `flag` value to a painter.
`tricolour_h` covers any three-band horizontal flag, so most new languages are
one line; `badge()` handles the rounded corners, ring and drop shadow.

## Voiceover

Clips are generated with ElevenLabs and committed under `voice/` and `voice_de/`,
so renders stay reproducible with no network and no API key. Voices used:
*Alice - Clear, Engaging Educator* for English, *Ava - youthful and expressive
German* for German, both on `eleven_multilingual_v2`.

The content file schedules them:

```json
"voice": {
  "intro": {"file": "voice_de/intro.mp3", "at": 0.3},
  "outro": {"file": "voice_de/outro.mp3", "at": 57.2},
  "cue_offset": -1.2,
  "reveal_offset": 0.0,
  "cues":  ["voice_de/en01.mp3", "..."],
  "files": ["voice_de/de01.mp3", "..."]
}
```

`cues` and `files` both line up with `pairs`. Per word the English cue plays
`cue_offset` seconds before the reveal and the translation lands on it, which is
what gives the video its call-and-answer rhythm. `audio.py` ducks the music bed
under whatever is speaking, so no manual level balancing is needed.

Every key is optional: drop `cues` for translations only, drop `outro` for no
closing line, drop the whole `"voice"` object and the video renders with music
only. A missing file is reported before rendering starts rather than halfway
through.
