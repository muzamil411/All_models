# 10 German Phrases You'll Actually Use Every Day

A finished vertical German-learning short for TikTok / Instagram Reels /
YouTube Shorts, generated end to end by one command.

**Output:** `output/german_daily_phrases.mp4` — 1080×1920, 9:16, 30 fps, H.264 +
AAC, 63.5 s, −14 LUFS.
**Thumbnail:** `output/thumbnail.png`

```bash
python3 generate_video.py
```

That single command prepares the assets, derives the timeline from the measured
voice clips, renders 1904 frames, mixes and masters the audio, muxes the MP4 and
then validates the result. It needs no network access — the ElevenLabs voice
clips and music bed are committed under `assets/`.

---

## Requirements

* Python 3.9+ with `Pillow` and `numpy` (`pip install -r requirements.txt`)
* `ffmpeg` and `ffprobe` on `PATH`

## What the video contains

A ~4.5 s spoken hook over an animated title card, then ten phrases. Each phrase
plays as **English cue → pause → German answer → pause long enough to repeat it
out loud**, with the English pre-listed in gold above and below so the viewer can
see what is coming. A call-to-action card closes it.

| # | English | German |
|---|---------|--------|
| 1 | Good morning. | Guten Morgen. |
| 2 | How are you? | Wie geht's dir? |
| 3 | I'm doing well. | Mir geht's gut. |
| 4 | What are you doing? | Was machst du? |
| 5 | Where do you live? | Wo wohnst du? |
| 6 | I'm from Pakistan. | Ich komme aus Pakistan. |
| 7 | How much does it cost? | Wie viel kostet das? |
| 8 | I don't understand. | Ich verstehe das nicht. |
| 9 | Can you help me? | Kannst du mir helfen? |
| 10 | See you tomorrow. | Bis morgen. |

## How timing works

Nothing is on a fixed grid. `src/timeline.py` probes the real duration of each of
the 23 ElevenLabs clips with `ffprobe` and lays the programme out from those
numbers, so a longer phrase simply takes longer on screen. The same cue list
drives both the on-screen text and the `adelay`/`amix` graph that places each
clip on the voice track — picture and sound are generated from one clock and
cannot drift apart.

Swap the voice or reword a phrase and the video re-times itself; only
`src/script.json` and the clips need to change.

## Layout

The reference video's visual language is kept — German flag badge and progress
bar pinned to the top, a cartoon presenter standing bottom-right, and a
left-aligned column where bold gold English is answered by white German beneath.

The one deliberate departure: these phrases are far longer than a two-word verb,
so listing all ten at once would force the type down to an unreadable size. The
column instead scrolls through a focus window — the active phrase settles on a
fixed line, gets a soft highlight and a glow, and the rest stay legible in a
muted gold above and below. Type stays at 64 px / 54 px, which reads comfortably
on a phone.

Inactive lines use their own muted inks rather than a lowered opacity: fading
gold toward the dark blue plate turns it olive, which reads as a mistake.

## Motion

* **Background** — a slow 10 % push-in with a lateral drift across the full
  running time.
* **Presenter** — breathing, a slow weight shift, a scale pulse on every new
  phrase, and a squash-based eye blink on an irregular schedule.
* **Phrases** — eased scroll steps, a highlight pad that fades in with the
  scroll, and the German line sliding up as it is first spoken.

## Audio

Voice is the priority throughout. The music bed sits ~26 dB down, is
sidechain-ducked against the voice, and the master applies **one static gain**
measured from the premaster rather than a one-pass `loudnorm` — a dynamic
normaliser rides the level back up during the repeat pauses, which both undoes
the ducking and puts an audible pump into the gaps.

Measured on the delivered file: **−14.0 LUFS** integrated, **−0.5 dBFS** peak,
voice **17.9 dB** above the bed.

## Assets

| Asset | Origin |
|---|---|
| `assets/audio/*.mp3` | 23 ElevenLabs clips, voice *Ava — youthful and expressive German* (`eleven_multilingual_v2`) |
| `assets/music/bed.mp3` | ElevenLabs `eleven_music_v2`; verified instrumental (transcription returns nothing) |
| `assets/background/plate.png` | ElevenLabs `bytedance-seedream-5-pro` render, extended to 9:16 by `scripts/prepare_assets.py` |
| `assets/character/teacher.png` | ElevenLabs `gemini-3-pro-image` render, background keyed out by border flood fill |
| `assets/background/flag.png` | Drawn in `scripts/prepare_assets.py` |
| `assets/fonts/*.ttf` | Montserrat (SIL Open Font License) via `@fontsource` |

The background generator returned 16:9 and the daily image quota was exhausted
before it could be re-rolled in portrait, so `prepare_assets.py` builds the
1296×2304 plate by growing the photograph's own sky upward: a clean patch from
left of the church spire is stretched, blurred and brightness-matched across a
300 px feather. The plate is 1.2× the output frame, which is the headroom the
push-in moves through.

## Layout of the repo

```
project/
├── reference/reference_video.mp4   the supplied reference, for comparison only
├── assets/{background,character,audio,music,fonts}/
├── src/
│   ├── script.json                 phrases, voice id, clip -> generation map
│   ├── timeline.py                 cue times derived from measured durations
│   ├── audio.py                    voice track, music bed, duck + master
│   └── render.py                   frame compositor
├── scripts/
│   ├── prepare_assets.py           cut-out, 9:16 plate, flag
│   ├── fetch_audio.py              re-record the voice (needs an API key)
│   └── build_fonts.py              woff2 -> ttf
├── output/{german_daily_phrases.mp4,thumbnail.png}
├── generate_video.py
└── requirements.txt
```

`build/` holds intermediates (`silent.mp4`, `voice.wav`, `mix.wav`,
`timeline.json`) and is safe to delete.

## Regenerating the voice

The committed clips make the build offline and deterministic. To re-record:

```bash
export ELEVENLABS_API_KEY=...     # never commit this
python3 scripts/fetch_audio.py    # add --music to re-roll the bed too
python3 generate_video.py
```

## Validation

`generate_video.py` refuses to report success unless the delivered file passes
every check, printed at the end of the run: resolution, codec, pixel format,
frame rate, audio codec and channel count, duration inside the 60–75 s brief and
matching the timeline, no black frames mid-programme, measurable audio below
clipping, integrated loudness in range, and at least 12 dB of separation between
voice and music bed.

## Reference

`reference/reference_video.mp4` was used only to study composition, character
placement, text hierarchy, colour, pacing and reveal style. No text, script,
phrase, voice line or graphical asset from it is reused — every asset here is
generated or drawn for this project.
