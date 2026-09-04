#!/usr/bin/env python3
"""
Regenerate every voiceover clip (and optionally the music bed) from ElevenLabs.

The clips checked into assets/audio/ were produced with these exact settings, so
the committed project builds offline. Run this only if you want to re-record the
voice — e.g. to swap the presenter's voice for a different one.

    export ELEVENLABS_API_KEY=...
    python3 scripts/fetch_audio.py            # voice clips
    python3 scripts/fetch_audio.py --music    # voice clips + a new music bed

Timings are measured from whatever comes back, so a different voice simply
produces a differently-paced video — nothing downstream needs editing.
"""
import argparse
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://api.elevenlabs.io/v1"

MUSIC_PROMPT = (
    "A soft, warm, minimal lo-fi study instrumental for a language-learning "
    "video. Gentle mellow electric piano chords, a soft rounded sub bass, light "
    "brushed hi-hats and a very relaxed slow beat around 75 BPM. Calm, friendly, "
    "optimistic and encouraging. Extremely sparse and understated, no melody that "
    "competes with a speaking voice, no vocals, no risers, no big drops, no sudden "
    "dynamic changes. Even, steady volume throughout so it can sit quietly under a "
    "narrator. Loopable."
)


def post(path, payload, key):
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(payload).encode(),
        headers={"xi-api-key": key, "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", action="store_true", help="also regenerate the bed")
    args = ap.parse_args()

    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("set ELEVENLABS_API_KEY (never hard-code it)")

    with open(os.path.join(ROOT, "src", "script.json"), encoding="utf-8") as f:
        script = json.load(f)
    voice_id = script["voice"]["voice_id"]
    model_id = script["voice"]["model_id"]

    out_dir = os.path.join(ROOT, "assets", "audio")
    os.makedirs(out_dir, exist_ok=True)

    for clip in script["clips"]:
        audio = post(f"/text-to-speech/{voice_id}",
                     {"text": clip["text"], "model_id": model_id}, key)
        dst = os.path.join(out_dir, clip["id"] + ".mp3")
        with open(dst, "wb") as f:
            f.write(audio)
        print(f"  {clip['id']:7s} {len(audio):7d} bytes  {clip['text']}")

    if args.music:
        music = post("/music",
                     {"prompt": MUSIC_PROMPT, "music_length_ms": 70000}, key)
        dst = os.path.join(ROOT, "assets", "music", "bed.mp3")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(music)
        print(f"  music   {len(music):7d} bytes")

    print("\ndone — now run: python3 generate_video.py")


if __name__ == "__main__":
    main()
