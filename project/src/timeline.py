"""
Timeline construction for the German Daily Phrases short.

Every cue is derived from the *measured* duration of the ElevenLabs clips, so
the on-screen text can never drift away from the voice. Nothing here assumes a
fixed slot length: a long phrase simply occupies more of the timeline.
"""
import json
import os
import subprocess

FPS = 30

# Beats, in seconds.
INTRO_LEAD = 0.35       # silence before the hook line starts
INTRO_TAIL = 0.55       # beat after the hook line, while the list fades in
EN_TO_DE_GAP = 0.40     # pause between the English cue and the German answer
REPEAT_GAP = 2.55      # room for the viewer to say the German phrase back
OUTRO_LEAD = 0.70       # beat between the last phrase and the call to action
OUTRO_MID = 0.55        # pause between the two call-to-action lines
OUTRO_HOLD = 2.60       # final card holds on screen after the voice stops


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def build(project_root):
    """Return (timeline dict, audio cue list) with absolute times in seconds."""
    with open(os.path.join(project_root, "src", "script.json"), encoding="utf-8") as f:
        script = json.load(f)

    audio_dir = os.path.join(project_root, "assets", "audio")
    dur = {c["id"]: probe_duration(os.path.join(audio_dir, c["id"] + ".mp3"))
           for c in script["clips"]}
    text = {c["id"]: c["text"] for c in script["clips"]}

    cues = []       # (clip_id, start_seconds) -> drives the voice track
    phrases = []    # per-phrase visual cue times

    t = INTRO_LEAD
    cues.append(("intro", t))
    intro_end = t + dur["intro"]
    # The list fades in under the tail of the hook line rather than after it,
    # which keeps the opening tight without clipping the voice.
    list_in = intro_end - 0.70
    t = intro_end + INTRO_TAIL

    for i in range(1, 11):
        en, de = f"en{i:02d}", f"de{i:02d}"
        en_start = t
        cues.append((en, en_start))
        de_start = en_start + dur[en] + EN_TO_DE_GAP
        cues.append((de, de_start))
        end = de_start + dur[de]
        phrases.append({
            "n": i,
            "en_text": text[en],
            "de_text": text[de],
            "en_start": en_start,
            "en_end": en_start + dur[en],
            "de_start": de_start,
            "de_end": end,
            "end": end,
        })
        t = end + REPEAT_GAP

    phrases_end = t - REPEAT_GAP

    t = phrases_end + OUTRO_LEAD
    outro1_start = t
    cues.append(("outro1", t))
    t += dur["outro1"] + OUTRO_MID
    outro2_start = t
    cues.append(("outro2", t))
    t += dur["outro2"] + OUTRO_HOLD

    total = t
    timeline = {
        "fps": FPS,
        "duration": total,
        "n_frames": int(round(total * FPS)),
        "intro": {"start": 0.0, "vo_start": INTRO_LEAD, "vo_end": intro_end,
                  "list_in": list_in},
        "phrases": phrases,
        "phrases_end": phrases_end,
        "outro": {"start": phrases_end, "line1": outro1_start,
                  "line2": outro2_start, "end": total},
        "durations": dur,
    }
    return timeline, cues


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tl, cues = build(root)
    print(f"duration {tl['duration']:.2f}s  frames {tl['n_frames']}")
    print(f"intro vo {tl['intro']['vo_start']:.2f}-{tl['intro']['vo_end']:.2f}, "
          f"list in {tl['intro']['list_in']:.2f}")
    for p in tl["phrases"]:
        print(f"  {p['n']:2d} EN {p['en_start']:6.2f}-{p['en_end']:6.2f}  "
              f"DE {p['de_start']:6.2f}-{p['de_end']:6.2f}   {p['de_text']}")
    print(f"outro {tl['outro']['line1']:.2f} / {tl['outro']['line2']:.2f} "
          f"-> {tl['outro']['end']:.2f}")
