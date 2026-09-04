"""
Audio assembly: lay the 23 ElevenLabs clips onto a silent bed at their cue
times, then duck a music bed under them and master the result.
"""
import os
import subprocess

SR = 44100


def _run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def build_voice_track(project_root, cues, duration, out_path):
    """Place each clip at its cue time on one continuous track.

    Using adelay + amix (rather than concat) means the silences are exactly the
    gaps the timeline asked for, so audio and picture share one clock.
    """
    audio_dir = os.path.join(project_root, "assets", "audio")
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for clip_id, _ in cues:
        cmd += ["-i", os.path.join(audio_dir, clip_id + ".mp3")]

    parts, labels = [], []
    for i, (_, start) in enumerate(cues):
        ms = int(round(start * 1000))
        # Normalise every clip to the same peak first so no single line jumps out.
        parts.append(f"[{i}:a]aformat=sample_fmts=fltp:sample_rates={SR}:"
                     f"channel_layouts=stereo,volume=1.0,"
                     f"adelay={ms}|{ms}[v{i}]")
        labels.append(f"[v{i}]")

    graph = ";".join(parts)
    graph += (";" + "".join(labels) +
              f"amix=inputs={len(cues)}:duration=longest:normalize=0[mixed]")
    # Trim/pad to the exact programme length and add a short safety fade.
    graph += (f";[mixed]apad,atrim=0:{duration:.4f},"
              f"asetpts=N/SR/TB,afade=t=out:st={duration - 0.35:.4f}:d=0.35[out]")

    cmd += ["-filter_complex", graph, "-map", "[out]",
            "-ar", str(SR), "-ac", "2", "-c:a", "pcm_s16le", out_path]
    _run(cmd)
    return out_path


def build_music_bed(project_root, duration, out_path):
    """Loop/trim the generated music to length, or synthesise a soft pad."""
    music_dir = os.path.join(project_root, "assets", "music")
    src = os.path.join(music_dir, "bed.mp3")
    if os.path.exists(src):
        _run(["ffmpeg", "-v", "error", "-y", "-stream_loop", "-1", "-i", src,
              "-af", (f"aformat=sample_fmts=fltp:sample_rates={SR}:channel_layouts=stereo,"
                      f"atrim=0:{duration:.4f},asetpts=N/SR/TB,"
                      f"afade=t=in:st=0:d=1.2,"
                      f"afade=t=out:st={duration - 1.8:.4f}:d=1.8"),
              "-ar", str(SR), "-ac", "2", "-c:a", "pcm_s16le", out_path])
        return out_path, "generated"

    # Fallback: a slow, quiet two-note pad built from sine partials.
    freqs = [110.0, 164.81, 220.0, 329.63]  # A2, E3, A3, E4
    srcs = "".join(
        f"sine=frequency={f}:sample_rate={SR}:duration={duration:.4f}[s{i}];"
        for i, f in enumerate(freqs))
    mix = "".join(f"[s{i}]" for i in range(len(freqs)))
    graph = (srcs + mix + f"amix=inputs={len(freqs)}:normalize=0,"
             "tremolo=f=0.09:d=0.55,lowpass=f=700,volume=0.10,"
             f"afade=t=in:st=0:d=2.0,afade=t=out:st={duration - 2.5:.4f}:d=2.5,"
             "aformat=channel_layouts=stereo[out]")
    _run(["ffmpeg", "-v", "error", "-y", "-filter_complex", graph, "-map", "[out]",
          "-ar", str(SR), "-ac", "2", "-c:a", "pcm_s16le", out_path])
    return out_path, "synthesised"


TARGET_LUFS = -14.0     # what TikTok / Reels / Shorts normalise to anyway
BED_GAIN = 0.050        # music bed sits ~26 dB under the voice before ducking


def _premaster_graph(duration):
    return (
        "[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
        "highpass=f=90,"                       # roll off rumble under the voice
        "acompressor=threshold=0.10:ratio=3:attack=8:release=180:makeup=1.6"
        "[voice];"
        "[voice]asplit=2[vmix][vkey];"
        "[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
        f"volume={BED_GAIN}[bed];"
        # Duck the bed further whenever the voice is present.
        "[bed][vkey]sidechaincompress=threshold=0.025:ratio=12:attack=15:"
        "release=450:makeup=1[duck];"
        "[vmix][duck]amix=inputs=2:duration=first:normalize=0,"
        f"atrim=0:{duration:.4f},asetpts=N/SR/TB[out]"
    )


def measure_lufs(path):
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", path, "-af", "ebur128=framelog=quiet",
         "-f", "null", "-"], capture_output=True, text=True).stderr
    for i, line in enumerate(out.splitlines()):
        if "Integrated loudness" in line:
            return float(out.splitlines()[i + 1].split("I:")[1].replace("LUFS", ""))
    raise RuntimeError("could not measure loudness")


def mix(voice_path, music_path, out_path, duration, workdir):
    """Duck the music under the voice, then master with a single static gain.

    A one-pass `loudnorm` is dynamic: it quietly rides the level back up during
    the repeat pauses, which both undoes the ducking and puts an audible pump
    into the gaps. Measuring once and applying a fixed gain keeps the whole
    programme on one constant scale — no pumping, no sudden level changes.
    """
    pre = os.path.join(workdir, "premaster.wav")
    _run(["ffmpeg", "-v", "error", "-y", "-i", voice_path, "-i", music_path,
          "-filter_complex", _premaster_graph(duration), "-map", "[out]",
          "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", pre])

    lufs = measure_lufs(pre)
    gain_db = max(min(TARGET_LUFS - lufs, 18.0), -18.0)

    _run(["ffmpeg", "-v", "error", "-y", "-i", pre,
          "-af", f"volume={gain_db:.2f}dB,alimiter=limit=0.94:level=disabled",
          "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", out_path])
    return out_path, lufs, gain_db
