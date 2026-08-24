#!/bin/bash
# Italy re-edit pipeline. Usage: ./render.sh <source.mp4> <outdir>
# Requires: ffmpeg, python3 with pillow + numpy + scipy, and the fonts in ./fonts.
set -euo pipefail
SRC="${1:?source video}"; OUT="${2:-out}"; HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"; cd "$OUT"

echo "[1/5] fonts"
mkdir -p fonts
for spec in "Anton:anton.ttf" "Montserrat:wght@900:montserrat900.ttf" \
            "Montserrat:wght@700:montserrat700.ttf" "Inter:wght@600:inter600.ttf"; do
  fam="${spec%:*}"; file="${spec##*:}"
  [ -s "fonts/$file" ] && continue
  url=$(curl -s -A "Mozilla/5.0" "https://fonts.googleapis.com/css2?family=$fam&display=swap" \
        | grep -oP "src: url\(\K[^)]+" | head -1)
  curl -sL -o "fonts/$file" "$url"
done

echo "[2/5] base grade + camera"
ffmpeg -v warning -stats -i "$SRC" -vf "$(python3 "$HERE/build/make_base.py")" \
  -an -c:v libx264 -preset medium -crf 17 -pix_fmt yuv420p base.mp4 -y

echo "[3/5] sound design"
python3 "$HERE/build/sfx.py"

echo "[4/5] audio enhance + mix"
ENH="highpass=f=75,equalizer=f=300:t=q:w=1.1:g=-2.0,equalizer=f=1000:t=q:w=1.4:g=-1.0,\
equalizer=f=3400:t=q:w=1.0:g=2.6,equalizer=f=6800:t=q:w=1.2:g=1.8,lowpass=f=15800,\
acompressor=threshold=-18dB:ratio=2.6:attack=8:release=180:makeup=2,aresample=48000"
ffmpeg -v error -i "$SRC" -i sfx.wav -filter_complex \
 "[0:a]$ENH[v];[1:a]volume=0.90[s];[v][s]amix=inputs=2:duration=first:normalize=0[m];\
  [m]alimiter=level_in=1:level_out=0.97:limit=0.97:attack=4:release=60,\
  loudnorm=I=-14:TP=-1.0:LRA=11[o]" -map "[o]" -c:a pcm_s16le -ar 48000 -ac 2 audio_final.wav -y

echo "[5/5] graphics overlay + encode"
python3 "$HERE/build/overlay.py" | ffmpeg -v warning -stats \
  -i base.mp4 \
  -f rawvideo -pix_fmt rgba -s 1080x1920 -framerate 30 -thread_queue_size 512 -i pipe:0 \
  -i audio_final.wav \
  -filter_complex "[0:v][1:v]overlay=0:0:format=auto:shortest=1[v]" \
  -map "[v]" -map 2:a -c:v libx264 -preset slow -b:v 2750k -g 60 \
  -profile:v high -level 4.2 -pix_fmt yuv420p -c:a aac -b:a 128k -ar 48000 \
  -movflags +faststart REEDIT.mp4 -y
echo "done -> $OUT/REEDIT.mp4"
