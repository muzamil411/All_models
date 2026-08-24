#!/bin/bash
cd /tmp/claude-0/-home-user-All-models/00850189-800c-554a-aff4-b363b3e732a4/scratchpad/jp
TOTAL=2175; NW=4; CH=$(( (TOTAL + NW - 1) / NW ))
rm -f seg_*.mp4 concat.txt
for i in $(seq 0 $((NW-1))); do
  LO=$(( i * CH )); HI=$(( LO + CH )); [ $HI -gt $TOTAL ] && HI=$TOTAL
  [ $LO -ge $TOTAL ] && continue
  ( python3 render.py $LO $HI | ffmpeg -hide_banner -v error \
      -f rawvideo -pix_fmt rgb24 -s 1080x1920 -framerate 30 -i pipe:0 \
      -c:v libx264 -preset veryfast -crf 15 -pix_fmt yuv420p -g 30 seg_$i.mp4 -y ) &
done
wait
for i in $(seq 0 $((NW-1))); do [ -f seg_$i.mp4 ] && echo "file 'seg_$i.mp4'" >> concat.txt; done
echo "SEGMENTS_DONE"
ffmpeg -hide_banner -v error -f concat -safe 0 -i concat.txt -c copy joined.mp4 -y
echo "JOINED=$?"
ffmpeg -hide_banner -v error -i score.wav -af "loudnorm=I=-14:TP=-1.0:LRA=11,aresample=48000" \
  -c:a pcm_s16le audio_jp.wav -y
echo "AUDIO=$?"
ffmpeg -hide_banner -v error -i joined.mp4 -i audio_jp.wav \
  -vf "eq=contrast=1.04:saturation=1.06,vignette=angle=PI/5.2:mode=forward,noise=alls=7:allf=t+u,format=yuv420p" \
  -map 0:v -map 1:a -c:v libx264 -preset slow -b:v 2700k -g 60 -profile:v main -level 4.0 \
  -refs 2 -bf 2 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
  -pass 1 -passlogfile jp -an -f null - -y
ffmpeg -hide_banner -v error -i joined.mp4 -i audio_jp.wav \
  -vf "eq=contrast=1.04:saturation=1.06,vignette=angle=PI/5.2:mode=forward,noise=alls=7:allf=t+u,format=yuv420p" \
  -map 0:v -map 1:a -c:v libx264 -preset slow -b:v 2700k -g 60 -profile:v main -level 4.0 \
  -refs 2 -bf 2 -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
  -pass 2 -passlogfile jp -c:a aac -b:a 160k -ar 44100 -ac 2 -movflags +faststart \
  JAPAN_TruthSeeker.mp4 -y
echo "FINAL=$?"
ls -la JAPAN_TruthSeeker.mp4
echo JPDONE
