#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
MANIM=${MANIM:-manim}
Q=${1:--qh}
"$MANIM" "$Q" --disable_caching --media_dir media section2.py Section2Assurance
if [ "$Q" = "-qh" ]; then
  mkdir -p ../videos
  ffmpeg -v error -y -i media/videos/section2/1080p60/Section2Assurance.mp4 -c copy -movflags +faststart \
    ../videos/section2_assurance.mp4
  ffmpeg -v error -y -ss 29.0 -i ../videos/section2_assurance.mp4 -frames:v 1 ../videos/section2_assurance_poster.png
fi
