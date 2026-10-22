#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
MANIM=${MANIM:-manim}
Q=${1:--qh}
WHICH=${2:-all}

render() {
  "$MANIM" "$Q" --disable_caching --media_dir media "$1.py" "$2"
  if [ "$Q" = "-qh" ]; then
    mkdir -p ../videos
    ffmpeg -v error -y -i "media/videos/$1/1080p60/$2.mp4" -c copy -movflags +faststart "../videos/$3.mp4"
    ffmpeg -v error -y -ss "$4" -i "../videos/$3.mp4" -frames:v 1 "../videos/$3_poster.png"
  fi
}

if [ "$WHICH" = "all" ] || [ "$WHICH" = "section2" ]; then
  render section2 Section2Assurance section2_assurance 29.0
fi
if [ "$WHICH" = "all" ] || [ "$WHICH" = "section4_sequences" ]; then
  render section4_sequences Section4Sequences section4_sequences 92.5
fi
