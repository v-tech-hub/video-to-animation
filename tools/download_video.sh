#!/usr/bin/env bash
set -euo pipefail

URL="${1:?Usage: tools/download_video.sh <video-url> [output-template]}"
OUTPUT="${2:-input.%(ext)s}"

if command -v yt-dlp >/dev/null 2>&1; then
  YTDLP=(yt-dlp)
elif [[ -x "./tools/yt-dlp" ]]; then
  YTDLP=(./tools/yt-dlp)
else
  echo "yt-dlp not found. Install it or place the standalone binary at tools/yt-dlp." >&2
  exit 127
fi

exec "${YTDLP[@]}" \
  --no-playlist \
  --cookies-from-browser chrome \
  -f 'bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b' \
  --merge-output-format mp4 \
  -o "$OUTPUT" \
  "$URL"
