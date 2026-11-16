#!/bin/sh
# Render public/assets/og.png and apple-touch-icon.png from tools/*.html
# with headless Chrome (macOS path). Run from the website/ directory.
set -eu
C="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
python3 -m http.server 8766 --bind 127.0.0.1 >/dev/null 2>&1 &
PID=$!
trap 'kill $PID' EXIT
sleep 1
"$C" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=1 --window-size=1200,630 \
  --virtual-time-budget=3000 --screenshot=public/assets/og.png http://127.0.0.1:8766/tools/og.html 2>/dev/null
"$C" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=1 --window-size=180,180 \
  --virtual-time-budget=2000 --screenshot=public/assets/apple-touch-icon.png http://127.0.0.1:8766/tools/icon.html 2>/dev/null
echo "rendered og.png and apple-touch-icon.png"
