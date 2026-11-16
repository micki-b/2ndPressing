#!/bin/sh
# Build 2ndPressing.app (macOS): release binary + Info.plist + ad-hoc signature.
#   crates/pressing-app/macos/bundle.sh            -> target/release/2ndPressing.app
# The Info.plist carries NSMicrophoneUsageDescription, so macOS asks for
# audio-input permission for the app itself (not for Terminal). With an
# ad-hoc signature macOS may ask again after each rebuild.
set -eu
cd "$(dirname "$0")/../../.."
cargo build -p pressing-app --release
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
APP=target/release/2ndPressing.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp target/release/2ndpressing "$APP/Contents/MacOS/2ndpressing"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>2ndPressing</string>
  <key>CFBundleDisplayName</key><string>2ndPressing Declicker</string>
  <key>CFBundleIdentifier</key><string>org.2ndpressing.app</string>
  <key>CFBundleExecutable</key><string>2ndpressing</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleVersion</key><string>${VERSION}</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSApplicationCategoryType</key><string>public.app-category.music</string>
  <key>NSMicrophoneUsageDescription</key>
  <string>2ndPressing listens to your audio interface (your turntable) to remove clicks and pops as you play.</string>
</dict>
</plist>
PLIST
codesign --force --sign - "$APP" >/dev/null
echo "built $APP"
