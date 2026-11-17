#!/bin/sh
# Cross-build the Windows (x64) release on macOS and package it with NSIS.
#
#   runtime/release/windows/build-release.sh
#
# Output: runtime/target/release-windows/2ndPressing-<version>-windows-x64-setup.exe
#
# Needs (one-time):
#   rustup target add x86_64-pc-windows-msvc
#   cargo install --locked cargo-xwin     # MSVC CRT + Windows SDK via xwin
#   brew install llvm makensis            # clang-cl / lld-link, and NSIS
# cargo-xwin downloads the Microsoft CRT and Windows SDK on first use; that
# requires accepting Microsoft's licence for them (XWIN_ACCEPT_LICENSE=1).
#
# Not code-signed: Windows SmartScreen will warn until it is signed.
set -eu
cd "$(dirname "$0")/../.."          # runtime/
RT=$(pwd)
HERE="$RT/release/windows"
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
NAME="2ndPressing Declicker"
T=x86_64-pc-windows-msvc
OUT="$RT/target/release-windows"
export PATH="/opt/homebrew/opt/llvm/bin:/opt/homebrew/opt/lld/bin:$PATH"

echo "== 2ndPressing $VERSION for Windows x64"
rm -rf "$OUT"; mkdir -p "$OUT/stage/app" "$OUT/stage/clap" "$OUT/stage/vst3/$NAME.vst3/Contents/x86_64-win"

echo "== Building plugin and app"
cargo xwin build -p pressing-plugin --release --target "$T" --lib
cargo xwin build -p pressing-app --release --target "$T"

DLL="target/$T/release/pressing_plugin.dll"
EXE="target/$T/release/2ndpressing.exe"
[ -f "$DLL" ] && [ -f "$EXE" ] || { echo "build outputs missing"; exit 1; }

# VST3: bundle folder with the DLL inside; CLAP: the DLL renamed. One DLL
# exports both entry points (nih-plug).
cp "$DLL" "$OUT/stage/vst3/$NAME.vst3/Contents/x86_64-win/$NAME.vst3"
cp "$DLL" "$OUT/stage/clap/$NAME.clap"
cp "$EXE" "$OUT/stage/app/2ndPressing.exe"
cp "$HERE/2ndPressing.ico" "$OUT/stage/2ndPressing.ico"
# Windows line endings so Notepad and the installer show the licence cleanly.
sed 's/$/\r/' "$RT/../LICENSE" > "$OUT/stage/LICENSE.txt"

echo "== Packaging (NSIS)"
SETUP="$OUT/2ndPressing-$VERSION-windows-x64-setup.exe"
makensis -V2 -DVERSION="$VERSION" -DSTAGE="$OUT/stage" -DOUTFILE="$SETUP" "$HERE/installer.nsi"
shasum -a 256 "$SETUP"
echo "== Done: $SETUP"
