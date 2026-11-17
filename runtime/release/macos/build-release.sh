#!/bin/sh
# Build, sign, package and notarize 2ndPressing for macOS.
#
#   runtime/release/macos/build-release.sh            # full release
#   NOTARIZE=0 runtime/release/macos/build-release.sh # sign and package only
#
# Output: runtime/target/release-macos/2ndPressing-<version>.pkg
#
# Needs: Rust targets aarch64-apple-darwin and x86_64-apple-darwin, CMake,
# Xcode command-line tools, the "Developer ID Application" and "Developer ID
# Installer" certificates in the login keychain, and (for notarization) a
# notarytool keychain profile, by default "2ndPressing-notary":
#   xcrun notarytool store-credentials 2ndPressing-notary --apple-id ... --team-id ... --password ...
set -eu
cd "$(dirname "$0")/../.."          # runtime/
RT=$(pwd)
HERE="$RT/release/macos"
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
NAME="2ndPressing Declicker"
OUT="$RT/target/release-macos"
NOTARIZE="${NOTARIZE:-1}"
PROFILE="${NOTARY_PROFILE:-2ndPressing-notary}"

ident() { security find-identity -v -p "$1" | sed -n "s/.*\"\($2[^\"]*\)\".*/\1/p" | head -1; }
SIGN_APP=$(ident codesigning "Developer ID Application")
SIGN_PKG=$(security find-identity -v | sed -n 's/.*"\(Developer ID Installer[^"]*\)".*/\1/p' | head -1)
[ -n "$SIGN_APP" ] || { echo "No 'Developer ID Application' certificate found"; exit 1; }
[ -n "$SIGN_PKG" ] || { echo "No 'Developer ID Installer' certificate found"; exit 1; }
echo "== 2ndPressing $VERSION"
echo "   app/plugins: $SIGN_APP"
echo "   installer:   $SIGN_PKG"

rm -rf "$OUT"; mkdir -p "$OUT/stage" "$OUT/pkgs"

# ---- 1. Universal builds ----
echo "== Building plugins (universal)"
cargo xtask bundle-universal pressing-plugin --release >/dev/null
echo "== Building app (universal)"
for t in aarch64-apple-darwin x86_64-apple-darwin; do
  cargo build -p pressing-app --release --target "$t" >/dev/null 2>&1
done
lipo -create -output "$OUT/2ndpressing" \
  target/aarch64-apple-darwin/release/2ndpressing target/x86_64-apple-darwin/release/2ndpressing
echo "== Building Audio Unit (universal)"
cmake -S au -B au/build-universal -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" >/dev/null
cmake --build au/build-universal --config Release >/dev/null
COMPONENT=$(find au/build-universal -name "$NAME.component" -type d -maxdepth 4 | head -1)
[ -n "$COMPONENT" ] || { echo "AU component not found"; exit 1; }

# ---- 2. Stage bundles with release identifiers and versions ----
mkdir -p "$OUT/stage/app" "$OUT/stage/clap" "$OUT/stage/vst3" "$OUT/stage/au"
cp -R "target/bundled/$NAME.clap" "$OUT/stage/clap/"
cp -R "target/bundled/$NAME.vst3" "$OUT/stage/vst3/"
cp -R "$COMPONENT" "$OUT/stage/au/"
# The AU's Info.plist must carry AudioComponents (type/subtype/manufacturer)
# or macOS never registers it. clap-wrapper copies its generated plist into
# the bundle only on a fresh link, and CMake overwrites it with a default on
# every configure, so always install the generated one explicitly.
cp au/build-universal/pressing_auv2-build-helper-output/auv2_Info.plist "$OUT/stage/au/$NAME.component/Contents/Info.plist"
[ "$(plutil -extract AudioComponents.0.subtype raw "$OUT/stage/au/$NAME.component/Contents/Info.plist" 2>/dev/null)" = "Dclk" ] \
  || { echo "AU Info.plist has no AudioComponents entry"; exit 1; }
setplist() { /usr/libexec/PlistBuddy -c "Set :$2 $3" "$1/Contents/Info.plist" 2>/dev/null || /usr/libexec/PlistBuddy -c "Add :$2 string $3" "$1/Contents/Info.plist"; }
for b in "$OUT/stage/clap/$NAME.clap:clap" "$OUT/stage/vst3/$NAME.vst3:vst3"; do
  p=${b%:*}; kind=${b##*:}
  setplist "$p" CFBundleIdentifier "com.secondpressing.declicker.$kind"
  setplist "$p" CFBundleShortVersionString "$VERSION"
  setplist "$p" CFBundleVersion "$VERSION"
  setplist "$p" NSHumanReadableCopyright "2ndPressing, GPL-3.0-or-later"
done

APP="$OUT/stage/app/2ndPressing.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$OUT/2ndpressing" "$APP/Contents/MacOS/2ndpressing"
cp crates/pressing-app/macos/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>2ndPressing</string>
  <key>CFBundleDisplayName</key><string>2ndPressing</string>
  <key>CFBundleIdentifier</key><string>com.secondpressing.app</string>
  <key>CFBundleExecutable</key><string>2ndpressing</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleVersion</key><string>${VERSION}</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSApplicationCategoryType</key><string>public.app-category.music</string>
  <key>NSHumanReadableCopyright</key><string>2ndPressing, GPL-3.0-or-later</string>
  <key>NSMicrophoneUsageDescription</key>
  <string>2ndPressing listens to your audio interface (your turntable) to remove clicks and pops as you play.</string>
</dict>
</plist>
PLIST

# ---- 3. Sign (hardened runtime, secure timestamp) ----
echo "== Signing"
sign() { codesign --force --timestamp --options runtime --sign "$SIGN_APP" "$@"; }
sign "$OUT/stage/clap/$NAME.clap"
sign "$OUT/stage/vst3/$NAME.vst3"
sign "$OUT/stage/au/$NAME.component"
sign --entitlements "$HERE/app.entitlements" "$APP"
for b in "$OUT/stage/clap/$NAME.clap" "$OUT/stage/vst3/$NAME.vst3" "$OUT/stage/au/$NAME.component" "$APP"; do
  codesign --verify --strict --deep "$b"
  archs=$(lipo -archs "$(find "$b/Contents/MacOS" -type f | head -1)")
  case "$archs" in *x86_64*arm64*|*arm64*x86_64*) ;; *) echo "not universal ($archs): $b"; exit 1;; esac
done

# ---- 4. Component packages ----
echo "== Packaging"
comp() {  # stage-dir install-location identifier pkg [scripts]
  plist="$OUT/$1.plist"
  pkgbuild --analyze --root "$OUT/stage/$1" "$plist" >/dev/null
  i=0
  while /usr/libexec/PlistBuddy -c "Print :$i" "$plist" >/dev/null 2>&1; do
    /usr/libexec/PlistBuddy -c "Set :$i:BundleIsRelocatable false" "$plist" 2>/dev/null \
      || /usr/libexec/PlistBuddy -c "Add :$i:BundleIsRelocatable bool false" "$plist"
    i=$((i + 1))
  done
  if [ $# -ge 5 ]; then
    pkgbuild --root "$OUT/stage/$1" --install-location "$2" --identifier "$3" --version "$VERSION" \
      --component-plist "$plist" --scripts "$5" "$OUT/pkgs/$4" >/dev/null
  else
    pkgbuild --root "$OUT/stage/$1" --install-location "$2" --identifier "$3" --version "$VERSION" \
      --component-plist "$plist" "$OUT/pkgs/$4" >/dev/null
  fi
}
mkdir -p "$OUT/scripts-au"
cat > "$OUT/scripts-au/postinstall" <<'SH'
#!/bin/sh
# Make the new Audio Unit visible without a restart.
killall -9 AudioComponentRegistrar 2>/dev/null || true
exit 0
SH
chmod +x "$OUT/scripts-au/postinstall"
comp app  /Applications                      com.secondpressing.pkg.app  app.pkg
comp clap /Library/Audio/Plug-Ins/CLAP       com.secondpressing.pkg.clap clap.pkg
comp vst3 /Library/Audio/Plug-Ins/VST3       com.secondpressing.pkg.vst3 vst3.pkg
comp au   /Library/Audio/Plug-Ins/Components com.secondpressing.pkg.au   au.pkg "$OUT/scripts-au"

mkdir -p "$OUT/resources"
cp "$HERE/resources/"*.html "$OUT/resources/"
cp "$RT/../LICENSE" "$OUT/resources/LICENSE.txt"
cat > "$OUT/distribution.xml" <<XML
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
  <title>2ndPressing Declicker ${VERSION}</title>
  <welcome file="welcome.html" mime-type="text/html"/>
  <license file="LICENSE.txt" mime-type="text/plain"/>
  <conclusion file="conclusion.html" mime-type="text/html"/>
  <options customize="allow" require-scripts="false" hostArchitectures="arm64,x86_64"/>
  <domains enable_localSystem="true"/>
  <os-version min="11.0"/>
  <choices-outline>
    <line choice="app"/>
    <line choice="au"/>
    <line choice="vst3"/>
    <line choice="clap"/>
  </choices-outline>
  <choice id="app" title="2ndPressing app" description="The standalone app, in Applications."><pkg-ref id="com.secondpressing.pkg.app"/></choice>
  <choice id="au" title="Audio Unit" description="For Logic Pro, GarageBand and other AU hosts. Installs the CLAP plugin too, which the Audio Unit loads."><pkg-ref id="com.secondpressing.pkg.au"/><pkg-ref id="com.secondpressing.pkg.clap"/></choice>
  <choice id="vst3" title="VST3 plugin" description="For VST3 hosts."><pkg-ref id="com.secondpressing.pkg.vst3"/></choice>
  <choice id="clap" title="CLAP plugin" description="For CLAP hosts such as Bitwig and REAPER."><pkg-ref id="com.secondpressing.pkg.clap"/></choice>
  <pkg-ref id="com.secondpressing.pkg.app" version="${VERSION}">app.pkg</pkg-ref>
  <pkg-ref id="com.secondpressing.pkg.au" version="${VERSION}">au.pkg</pkg-ref>
  <pkg-ref id="com.secondpressing.pkg.vst3" version="${VERSION}">vst3.pkg</pkg-ref>
  <pkg-ref id="com.secondpressing.pkg.clap" version="${VERSION}">clap.pkg</pkg-ref>
</installer-gui-script>
XML
PKG="$OUT/2ndPressing-$VERSION.pkg"
productbuild --distribution "$OUT/distribution.xml" --package-path "$OUT/pkgs" --resources "$OUT/resources" \
  --sign "$SIGN_PKG" --timestamp "$PKG" >/dev/null
pkgutil --check-signature "$PKG" | sed -n '1,3p'

# ---- 5. Notarize and staple ----
if [ "$NOTARIZE" = "1" ]; then
  if xcrun notarytool history --keychain-profile "$PROFILE" >/dev/null 2>&1; then
    echo "== Notarizing (a few minutes)"
    xcrun notarytool submit "$PKG" --keychain-profile "$PROFILE" --wait
    xcrun stapler staple "$PKG"
    spctl --assess --type install -vv "$PKG"
  else
    echo "== Not notarized: no notarytool profile '$PROFILE' (see the header of this script)"
  fi
fi
echo "== Done: $PKG"
