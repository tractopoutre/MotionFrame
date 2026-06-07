#!/usr/bin/env bash
# Build a universal (arm64 + x86_64) macOS .app bundle and wrap it in a .dmg.
#
# The bundle is UNSIGNED and NOT notarized: this project has no Apple Developer
# ID certificate. Gatekeeper will quarantine the download — users open it via
# right-click > Open, or `xattr -dr com.apple.quarantine MotionFrame.app`.
#
# Usage:
#   scripts/package-macos.sh            # build both targets, bundle, dmg
#   SKIP_BUILD=1 scripts/package-macos.sh   # reuse existing target/ binaries
#
# Outputs (in repo root): MotionFrame.app/ and motionframe-macos-universal.dmg
set -euo pipefail

APP_NAME="MotionFrame"
BIN_NAME="motionframe"
IDENTIFIER="net.aki-null.motionframe"
CRATE_MANIFEST="crates/motionframe-desktop/Cargo.toml"
DMG_NAME="motionframe-macos-universal.dmg"
MIN_MACOS="11.0"
DOCS=(README.md LICENSE THIRD-PARTY-LICENSES.md)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Version from the desktop crate's [package] section (first `version = "..."`).
VERSION="$(grep -m1 '^version' "$CRATE_MANIFEST" | sed -E 's/.*"([^"]+)".*/\1/')"
: "${VERSION:?could not parse version from $CRATE_MANIFEST}"

ARM_BIN="target/aarch64-apple-darwin/release/$BIN_NAME"
X86_BIN="target/x86_64-apple-darwin/release/$BIN_NAME"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo ">> building aarch64-apple-darwin"
  cargo build -p motionframe-desktop --release --target aarch64-apple-darwin
  echo ">> building x86_64-apple-darwin"
  cargo build -p motionframe-desktop --release --target x86_64-apple-darwin
fi

for b in "$ARM_BIN" "$X86_BIN"; do
  [[ -f "$b" ]] || { echo "missing binary: $b" >&2; exit 1; }
done

# --- universal binary ---------------------------------------------------------
APP_DIR="$APP_NAME.app"
MACOS_DIR="$APP_DIR/Contents/MacOS"
RES_DIR="$APP_DIR/Contents/Resources"
rm -rf "$APP_DIR"
mkdir -p "$MACOS_DIR" "$RES_DIR"

echo ">> lipo -> universal binary"
lipo -create -output "$MACOS_DIR/$BIN_NAME" "$ARM_BIN" "$X86_BIN"
chmod +x "$MACOS_DIR/$BIN_NAME"
lipo -info "$MACOS_DIR/$BIN_NAME"

# --- Info.plist (no CFBundleIconFile: no icon asset yet) ----------------------
cat > "$APP_DIR/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleName</key>
	<string>$APP_NAME</string>
	<key>CFBundleDisplayName</key>
	<string>$APP_NAME</string>
	<key>CFBundleExecutable</key>
	<string>$BIN_NAME</string>
	<key>CFBundleIdentifier</key>
	<string>$IDENTIFIER</string>
	<key>CFBundleVersion</key>
	<string>$VERSION</string>
	<key>CFBundleShortVersionString</key>
	<string>$VERSION</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>LSMinimumSystemVersion</key>
	<string>$MIN_MACOS</string>
	<key>LSApplicationCategoryType</key>
	<string>public.app-category.graphics-design</string>
	<key>NSHighResolutionCapable</key>
	<true/>
</dict>
</plist>
PLIST

echo "APPL????" > "$APP_DIR/Contents/PkgInfo"

# --- dmg ----------------------------------------------------------------------
echo ">> staging dmg"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
cp -R "$APP_DIR" "$STAGING/"
ln -s /Applications "$STAGING/Applications"
for d in "${DOCS[@]}"; do
  [[ -f "$d" ]] && cp "$d" "$STAGING/"
done

echo ">> hdiutil create $DMG_NAME"
rm -f "$DMG_NAME"
hdiutil create -volname "$APP_NAME" -srcfolder "$STAGING" -ov -format UDZO "$DMG_NAME" >/dev/null

echo ">> done: $APP_DIR and $DMG_NAME (version $VERSION, unsigned)"
