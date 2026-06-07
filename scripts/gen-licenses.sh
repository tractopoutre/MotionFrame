#!/usr/bin/env bash
# Regenerate THIRD-PARTY-LICENSES.md from two sources:
#   1. about-preamble.md   — hand-written notices (bundled font, ported OpenCV).
#   2. cargo-about output  — license text for every linked third-party crate,
#                            rendered through about.hbs (the `## Linked
#                            Libraries` section). First-party workspace crates
#                            are excluded via `[private] ignore` in about.toml.
#
# The desktop and web build.rs embed the resulting file into the binaries, so
# run this whenever dependencies change and commit the updated output.
#
# Usage: scripts/gen-licenses.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="THIRD-PARTY-LICENSES.md"
PREAMBLE="about-preamble.md"

if ! command -v cargo-about >/dev/null 2>&1; then
  echo "cargo-about not found. Install it with:" >&2
  echo "  cargo install --locked cargo-about" >&2
  exit 1
fi

[[ -f "$PREAMBLE" ]] || { echo "missing $PREAMBLE" >&2; exit 1; }

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

cat "$PREAMBLE" > "$tmp"
cargo about generate about.hbs >> "$tmp"

mv "$tmp" "$OUT"
trap - EXIT
echo "wrote $OUT ($(grep -c '^### ' "$OUT") third-party crates)"
