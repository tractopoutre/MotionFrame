#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v trunk >/dev/null; then
  cargo install --locked trunk --version 0.21.7
fi
if ! command -v wasm-pack >/dev/null; then
  cargo install --locked wasm-pack --version 0.14.0
fi

bash scripts/build-web-worker.sh

(cd crates/motionframe-web && env -u NO_COLOR trunk build --release)

# Stage THIRD-PARTY-NOTICES.md into dist
cp THIRD-PARTY-NOTICES.md crates/motionframe-web/dist/

echo "build-web.sh: dist at crates/motionframe-web/dist/"
