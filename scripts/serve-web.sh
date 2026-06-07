#!/usr/bin/env bash
# Serve the web build locally with the headers wasm-bindgen-rayon needs.
#
# `trunk serve` is the only supported way to run the browser app on localhost:
# the bundle uses SharedArrayBuffer, which requires a cross-origin isolated
# page (COOP: same-origin + COEP: require-corp). Plain `python3 -m http.server`
# will *not* work — the worker will fail at `initThreadPool` with a
# DataCloneError. Headers and port are pinned in crates/motionframe-web/Trunk.toml.
set -euo pipefail

cd "$(dirname "$0")/../crates/motionframe-web"
exec env -u NO_COLOR trunk serve --release "$@"
