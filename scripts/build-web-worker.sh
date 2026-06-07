#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Make sure cargo's bin dir is on PATH — `cargo install` puts wasm-pack and
# wasm-bindgen-cli there, and `wasm-pack --mode no-install` (below) relies on
# finding wasm-bindgen via PATH. Standard rustup setups already do this, but
# CI shells and minimal environments may not.
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"

WASM_PACK_VERSION="0.14.0"
if [ "$(wasm-pack --version 2>/dev/null | awk '{print $2}')" != "$WASM_PACK_VERSION" ]; then
  cargo install --locked wasm-pack --version "$WASM_PACK_VERSION"
fi

# Pre-install wasm-bindgen-cli with default RUSTFLAGS — without this, wasm-pack
# auto-installs it under the +simd128/+atomics flags we set below for the wasm
# build, and the native install fails because those features don't exist on
# the host target. Pin the cli version to match the `wasm-bindgen` dep so
# wasm-bindgen's CLI/runtime check passes.
WASM_BINDGEN_VERSION="0.2.121"
if [ "$(wasm-bindgen --version 2>/dev/null | awk '{print $2}')" != "$WASM_BINDGEN_VERSION" ]; then
  cargo install --locked wasm-bindgen-cli --version "$WASM_BINDGEN_VERSION"
fi

# Rebuild std with atomics for wasm-bindgen-rayon (requires nightly + rust-src,
# pinned in crates/motionframe-web-worker/rust-toolchain.toml). The +atomics
# target-feature alone does not currently auto-pass --shared-memory to wasm-ld
# in this nightly, so we forward the linker args explicitly. wasm-bindgen's
# threading post-processing requires the four __wasm_init_tls / __tls_*
# symbols to be exported (verified by trial: dropping any of them fails at
# `wasm-bindgen` step with "failed to find __tls_<name>").
# `+atomics` is unstable in rustc, which emits a multi-line warning on every
# compilation. The diagnostic isn't tied to a lint name so `#[allow(...)]`
# can't suppress it. Filter the warning header, its `|` border line, the
# `= note:` continuation, and cargo's per-crate "generated 1 warning" rollups.
# Anything else (real warnings, errors) passes through untouched.
RUSTFLAGS="-C target-feature=+simd128,+bulk-memory,+mutable-globals,+atomics \
  -C link-arg=--shared-memory \
  -C link-arg=--import-memory \
  -C link-arg=--max-memory=4294967296 \
  -C link-arg=--export=__wasm_init_tls \
  -C link-arg=--export=__tls_size \
  -C link-arg=--export=__tls_align \
  -C link-arg=--export=__tls_base" \
  wasm-pack build --release --target web --mode no-install crates/motionframe-web-worker -- \
  -Z build-std=panic_abort,std \
  2> >(awk '
    /^warning: unstable feature specified for .-Ctarget-feature./ { skip=2; next }
    skip > 0 { skip--; next }
    /^warning: .motionframe-(engine|web-worker). \(lib\) generated 1 warning/ { next }
    { print }
  ' >&2)

mkdir -p crates/motionframe-web/static/worker
cp crates/motionframe-web-worker/pkg/motionframe_web_worker.js \
   crates/motionframe-web-worker/pkg/motionframe_web_worker_bg.wasm \
   crates/motionframe-web/static/worker/

# wasm-bindgen-rayon emits a workerHelpers.js (and any other JS snippets)
# under pkg/snippets/. Child workers spawned by initThreadPool fetch it
# relative to the main worker URL, so it must ship at static/worker/snippets/.
rm -rf crates/motionframe-web/static/worker/snippets
if [ -d crates/motionframe-web-worker/pkg/snippets ]; then
  cp -R crates/motionframe-web-worker/pkg/snippets \
        crates/motionframe-web/static/worker/snippets

  # wasm-bindgen-rayon's workerHelpers.js does `import('../../..')` to load
  # the pkg entry — that relies on a bundler to resolve the directory via
  # package.json `main`. Native ES modules can't, and the static server
  # returns the directory listing as HTML, which fails MIME-type checks.
  # Rewrite to the explicit module URL. perl -i is portable across BSD/GNU.
  patched=0
  while IFS= read -r f; do
    before=$(grep -c "'\.\./\.\./\.\.'" "$f" || true)
    [ "$before" -gt 0 ] || continue
    perl -i -pe "s|'\.\./\.\./\.\.'|'../../../motionframe_web_worker.js'|g" "$f"
    patched=$((patched + before))
  done < <(find crates/motionframe-web/static/worker/snippets -name workerHelpers.js)
  if [ "$patched" -eq 0 ]; then
    echo "build-web-worker.sh: WARNING — no '../../..' import found in workerHelpers.js." >&2
    echo "  wasm-bindgen-rayon may have changed its emitted glue; verify thread-pool init." >&2
    exit 1
  fi
fi

# Hand-written glue: `worker.js` that initializes the rayon thread pool, then
# wires onmessage into wasm. Pool init is async — main-thread postMessage is
# deferred behind `await ready`.
cat > crates/motionframe-web/static/worker/worker.js <<'JS'
import init, { initThreadPool, handle_message } from './motionframe_web_worker.js';
const ready = (async () => {
  await init();
  await initThreadPool(navigator.hardwareConcurrency);
})();
self.onmessage = async (ev) => {
  await ready;
  handle_message(ev.data);
};
JS
