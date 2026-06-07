#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Force software adapter for wgpu so tests pass in headless / non-GPU environments.
export WGPU_FORCE_FALLBACK_ADAPTER=1

echo "[1/6] cargo fmt --check"
cargo fmt --all -- --check

echo "[2/6] cargo clippy"
cargo clippy --workspace --all-targets --all-features -- -D warnings

echo "[3/6] cargo test"
cargo test --workspace --all-features

echo "[4/6] cargo build --release"
cargo build --workspace --release

echo "[5/6] build web target"
./scripts/build-web.sh

echo "[6/6] cargo deny check"
if ! command -v cargo-deny >/dev/null 2>&1; then
  echo "cargo-deny is required. Install it with: cargo install --locked cargo-deny" >&2
  exit 1
fi
cargo deny --all-features check

echo "verify.sh: all checks passed"
