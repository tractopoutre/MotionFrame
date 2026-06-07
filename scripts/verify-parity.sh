#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f tests/fixtures/explosion00_001_002.flo.gz ]]; then
    echo "Flow parity fixtures missing from tests/fixtures." >&2
    exit 1
fi

if [[ ! -d tests/fixtures/explosion00 ]]; then
    echo "Atlas pipeline fixtures missing from tests/fixtures/explosion00." >&2
    exit 1
fi

echo "=== Flow parity ==="
cargo test -p motionframe-engine --release --test flow_parity -- --nocapture

echo "=== Pipeline parity ==="
cargo test -p motionframe-engine --release --test atlas_pipeline_parity -- --nocapture
