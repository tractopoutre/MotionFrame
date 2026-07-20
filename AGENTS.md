# MotionFrame — AGENTS.md

Agentic coding instructions for the MotionFrame Rust codebase.

## Project Overview

MotionFrame is a Rust tool for generating motion-vector textures for flipbook animation. It analyzes image sequences, computes optical flow, accumulates sub-frame motion, and writes color/motion atlases for smoother playback. The project has a desktop app (egui-based GUI + CLI), a web build (WASM + Trunk), and a shared engine crate.

## Build / Lint / Test Commands

```bash
# Full CI pipeline (fmt + clippy + test + release build + web + license check)
./scripts/verify.sh

# Quick local dev loop (fmt + clippy + test)
./scripts/verify-quick.sh

# OpenCV parity tests (release only, needs tests/fixtures/)
./scripts/verify-parity.sh
```

Individual commands:

```bash
# Format
cargo fmt --all -- --check                        # check only
cargo fmt --all                                   # apply

# Lint (clippy with deny-level warnings)
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Test
cargo test --workspace --all-features             # all tests
cargo test -p motionframe-engine                  # single crate
cargo test -p motionframe-engine -- flow          # filter by test name
cargo test -p motionframe-engine --test flow_parity -- --nocapture  # single integration test
cargo test -p motionframe-engine --lib flow::poly # single module test

# Build
cargo build --release --bin motionframe           # desktop binary
cargo build --workspace --release                 # everything

# Web (requires trunk + wasm-pack)
npm run build:web                                  # package.json wrapper
./scripts/build-web.sh                             # full web build

# License / dependency check (requires cargo-deny)
cargo deny --all-features check

# Run desktop app
cargo run --release --bin motionframe

# Run CLI convert
cargo run --release --bin motionframe -- convert --input <dir> --output <prefix> --output-count 64
```

## Code Style Guidelines

### Imports

Group by source (std → external → crate), alphabetized within groups. Glob imports (e.g. `prelude::*`) last in their group. One `use` per line inside braces.

```rust
use std::path::PathBuf;

use rayon::prelude::*;
use thiserror::Error;

use crate::pipeline::{Flow, ImageF32, PipelineError};
```

### Naming

| Category | Convention | Example |
|---|---|---|
| Types (structs, enums) | `PascalCase` | `ImageF32`, `PipelineError` |
| Traits | `PascalCase` | `FrameSource`, `Platform` |
| Enum variants | `PascalCase` | `Progress::Done` |
| Functions / methods | `snake_case` | `run_pipeline()`, `ensure_size()` |
| Module names | `snake_case` | `flow`, `pipeline` |
| Constants / statics | `SCREAMING_SNAKE_CASE` | `MAX_IMAGE_DIM`, `CANCEL_FLAG` |
| Type parameters | single uppercase | `P: Platform` |
| Acronyms | first letter only | `Rgba8`, `TgaEncoder` |
| Boolean getters | `is_` prefix | `is_empty()`, `is_loop()` |

### Error Handling

Three layers, matched to context:

- **Engine (core library):** `thiserror` derive enum (`PipelineError`) with `#[error("...")]` on every variant and `#[from]` for delegation.
- **CLI (desktop):** Custom `CliError` enum with `Display` + `std::error::Error` manual impls and `exit_code()` method.
- **Boundary layers (worker protocol, FFI):** `Result<_, String>` for thin interfaces.
- **No `anyhow`, `eyre`, or `failure`** anywhere.

### Formatting

- `rustfmt.toml`: edition 2021, max_width 100, use_field_init_shorthand = true.
- `#[allow(clippy::...)]` always includes a `// reason:` comment.
- Doc comments: `//!` for module-level, `///` for items, `//` for internal notes.
- Field init shorthand used per config.
- Long argument lists use one-per-line with `#[allow(clippy::too_many_arguments)]`.
- Math code with known rounding uses explicit `a * b + c` form (not `mul_add`), documented in workspace lints.

### Lint Configuration

- `unsafe_code = "deny"` at workspace level; only FFI calls (Win32 `AttachConsole`, wasm-bindgen glue) are exempted with `#[allow(unsafe_code, reason = "...")]`.
- Clippy groups `all`, `pedantic`, `nursery` enabled at warn level with specific relaxations (cast_*, similar_names, must_use_candidate, module_name_repetitions, suboptimal_flops, missing_errors_doc, missing_panics_doc) allowed.
- `-D warnings` enforced in CI/lint commands.
- MSRV: 1.95.0 (pinned in `rust-toolchain.toml` and `clippy.toml`).

### Testing

- Unit tests: `#[cfg(test)] mod tests { use super::*; }` at bottom of source files.
- Integration tests: `tests/` directory at crate level, e.g. `motionframe-engine/tests/flow_parity.rs`.
- Benchmarks: `benches/` using `criterion` with `harness = false` in Cargo.toml.
- Test naming: `verb_noun_condition` pattern, e.g. `farneback_zero_images_zero_flow`.
- Parity tests against OpenCV reference use `_parity_` suffix.
- Float comparison: explicit epsilon checks (`< 1e-4`), no assertion crates.
- Fixtures required for parity tests live in `tests/fixtures/`.

### Project Structure

```
crates/
  motionframe-engine/   -- Core library: flow, pipeline, io, preview, viz
  motionframe-ui/       -- Shared egui UI: MotionFrameApp, Platform trait, panels
  motionframe-desktop/  -- Binary: main, CliError, DesktopPlatform, clap CLI
  motionframe-web-worker/ -- WASM worker: protocol, streaming decode
  motionframe-web/      -- WASM frontend: WebPlatform, bridge, worker client
scripts/                -- verify.sh, verify-quick.sh, verify-parity.sh, build-web*.sh
tests/fixtures/         -- Parity test fixtures (OpenCV reference data)
```

## Key Dependencies

- `egui` / `eframe` — GUI framework
- `wgpu` — GPU preview (gated behind `preview` feature)
- `rayon` — parallel iteration in engine
- `thiserror` — error derives in engine
- `clap` — CLI argument parsing (desktop)
- `serde` / `serde_json` — metadata serialization
- `image` — image format decode/encode
- `wasm-bindgen` / `wasm-pack` / `trunk` — web target
- `criterion` — benchmarks
