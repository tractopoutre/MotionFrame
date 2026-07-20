# Output Naming, Frame Range & Config Extension Design

## Summary

Three independent features that extend MotionFrame's CLI and GUI:

1. **Customizable output naming** — a format string with tokens replaces the hardcoded `{prefix}_color_atlas.tga` scheme
2. **Start/end frame range** — `--start`/`--end` flags slice the input sequence
3. **Config reuse** — the existing `--config` TOML system is unchanged; new fields are added to `CliConfig` and `GenerateOptions`

## 1. Output Naming Format

### Format string

A single format string interpolated per-output-file. Tokens use `[...]` delimiters:

| Token | Meaning |
|---|---|
| `[basename]` | Input filename stem, overridable via `--out-base` |
| `[rows]` | Atlas rows count |
| `[cols]` | Atlas columns count |
| `[type]` | Per-output-type label (see below) |
| `[ext]` | File extension: `tga` or `json` |

### Per-type labels

Each output file type has a configurable label that `[type]` resolves to. The default labels include the separator prefix so the format string doesn't need a hardcoded `_` before `[type]`:

| Output | Default `[type]` value | Example with default format |
|---|---|---|
| Color atlas | `""` (empty) | `explosion_4x4.tga` |
| Motion atlas | `"_MV"` | `explosion_4x4_MV.tga` |
| Metadata JSON | `"_meta"` | `explosion_4x4_meta.json` |

### New fields

**`GenerateOptions` (engine crate):**
- `output_name_format: String` — default `"[basename]_[cols]x[rows][type].[ext]"` (no `_` before `[type]`)
- `output_name_basename: String` — default `""` (auto-derived); when non-empty overrides `[basename]`
- `output_type_color: String` — default `""`
- `output_type_motion: String` — default `"_MV"`
- `output_type_meta: String` — default `"_meta"`

**Backward compatibility:** The default produces `explosion_color_atlas.tga` → `explosion_4x4.tga` (color) and `explosion_MV.tga` → `explosion_4x4_MV.tga` (motion). This is a **deliberate naming change** — users relying on the old hardcoded suffixes will need to set `--out-format "[basename]_color_atlas.[ext]"` or use a config template to restore the old scheme.

**CLI flags:**
- `--out-format <STRING>` — overrides `output_name_format`
- `--out-base <STRING>` — overrides `output_name_basename`
- `--type-color <STRING>` — overrides `output_type_color`
- `--type-motion <STRING>` — overrides `output_type_motion`
- `--type-meta <STRING>` — overrides `output_type_meta`

**`CliConfig` (TOML):**
All the above as optional `Option<String>` fields.

### Resolution logic

Replace `output_paths()` in `run.rs` with:

```rust
fn resolve_output_paths(
    prefix: &Path,
    opts: &GenerateOptions,
    output_type: OutputFileType,  // enum { Color, Motion, Meta } defined locally in run.rs
) -> PathBuf
```

Steps:
1. Read format string from `opts.output_name_format`
2. If format string is empty, fall back to the default
3. Determine `[basename]`: `opts.output_name_basename` if non-empty, else `prefix.file_stem()`
4. Resolve `[cols]`, `[rows]` from `opts.atlas_dims`
5. Resolve `[type]` from the per-type label matching `output_type`
6. Resolve `[ext]` from `output_type` (`.tga` for Color/Motion, `.json` for Meta)
7. Replace every `[token]` in the format string with its resolved value
8. **Unknown tokens** (e.g. `[foo]`) are left verbatim in the output — no error
9. Join with `prefix.parent()` to produce the final `PathBuf`

**Collision prevention:** In `ConvertJob::from_config()`, after resolving all three output paths, validate they are distinct. If two resolve to the same `PathBuf`, return `CliError::Argument("output paths for {a} and {b} collide: {path}")`.

The existing `OutputPaths { color, motion, meta }` struct (private to `run.rs`, used only within that file) is replaced by calling `resolve_output_paths()` three times. No public API or cross-crate consumers are affected.

### GUI

A new "Output" section in `input_panel.rs` with:
- `TextEdit` for the format string
- `TextEdit` for `[basename]` override
- `TextEdit` for each output type label
- Live preview of the three filenames below the fields

**Preview behavior:** When no sequence is loaded (`AppState::Empty`), the preview area is hidden. When a sequence is loaded, the preview shows the three filenames using the current format string + type labels + atlas dims (same fallback logic as the engine: empty format string falls back to default). If the format string is empty, a red warning label is shown below the preview.

## 2. Start / End Frame Range

### Semantics

- 0-based indices into the discovered input sequence
- `start` is inclusive, `end` is exclusive (frames `start..end`)
- Default: `start = 0`, `end = 0` (meaning "use all frames")
- When `end` is 0 at resolution time, it is replaced with the frame count

### New fields

**`GenerateOptions`:**
- `start_frame: u32` — default `0`
- `end_frame: u32` — default `0` (all)

**CLI flags:**
- `--start <N>` — overrides `start_frame`
- `--end <N>` — overrides `end_frame`

**`CliConfig` (TOML):**
- `start_frame: Option<u32>`
- `end_frame: Option<u32>`

### CLI validation

In `ConvertJob::from_config()`:
- If `start >= end` after resolution (and `end != 0`): return `CliError::Argument("--start must be less than --end")`
- If `start >= frame_count`: return `CliError::Argument("--start exceeds frame count")`
- If `end > frame_count`: return `CliError::Argument("--end {end} exceeds frame count {frame_count}")`

### Engine change

In `load_frames()` in `run.rs`, after collecting the file list:

```rust
let frame_count = files.len();
let start = opts.start_frame as usize;
let end = match opts.end_frame {
    0 => frame_count,
    n => n as usize,  // validated ≤ frame_count in from_config()
};
let files = &files[start..end];
```

For **directory sequence mode**: the slice is applied to the sorted file list.

For **single-atlas mode**: the source image is decoded and sliced into tiles first (producing a `Vec<ImageRgba8>` of `cols × rows` tiles). The range slice is then applied to this tile vector. This means `start`/`end` refer to tile indices, not pixel regions.

### GUI

Two `DragValue` widgets in the Atlas group:
- Only shown in sequence mode (not atlas mode, where tile count = `cols × rows`)
- When a new sequence is loaded, `start_frame` resets to 0 and `end_frame` resets to the frame count
- Constraints: `start < end`, `end <= frame_count`

**Sentinel behavior:** The engine interprets `end_frame == 0` as "use all frames." In the GUI, `end_frame` is always set to the concrete frame count on sequence load (never 0). When serialized via eframe persistence, it stores the concrete value. In TOML config, a user can write `end_frame = 0` to mean "all frames" (the default). This is consistent: the sentinel is resolved at the engine layer, while the GUI and persistence always work with concrete values.

## 3. Config / Template (Existing Mechanism)

### No changes to mechanism

The existing `--config <file.toml>` + CLI override pattern (`merge_args()`) already supports the workflow the user described. No new flags, no YAML, no `.mf` extension.

### What gets added

All new fields (output naming + frame range) are added as `Option<T>` fields to `CliConfig` and flow through `merge_args()` (CLI overrides config) and `CliConfig → ConvertJob → GenerateOptions` pipeline.

## File Changes

| File | Change |
|---|---|
| `crates/motionframe-engine/src/pipeline/mod.rs` | Add fields to `GenerateOptions` + `Default` impl |
| `crates/motionframe-engine/src/pipeline/run.rs` | Replace `output_paths()` with format-string interpolation; add frame-range slicing in `load_frames()` |
| `crates/motionframe-desktop/src/cli/args.rs` | Add new CLI flags to `ConvertArgs` |
| `crates/motionframe-desktop/src/cli/config.rs` | Add new fields to `CliConfig` + extend `merge_args()` |
| `crates/motionframe-desktop/src/cli/job.rs` | Map new config fields to `GenerateOptions` in `from_config()` |
| `crates/motionframe-desktop/src/cli/run.rs` | Pass new fields to resolution logic; update `output_paths()` call |
| `crates/motionframe-ui/src/input_panel.rs` | Add Output naming section + start/end `DragValue` |
| `crates/motionframe-ui/src/i18n.rs` | Add i18n keys for new UI strings |
| `crates/motionframe-ui/src/app.rs` | Reset `end_frame` on sequence load |

## Test Strategy

### Unit tests (engine crate)

**Output naming:**
- Format string with all tokens resolves correctly
- Unknown tokens left verbatim
- Empty `[type]` produces no extra separator (`name__MV.tga` edge case — if `[type]` is empty and format is `[name]_[type]_MV`, collapse consecutive underscores? No — leave as-is, user controls the format)
- `[basename]` override vs auto-derive
- `[ext]` correct per output type

**Frame range:**
- `start=0, end=0` resolves to full frame count
- `start=5, end=10` on 20 frames produces indices 5..9
- `end > frame_count` clamps to frame_count
- `start=0, end=0` on atlas-mode tiles
- Edge: `start=0, end=0` with zero frames (single-atlas edge case)

### Integration tests (desktop crate or test binary)

- CLI `--start 5 --end 10` filters frames in `run_convert`
- CLI `--out-format` produces expected filenames
- CLI `--out-base` overrides `[basename]`
- CLI `--start 10 --end 5` returns error exit code 2
- Config TOML with new fields is loaded and overridden by CLI flags

### GUI tests (input_panel unit tests)

- Output preview format string display
- Frame range `DragValue` constraints

## Out of Scope

- YAML support (the user chose to keep TOML)
- Preview/playback changes (no visual impact)
- Web worker protocol changes (engine-only; worker receives `GenerateOptions` unchanged)
