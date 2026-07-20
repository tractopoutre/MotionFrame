# Output Naming, Frame Range & Config Extension — Implementation Plan

> **For agentic workers:** Use subagent-driven-development or executing-plans. Steps use checkbox syntax.

**Goal:** Add customizable output naming format, start/end frame range, and extend the existing TOML config with new fields.

**Architecture:** Three features sharing `GenerateOptions` fields and the CLI config→job pipeline. Output naming lives in a new engine module (`output_naming.rs`); frame-range slicing lives in `run.rs`; config/TUI wiring lives in the desktop crate.

**Tech Stack:** Rust, serde for TOML config, clap for CLI, egui for GUI

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `crates/motionframe-engine/src/pipeline/output_naming.rs` | **Create** | `OutputFileType` enum, `interpolate_name_format()`, `NameTokens` struct |
| `crates/motionframe-engine/src/pipeline/mod.rs` | Modify | Add fields to `GenerateOptions` + `Default`, register `output_naming` module |
| `crates/motionframe-engine/src/pipeline/run.rs` | Modify | Frame-range slicing in `load_frames()` |
| `crates/motionframe-desktop/src/cli/args.rs` | Modify | New CLI flags on `ConvertArgs` |
| `crates/motionframe-desktop/src/cli/config.rs` | Modify | New `CliConfig` TOML fields + `merge_args()` |
| `crates/motionframe-desktop/src/cli/job.rs` | Modify | Validation + map to `GenerateOptions` |
| `crates/motionframe-desktop/src/cli/run.rs` | Modify | Use format string for output paths |
| `crates/motionframe-ui/src/i18n.rs` | Modify | New i18n keys |
| `crates/motionframe-ui/src/input_panel.rs` | Modify | "Output" section + start/end `DragValue` |
| `crates/motionframe-ui/src/app.rs` | Modify | Reset `end_frame` on sequence load |

---

## Chunk 1: Engine — output naming module + GenerateOptions fields

### Task 1.1: Create `output_naming.rs` module

**Files:**
- Create: `crates/motionframe-engine/src/pipeline/output_naming.rs`
- Modify: `crates/motionframe-engine/src/pipeline/mod.rs`

- [ ] **Step 1: Write tests for `interpolate_name_format`**

Create the test module in `output_naming.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_all_tokens() {
        let tokens = NameTokens {
            basename: "explosion".into(),
            rows: 4,
            cols: 8,
            type_label: "_MV",
            ext: "tga",
        };
        let result = interpolate_name_format("[basename]_[cols]x[rows][type].[ext]", &tokens);
        assert_eq!(result, "explosion_8x4_MV.tga");
    }

    #[test]
    fn unknown_tokens_left_verbatim() {
        let tokens = NameTokens {
            basename: "x".into(), rows: 1, cols: 1, type_label: "", ext: "tga",
        };
        let result = interpolate_name_format("[basename]_[foo].[ext]", &tokens);
        assert_eq!(result, "x_[foo].tga");
    }

    #[test]
    fn empty_format_falls_back_to_default() {
        let tokens = NameTokens {
            basename: "x".into(), rows: 3, cols: 4, type_label: "_meta", ext: "json",
        };
        let result = interpolate_name_format("", &tokens);
        assert_eq!(result, "x_4x3_meta.json");
    }

    #[test]
    fn empty_type_label_no_extra_separator() {
        let tokens = NameTokens {
            basename: "x".into(), rows: 2, cols: 2, type_label: "", ext: "tga",
        };
        let result = interpolate_name_format("[basename]_[cols]x[rows][type].[ext]", &tokens);
        assert_eq!(result, "x_2x2.tga");
    }

    #[test]
    fn custom_basename_overrides() {
        let tokens = NameTokens {
            basename: "my_custom_name".into(), rows: 1, cols: 1, type_label: "", ext: "tga",
        };
        let result = interpolate_name_format("[basename].[ext]", &tokens);
        assert_eq!(result, "my_custom_name.tga");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p motionframe-engine -- pipeline::output_naming`
Expected: error[E0432] — module not found

- [ ] **Step 3: Write the implementation**

```rust
//! Format-string interpolation for output file naming.
//!
//! Tokens are delimited by `[...]`. Unknown tokens are left verbatim.

/// Which output file type is being named.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFileType {
    Color,
    Motion,
    Meta,
}

/// Resolved values for every token in the format string.
pub struct NameTokens<'a> {
    pub basename: &'a str,
    pub cols: u32,
    pub rows: u32,
    pub type_label: &'a str,
    pub ext: &'a str,
}

static DEFAULT_FORMAT: &str = "[basename]_[cols]x[rows][type].[ext]";

/// Interpolate a format string with the given token values.
///
/// If `format` is empty, [`DEFAULT_FORMAT`] is used.
/// Unknown tokens (e.g. `[foo]`) are left verbatim.
pub fn interpolate_name_format(format: &str, tokens: &NameTokens<'_>) -> String {
    let format = if format.is_empty() { DEFAULT_FORMAT } else { format };
    let mut result = String::with_capacity(format.len());
    let mut rest = format;
    while let Some(open) = rest.find('[') {
        result.push_str(&rest[..open]);
        rest = &rest[open + 1..];
        if let Some(close) = rest.find(']') {
            let token = &rest[..close];
            match token {
                "basename" => result.push_str(tokens.basename),
                "cols" => result.push_str(&tokens.cols.to_string()),
                "rows" => result.push_str(&tokens.rows.to_string()),
                "type" => result.push_str(tokens.type_label),
                "ext" => result.push_str(tokens.ext),
                _ => {
                    result.push('[');
                    result.push_str(token);
                    result.push(']');
                }
            }
            rest = &rest[close + 1..];
        } else {
            result.push('[');
            result.push_str(rest);
            rest = "";
        }
    }
    result.push_str(rest);
    result
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p motionframe-engine -- pipeline::output_naming`
Expected: all 6 tests pass

- [ ] **Step 5: Register module in `mod.rs`**

In `crates/motionframe-engine/src/pipeline/mod.rs`, add after the existing pub mod declarations:
```rust
pub mod output_naming;
```

And add the re-export:
```rust
pub use output_naming::{interpolate_name_format, NameTokens, OutputFileType};
```

- [ ] **Step 6: Run all engine tests to verify no regressions**

Run: `cargo test -p motionframe-engine`
Expected: all existing tests pass

- [ ] **Step 7: Commit**

```bash
git add crates/motionframe-engine/src/pipeline/output_naming.rs
git add crates/motionframe-engine/src/pipeline/mod.rs
git commit -m "feat(engine): add output name format interpolation"
```

### Task 1.2: Add new fields to `GenerateOptions`

**Files:**
- Modify: `crates/motionframe-engine/src/pipeline/mod.rs`

- [ ] **Step 1: Write test for new field defaults**

Add to the existing `mod tests` block in `mod.rs`:

```rust
#[test]
fn output_name_format_defaults() {
    let opts = GenerateOptions::default();
    assert_eq!(opts.output_name_format, "[basename]_[cols]x[rows][type].[ext]");
    assert_eq!(opts.output_name_basename, "");
    assert_eq!(opts.output_type_color, "");
    assert_eq!(opts.output_type_motion, "_MV");
    assert_eq!(opts.output_type_meta, "_meta");
    assert_eq!(opts.start_frame, 0);
    assert_eq!(opts.end_frame, 0);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p motionframe-engine -- pipeline::mod`
Expected: compile error, fields don't exist

- [ ] **Step 3: Add fields to `GenerateOptions`**

In `crates/motionframe-engine/src/pipeline/mod.rs`, add to the `GenerateOptions` struct:

```rust
    /// Format string for output file naming. Tokens: `[basename]`, `[cols]`,
    /// `[rows]`, `[type]`, `[ext]`. Empty string = use default.
    #[serde(default)]
    pub output_name_format: String,
    /// Override for `[basename]` token. Empty = auto-derive from input.
    #[serde(default)]
    pub output_name_basename: String,
    /// Label for `[type]` token in color atlas filenames.
    #[serde(default)]
    pub output_type_color: String,
    /// Label for `[type]` token in motion atlas filenames.
    #[serde(default)]
    pub output_type_motion: String,
    /// Label for `[type]` token in metadata filenames.
    #[serde(default)]
    pub output_type_meta: String,
    /// First frame index to process (0-based, inclusive).
    #[serde(default)]
    pub start_frame: u32,
    /// Last frame index to process (0-based, exclusive). 0 = all frames.
    #[serde(default)]
    pub end_frame: u32,
```

- [ ] **Step 4: Add defaults to `Default` impl**

In the `Default for GenerateOptions` impl:

```rust
            output_name_format: "[basename]_[cols]x[rows][type].[ext]".into(),
            output_name_basename: String::new(),
            output_type_color: String::new(),
            output_type_motion: "_MV".into(),
            output_type_meta: "_meta".into(),
            start_frame: 0,
            end_frame: 0,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p motionframe-engine`
Expected: all tests pass, including the new defaults test

- [ ] **Step 6: Commit**

```bash
git add crates/motionframe-engine/src/pipeline/mod.rs
git commit -m "feat(engine): add output naming and frame range fields to GenerateOptions"
```

### Task 1.3: (No engine changes needed)

Frame-range slicing happens in the desktop crate's `load_frames()` (Chunk 2, Task 2.4). The `GenerateOptions` fields for `start_frame`/`end_frame` were already added in Task 1.2. No additional engine changes required.

---

## Chunk 2: Desktop CLI — flags, config, validation, naming

### Task 2.1: Add CLI flags

**Files:**
- Modify: `crates/motionframe-desktop/src/cli/args.rs`

- [ ] **Step 1: Write test for CLI flag parsing**

In `args.rs`, add to an existing or new test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parse_out_format_flag() {
        let args = ConvertArgs::parse_from(["test", "--out-format", "[basename].[ext]"]);
        assert_eq!(args.output_name_format.unwrap(), "[basename].[ext]");
    }

    #[test]
    fn parse_out_base_flag() {
        let args = ConvertArgs::parse_from(["test", "--out-base", "custom_name"]);
        assert_eq!(args.output_name_basename, Some("custom_name".into()));
    }

    #[test]
    fn parse_type_label_flags() {
        let args = ConvertArgs::parse_from([
            "test",
            "--type-color", "",
            "--type-motion", "_vec",
            "--type-meta", "_metadata",
        ]);
        assert_eq!(args.output_type_color, Some("".into()));
        assert_eq!(args.output_type_motion, Some("_vec".into()));
        assert_eq!(args.output_type_meta, Some("_metadata".into()));
    }

    #[test]
    fn parse_start_end_flags() {
        let args = ConvertArgs::parse_from(["test", "--start", "5", "--end", "20"]);
        assert_eq!(args.start_frame, Some(5));
        assert_eq!(args.end_frame, Some(20));
    }

    #[test]
    fn start_end_default_to_none() {
        let args = ConvertArgs::parse_from(["test"]);
        assert_eq!(args.start_frame, None);
        assert_eq!(args.end_frame, None);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p motionframe-desktop -- cli::args`
Expected: compile error, fields don't exist

- [ ] **Step 3: Add new CLI flags to `ConvertArgs`**

Add after the existing fields in `ConvertArgs`:

```rust
    /// Output filename format string (tokens: [basename], [cols], [rows], [type], [ext]).
    #[arg(long = "out-format")]
    pub output_name_format: Option<String>,
    /// Override the [basename] token in the output format.
    #[arg(long = "out-base")]
    pub output_name_basename: Option<String>,
    /// Label for [type] token in color atlas filename.
    #[arg(long = "type-color")]
    pub output_type_color: Option<String>,
    /// Label for [type] token in motion atlas filename.
    #[arg(long = "type-motion")]
    pub output_type_motion: Option<String>,
    /// Label for [type] token in metadata filename.
    #[arg(long = "type-meta")]
    pub output_type_meta: Option<String>,
    /// First frame to process (0-based, inclusive).
    #[arg(long = "start")]
    pub start_frame: Option<u32>,
    /// Last frame to process (0-based, exclusive). 0 = all frames.
    #[arg(long = "end")]
    pub end_frame: Option<u32>,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p motionframe-desktop -- cli::args`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/motionframe-desktop/src/cli/args.rs
git commit -m "feat(cli): add output naming and frame range CLI flags"
```

### Task 2.2: Add TOML config fields + merge_args

**Files:**
- Modify: `crates/motionframe-desktop/src/cli/config.rs`

- [ ] **Step 1: Write test for config merge**

In `config.rs`, add a test:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::args::ConvertArgs;

    #[test]
    fn merge_args_overrides_output_name_format() {
        let cfg = CliConfig {
            output_name_format: Some("old_format.[ext]".into()),
            ..Default::default()
        };
        let args = ConvertArgs {
            output_name_format: Some("new_format.[ext]".into()),
            ..Default::default()
        };
        let merged = cfg.merge_args(&args).unwrap();
        assert_eq!(merged.output_name_format.unwrap(), "new_format.[ext]");
    }

    #[test]
    fn merge_args_preserves_config_when_cli_not_set() {
        let cfg = CliConfig {
            output_name_format: Some("config_format.[ext]".into()),
            ..Default::default()
        };
        let args = ConvertArgs::default();
        let merged = cfg.merge_args(&args).unwrap();
        assert_eq!(merged.output_name_format.unwrap(), "config_format.[ext]");
    }

    #[test]
    fn merge_args_start_end_flags() {
        let cfg = CliConfig::default();
        let args = ConvertArgs {
            start_frame: Some(5),
            end_frame: Some(20),
            ..Default::default()
        };
        let merged = cfg.merge_args(&args).unwrap();
        assert_eq!(merged.start_frame, Some(5));
        assert_eq!(merged.end_frame, Some(20));
    }

    #[test]
    fn merge_args_skips_none_start_end() {
        let cfg = CliConfig {
            start_frame: Some(3),
            end_frame: Some(15),
            ..Default::default()
        };
        let args = ConvertArgs {
            start_frame: None,
            end_frame: None,
            ..Default::default()
        };
        let merged = cfg.merge_args(&args).unwrap();
        assert_eq!(merged.start_frame, Some(3));  // preserved from config
        assert_eq!(merged.end_frame, Some(15));
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p motionframe-desktop -- cli::config`
Expected: compile error, fields don't exist on CliConfig

- [ ] **Step 3: Add fields to `CliConfig`**

```rust
    pub output_name_format: Option<String>,
    pub output_name_basename: Option<String>,
    pub output_type_color: Option<String>,
    pub output_type_motion: Option<String>,
    pub output_type_meta: Option<String>,
    pub start_frame: Option<u32>,
    pub end_frame: Option<u32>,
```

- [ ] **Step 4: Add merge logic to `merge_args()`**

Add before `Ok(self)`:

```rust
        self.output_name_format = args.output_name_format.clone().or(self.output_name_format);
        self.output_name_basename = args.output_name_basename.clone().or(self.output_name_basename);
        self.output_type_color = args.output_type_color.clone().or(self.output_type_color);
        self.output_type_motion = args.output_type_motion.clone().or(self.output_type_motion);
        self.output_type_meta = args.output_type_meta.clone().or(self.output_type_meta);
        self.start_frame = args.start_frame.or(self.start_frame);
        self.end_frame = args.end_frame.or(self.end_frame);
```

Note: CLI flags are `Option<u32>`, so `None` means the user didn't pass them. The merge uses `.or()` which preserves the config value when the CLI flag is absent.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p motionframe-desktop -- cli::config`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/motionframe-desktop/src/cli/config.rs
git commit -m "feat(cli): add output naming and frame range to TOML config"
```

### Task 2.3: Validation and mapping in ConvertJob

**Files:**
- Modify: `crates/motionframe-desktop/src/cli/job.rs`

- [ ] **Step 1: Write tests for validation**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_greater_than_end_returns_error() {
        let cfg = CliConfig {
            start_frame: Some(10),
            end_frame: Some(5),
            input: Some("tests/fixtures".into()),
            output: Some("out".into()),
            ..Default::default()
        };
        let result = ConvertJob::from_config(cfg);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("start"));
    }

    #[test]
    fn valid_start_end_ok() {
        let cfg = CliConfig {
            start_frame: Some(2),
            end_frame: Some(8),
            input: Some("tests/fixtures".into()),
            output: Some("out".into()),
            ..Default::default()
        };
        if let Ok(job) = ConvertJob::from_config(cfg) {
            assert_eq!(job.options.start_frame, 2);
            assert_eq!(job.options.end_frame, 8);
        }
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p motionframe-desktop -- cli::job`
Expected: tests compile but fail (validation not implemented)

- [ ] **Step 3: Add field mapping in `from_config()`**

After the existing option mappings:

```rust
        options.output_name_format = cfg.output_name_format.unwrap_or(options.output_name_format);
        options.output_name_basename = cfg.output_name_basename.unwrap_or(options.output_name_basename);
        options.output_type_color = cfg.output_type_color.unwrap_or(options.output_type_color);
        options.output_type_motion = cfg.output_type_motion.unwrap_or(options.output_type_motion);
        options.output_type_meta = cfg.output_type_meta.unwrap_or(options.output_type_meta);
        options.start_frame = cfg.start_frame.unwrap_or(options.start_frame);
        options.end_frame = cfg.end_frame.unwrap_or(options.end_frame);
```

- [ ] **Step 4: Add frame range validation after mapping**

After `options.output_atlas_max_dim = max_dim;` block, add:

```rust
        // Frame range validation (CLI only — values come from config or args)
        if cfg.start_frame.is_some() || cfg.end_frame.is_some() {
            let start = options.start_frame;
            let end = options.end_frame;
            if end != 0 && start >= end {
                return Err(CliError::Argument(
                    "--start must be less than --end".into(),
                ));
            }
        }
```

Note: Full frame-count-dependent validation (`start >= frame_count`) happens in `resolve_after_load()` since frame count isn't known until source is loaded. Add there:

In `resolve_after_load()`, after `effective_frame_count` is known:

```rust
        let start = self.options.start_frame;
        let end = self.options.end_frame;
        if end != 0 && end > effective_frame_count {
            return Err(CliError::Argument(format!(
                "--end {end} exceeds effective frame count {effective_frame_count}"
            )));
        }
        if start >= effective_frame_count {
            return Err(CliError::Argument(format!(
                "--start {start} exceeds effective frame count {effective_frame_count}"
            )));
        }
```

- [ ] **Step 5: Defer collision detection to resolve time**

> **Dependency:** This step calls `crate::cli::run::resolve_output_paths()` which is defined in Task 2.4. Execute this step AFTER Task 2.4 is complete, or implement a stub first.

Output path collision requires knowing `basename`, `cols`, `rows` — these aren't available until `resolve_after_load()`. Add collision detection there, after atlas dims are known.

In `resolve_after_load()`, after atlas dims are set:

```rust
        // Check for output path collisions
        let color_path = crate::cli::run::resolve_output_paths(
            &self.output, &self.options, OutputFileType::Color,
        );
        let motion_path = crate::cli::run::resolve_output_paths(
            &self.output, &self.options, OutputFileType::Motion,
        );
        let meta_path = crate::cli::run::resolve_output_paths(
            &self.output, &self.options, OutputFileType::Meta,
        );
        if color_path == motion_path {
            return Err(CliError::Argument(format!(
                "output paths for color and motion atlases collide: {}",
                color_path.display()
            )));
        }
        if color_path == meta_path {
            return Err(CliError::Argument(format!(
                "output paths for color and meta collide: {}",
                color_path.display()
            )));
        }
        if motion_path == meta_path {
            return Err(CliError::Argument(format!(
                "output paths for motion and meta collide: {}",
                motion_path.display()
            )));
        }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -p motionframe-desktop -- cli::job`
Expected: tests pass

- [ ] **Step 7: Commit**

```bash
git add crates/motionframe-desktop/src/cli/job.rs
git commit -m "feat(cli): wire output naming and frame range to ConvertJob"
```

### Task 2.4: Frame-range slicing and output naming in CLI run

**Files:**
- Modify: `crates/motionframe-desktop/src/cli/run.rs`

- [ ] **Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use motionframe_engine::pipeline::output_naming::OutputFileType;

    #[test]
    fn resolve_output_paths_color_tga() {
        let opts = GenerateOptions {
            output_name_format: "[basename]_[cols]x[rows][type].[ext]".into(),
            output_type_color: "".into(),
            output_type_motion: "_MV".into(),
            output_type_meta: "_meta".into(),
            atlas_dims: (4, 3),
            ..Default::default()
        };
        let prefix = Path::new("out/test");
        let path = resolve_output_paths(prefix, &opts, OutputFileType::Color);
        assert_eq!(path, Path::new("out/test_4x3.tga"));
    }

    #[test]
    fn resolve_output_paths_motion_tga() {
        let opts = GenerateOptions {
            output_name_format: "[basename]_[cols]x[rows][type].[ext]".into(),
            output_type_color: "".into(),
            output_type_motion: "_MV".into(),
            output_type_meta: "_meta".into(),
            atlas_dims: (4, 3),
            ..Default::default()
        };
        let prefix = Path::new("out/test");
        let path = resolve_output_paths(prefix, &opts, OutputFileType::Motion);
        assert_eq!(path, Path::new("out/test_4x3_MV.tga"));
    }

    #[test]
    fn resolve_output_paths_custom_basename() {
        let opts = GenerateOptions {
            output_name_format: "[basename]_custom.[ext]".into(),
            output_name_basename: "my_seq".into(),
            atlas_dims: (1, 1),
            ..Default::default()
        };
        let path = resolve_output_paths(Path::new("out/x"), &opts, OutputFileType::Color);
        assert_eq!(path, Path::new("out/my_seq_custom.tga"));
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo build -p motionframe-desktop`
Expected: succeeds

- [ ] **Step 3: Modify `load_frames()` for frame range**

In `load_frames()`, after collecting the full frame list and before returning, apply the range:

For `SourceKind::DirectorySequence`:
```rust
            // After building frames vec:
            let start = job.options.start_frame as usize;
            let end = if job.options.end_frame == 0 {
                frames.len()
            } else {
                job.options.end_frame as usize
            };
            // Clone the sliced subset (can't move out of indexed vec)
            let frames: Vec<ImageRgba8> = frames.drain(start..end).collect();
            let count = u32::try_from(frames.len())
                .map_err(|_| CliError::Argument("sliced frame range has too many frames".into()))?;
            Ok((frames, count))
```

For `SourceKind::SingleAtlas`:
```rust
            // Similarly apply range to sliced tiles
            let start = job.options.start_frame as usize;
            let end = if job.options.end_frame == 0 {
                frames.len()
            } else {
                job.options.end_frame as usize
            };
            let frames: Vec<ImageRgba8> = frames.drain(start..end).collect();
            let count = u32::try_from(frames.len())
                .map_err(|_| CliError::Argument("sliced frame range has too many frames".into()))?;
            Ok((frames, count))
```

- [ ] **Step 4: Replace `output_paths()` with format-string resolution**

Replace the `OutputPaths` struct and `output_paths()` function with:

```rust
use motionframe_engine::pipeline::output_naming::{interpolate_name_format, NameTokens, OutputFileType};

fn resolve_output_paths(
    output_prefix: &Path,
    opts: &GenerateOptions,
    file_type: OutputFileType,
) -> PathBuf {
    let out_dir = output_prefix.parent().unwrap_or_else(|| Path::new("."));
    let stem = output_prefix.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    let basename = if opts.output_name_basename.is_empty() {
        stem
    } else {
        &opts.output_name_basename
    };
    let (cols, rows) = opts.atlas_dims;
    let type_label = match file_type {
        OutputFileType::Color => &opts.output_type_color,
        OutputFileType::Motion => &opts.output_type_motion,
        OutputFileType::Meta => &opts.output_type_meta,
    };
    let ext = match file_type {
        OutputFileType::Color | OutputFileType::Motion => "tga",
        OutputFileType::Meta => "json",
    };

    let tokens = NameTokens { basename, cols, rows, type_label, ext };
    let filename = interpolate_name_format(&opts.output_name_format, &tokens);
    out_dir.join(filename)
}
```

- [ ] **Step 5: Remove `OutputPaths` struct and `output_paths()` function**

Delete the old struct and function entirely. All callers now use `resolve_output_paths()`.

- [ ] **Step 7: Update `ensure_outputs_writable()` and `write_outputs()`**

Change `ensure_outputs_writable()` to use `resolve_output_paths()`:

```rust
fn ensure_outputs_writable(job: &ConvertJob) -> Result<(), CliError> {
    let color = resolve_output_paths(&job.output, &job.options, OutputFileType::Color);
    let motion = resolve_output_paths(&job.output, &job.options, OutputFileType::Motion);
    let meta = resolve_output_paths(&job.output, &job.options, OutputFileType::Meta);
    // ... rest unchanged but using these paths
}
```

Similarly for `write_outputs()`. Remove the old `OutputPaths` struct.

- [ ] **Step 8: Build and test**

Run: `cargo build -p motionframe-desktop`
Expected: compiles without errors

Run: `cargo test -p motionframe-desktop`
Expected: all tests pass

- [ ] **Step 9: Commit**

```bash
git add crates/motionframe-desktop/src/cli/run.rs
git commit -m "feat(cli): apply frame-range slicing and format-string output naming"
```

---

## Chunk 3: GUI — i18n keys, Output section, start/end widgets

### Task 3.1: Add i18n keys

**Files:**
- Modify: `crates/motionframe-ui/src/i18n.rs`

- [ ] **Step 1: Add new `Key` variants**

Add to the `Key` enum:
```rust
    // Sidebar — Output group
    OutputHeading,
    OutputFormat,
    OutputFormatHover,
    OutputBaseName,
    OutputBaseNameHover,
    OutputTypeColor,
    OutputTypeColorHover,
    OutputTypeMotion,
    OutputTypeMotionHover,
    OutputTypeMeta,
    OutputTypeMetaHover,
    OutputPreview,
    OutputPreviewEmpty,
    // Sidebar — Frame range
    FrameRangeHeading,
    StartFrame,
    StartFrameHover,
    EndFrame,
    EndFrameHover,
```

- [ ] **Step 2: Add English translations**

In the `t()` function, add English (`Lang::En`) arms:

```rust
    Key::OutputHeading => "Output",
    Key::OutputFormat => "Format",
    Key::OutputFormatHover => "Use [basename], [cols], [rows], [type], [ext] tokens",
    Key::OutputBaseName => "Base name",
    Key::OutputBaseNameHover => "Override the [basename] token (empty = auto-detect)",
    Key::OutputTypeColor => "Color label",
    Key::OutputTypeColorHover => "What [type] resolves to in the color atlas filename",
    Key::OutputTypeMotion => "Motion label",
    Key::OutputTypeMotionHover => "What [type] resolves to in the motion atlas filename",
    Key::OutputTypeMeta => "Meta label",
    Key::OutputTypeMetaHover => "What [type] resolves to in the metadata filename",
    Key::OutputPreview => "Preview",
    Key::OutputPreviewEmpty => "Using default format",
    Key::FrameRangeHeading => "Frame Range",
    Key::StartFrame => "Start frame",
    Key::StartFrameHover => "First frame to process (0-based)",
    Key::EndFrame => "End frame",
    Key::EndFrameHover => "Last frame to process (0-based, exclusive; 0 = all)",
```

- [ ] **Step 3: Add Japanese translations**

Add Japanese (`Lang::Ja`) arms for the same keys (use the English strings as placeholders if Japanese translation isn't available, or use reasonable translations):

```rust
    Key::OutputHeading => "出力",
    Key::OutputFormat => "フォーマット",
    Key::OutputFormatHover => "[basename], [cols], [rows], [type], [ext] トークンを使用",
    Key::OutputBaseName => "ベース名",
    Key::OutputBaseNameHover => "[basename] トークンの上書き（空=自動検出）",
    Key::OutputTypeColor => "カラーラベル",
    Key::OutputTypeColorHover => "カラーアトラスファイル名の [type] 値",
    Key::OutputTypeMotion => "モーションラベル",
    Key::OutputTypeMotionHover => "モーションアトラスファイル名の [type] 値",
    Key::OutputTypeMeta => "メタラベル",
    Key::OutputTypeMetaHover => "メタデータファイル名の [type] 値",
    Key::OutputPreview => "プレビュー",
    Key::OutputPreviewEmpty => "デフォルトフォーマットを使用中",
    Key::FrameRangeHeading => "フレーム範囲",
    Key::StartFrame => "開始フレーム",
    Key::StartFrameHover => "処理する最初のフレーム（0始まり）",
    Key::EndFrame => "終了フレーム",
    Key::EndFrameHover => "処理する最後のフレーム（0始まり、排他。0=すべて）",
```

- [ ] **Step 4: Build and verify**

Run: `cargo build -p motionframe-ui`
Expected: compiles successfully

- [ ] **Step 5: Run i18n completeness test**

If there's a test that checks all `Key` variants produce non-empty strings, run it:
Run: `cargo test -p motionframe-ui`
Expected: passes

- [ ] **Step 6: Commit**

```bash
git add crates/motionframe-ui/src/i18n.rs
git commit -m "feat(ui): add i18n keys for output naming and frame range"
```

### Task 3.2: Add Output section and frame range widgets

**Files:**
- Modify: `crates/motionframe-ui/src/input_panel.rs`

- [ ] **Step 1: Add `show_output_group()` function**

Add a new function that renders the "Output" section:

```rust
fn show_output_group(ui: &mut egui::Ui, options: &mut GenerateOptions, sequence_loaded: bool, lang: Lang) {
    ui.heading(t(lang, Key::OutputHeading));

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::OutputFormat));
        ui.add(
            egui::TextEdit::singleline(&mut options.output_name_format)
                .desired_width(f32::INFINITY)
                .hint_text("[basename]_[cols]x[rows][type].[ext]"),
        )
        .on_hover_text(t(lang, Key::OutputFormatHover));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::OutputBaseName));
        ui.add(
            egui::TextEdit::singleline(&mut options.output_name_basename)
                .desired_width(120.0)
                .hint_text(t(lang, Key::OutputBaseNameHover)),
        )
        .on_hover_text(t(lang, Key::OutputBaseNameHover));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::OutputTypeColor));
        ui.add(egui::TextEdit::singleline(&mut options.output_type_color).desired_width(80.0))
            .on_hover_text(t(lang, Key::OutputTypeColorHover));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::OutputTypeMotion));
        ui.add(egui::TextEdit::singleline(&mut options.output_type_motion).desired_width(80.0))
            .on_hover_text(t(lang, Key::OutputTypeMotionHover));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::OutputTypeMeta));
        ui.add(egui::TextEdit::singleline(&mut options.output_type_meta).desired_width(80.0))
            .on_hover_text(t(lang, Key::OutputTypeMetaHover));
    });

    // Live preview (only when a sequence is loaded)
    if sequence_loaded {
        let (cols, rows) = options.atlas_dims;
        let basename = if options.output_name_basename.is_empty() {
            "input"
        } else {
            &options.output_name_basename
        };
        ui.add_space(4.0);
        ui.label(t(lang, Key::OutputPreview));

        let format = &options.output_name_format;
        if format.is_empty() {
            ui.colored_label(egui::Color32::RED, t(lang, Key::OutputPreviewEmpty));
        } else {
            let tokens = |type_label: &str, ext: &str| motionframe_engine::pipeline::output_naming::NameTokens {
                basename,
                cols,
                rows,
                type_label,
                ext,
            };
            let color_name = motionframe_engine::pipeline::output_naming::interpolate_name_format(
                format, &tokens(&options.output_type_color, "tga"),
            );
            let motion_name = motionframe_engine::pipeline::output_naming::interpolate_name_format(
                format, &tokens(&options.output_type_motion, "tga"),
            );
            let meta_name = motionframe_engine::pipeline::output_naming::interpolate_name_format(
                format, &tokens(&options.output_type_meta, "json"),
            );
            ui.label(egui::RichText::new(color_name).size(12.0).weak());
            ui.label(egui::RichText::new(motion_name).size(12.0).weak());
            ui.label(egui::RichText::new(meta_name).size(12.0).weak());
        }
    }
}
```

- [ ] **Step 2: Add `show_frame_range_group()` function**

```rust
fn show_frame_range_group(ui: &mut egui::Ui, options: &mut GenerateOptions, n_input: u32, lang: Lang) {
    if options.input_atlas_dims.is_some() {
        return; // only in sequence mode
    }
    ui.heading(t(lang, Key::FrameRangeHeading));

    let mut start = options.start_frame;
    let mut end = options.end_frame;

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::StartFrame));
        ui.add(egui::DragValue::new(&mut start).range(0..=end.saturating_sub(1).max(0)))
            .on_hover_text(t(lang, Key::StartFrameHover));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::EndFrame));
        ui.add(egui::DragValue::new(&mut end).range(1..=n_input))
            .on_hover_text(t(lang, Key::EndFrameHover));
    });

    if start > end.saturating_sub(1) {
        ui.colored_label(egui::Color32::RED, "Start must be less than end");
    }

    options.start_frame = start;
    options.end_frame = end;
}
```

- [ ] **Step 3: Wire into `show_options()`**

Add calls in `show_options()` after the atlas section. The function signature gains a `sequence_loaded: bool` parameter:

```rust
pub fn show_options(
    ui: &mut egui::Ui,
    options: &mut GenerateOptions,
    atlas_layout_manual: &mut bool,
    output_dims: Option<OutputDims>,
    n_input: u32,
    canonical_layouts: &[DetentEntry],
    sequence_loaded: bool,
    lang: Lang,
) {
```

Inside, add after the atlas section:

```rust
    ui.separator();
    show_frame_range_group(ui, options, n_input, lang);
    ui.separator();
    show_output_group(ui, options, sequence_loaded, lang);
```

Update all callers of `show_options()` in `app.rs` to pass `sequence_loaded` (true when `state != AppState::Empty`).

- [ ] **Step 4: Write GUI unit test for preview generation**

In the existing test module of `input_panel.rs`:

```rust
    #[test]
    fn preview_uses_interpolate_name_format() {
        let mut opts = GenerateOptions::default();
        opts.output_name_format = "[basename]_custom.[ext]".into();
        let (cols, rows) = opts.atlas_dims;
        let tokens = motionframe_engine::pipeline::output_naming::NameTokens {
            basename: "test",
            cols, rows,
            type_label: "",
            ext: "tga",
        };
        let name = motionframe_engine::pipeline::output_naming::interpolate_name_format(
            &opts.output_name_format, &tokens,
        );
        assert_eq!(name, "test_custom.tga");
    }

    #[test]
    fn empty_format_falls_back_to_default_in_preview() {
        let mut opts = GenerateOptions::default();
        opts.output_name_format = String::new();
        let (cols, rows) = opts.atlas_dims;
        let tokens = motionframe_engine::pipeline::output_naming::NameTokens {
            basename: "test",
            cols, rows,
            type_label: "_MV",
            ext: "tga",
        };
        let name = motionframe_engine::pipeline::output_naming::interpolate_name_format(
            &opts.output_name_format, &tokens,
        );
        assert_eq!(name, "test_8x8_MV.tga");
    }
```

- [ ] **Step 5: Run tests to verify**

Run: `cargo test -p motionframe-ui`
Expected: all tests pass

- [ ] **Step 6: Build and verify**

Run: `cargo build -p motionframe-ui`
Expected: compiles successfully

- [ ] **Step 7: Commit**

```bash
git add crates/motionframe-ui/src/input_panel.rs
git commit -m "feat(ui): add output naming and frame range widgets"
```

### Task 3.3: Reset end_frame on sequence load

**Files:**
- Modify: `crates/motionframe-ui/src/app.rs`

- [ ] **Step 1: Modify `accept_picked_frames()`**

In `accept_picked_frames()`, after `self.frame_dims = Some((w, h));`, add:

```rust
        self.options.start_frame = 0;
        self.options.end_frame = frames.len() as u32;
```

- [ ] **Step 2: Build and verify**

Run: `cargo build -p motionframe-ui`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add crates/motionframe-ui/src/app.rs
git commit -m "feat(ui): reset start/end frame on sequence load"
```

---

## Verification

After all chunks are implemented:

```bash
./scripts/verify-quick.sh  # fmt + clippy + test
```

Expected: all passes. If clippy warnings arise, fix them.

Manual testing:
```bash
# CLI frame range
cargo run --release --bin motionframe -- convert --input tests/fixtures/explosion00 --output out/test --start 5 --end 10

# CLI custom format
cargo run --release --bin motionframe -- convert --input tests/fixtures/explosion00 --output out/test --out-format "[basename]_custom.[ext]"

# CLI config override
cargo run --release --bin motionframe -- convert --config my_config.toml --output out/test
```
