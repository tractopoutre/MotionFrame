use std::path::PathBuf;

use motionframe_engine::pipeline::atlas_layout::{compute_tile_dims, pick_layout, AtlasLayout, DEFAULT_PADDING_BOUND};
use motionframe_engine::pipeline::output_detents::{build_output_count_detents, DetentEntry};
use motionframe_engine::pipeline::output_naming::OutputFileType;
use motionframe_engine::pipeline::{GenerateOptions, Interpolation, MotionVectorEncoding};

use crate::cli::config::{
    CliConfig, LayoutMode, MotionVectorEncodingArg, ProgressMode, ResizeAlgorithmArg,
};
use crate::cli::CliError;

/// Whether the input is a directory sequence or a single atlas file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    DirectorySequence,
    SingleAtlas,
}

/// Resolved atlas layout strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputLayout {
    Auto,
    Manual { cols: u32, rows: u32 },
}

/// Validated single conversion job, ready for execution.
///
/// Created from a merged `CliConfig` via `from_config`. Post-load resolution
/// (`resolve_after_load`) finalizes atlas dimensions once source geometry is known.
#[derive(Debug, Clone)]
pub struct ConvertJob {
    pub input: PathBuf,
    pub output: PathBuf,
    pub overwrite: bool,
    pub source_kind: SourceKind,
    pub layout: OutputLayout,
    pub progress: ProgressMode,
    pub options: GenerateOptions,
    pub requested_output_count: Option<u32>,
}

impl ConvertJob {
    pub fn from_config(cfg: CliConfig) -> Result<Self, CliError> {
        let input = cfg
            .input
            .ok_or_else(|| CliError::Argument("--input <path> is required".into()))?;
        let output = cfg
            .output
            .ok_or_else(|| CliError::Argument("--output <prefix> is required".into()))?;

        let source_kind = classify_source(&input)?;
        validate_input_atlas_dims(source_kind, cfg.input_atlas_cols, cfg.input_atlas_rows)?;

        let layout = match cfg.layout.unwrap_or(LayoutMode::Auto) {
            LayoutMode::Auto => {
                if cfg.atlas_cols.is_some() || cfg.atlas_rows.is_some() {
                    return Err(CliError::Argument(
                        "--atlas-cols and --atlas-rows require --layout manual".into(),
                    ));
                }
                OutputLayout::Auto
            }
            LayoutMode::Manual => {
                let cols = cfg.atlas_cols.ok_or_else(|| {
                    CliError::Argument(
                        "manual layout requires both --atlas-cols and --atlas-rows".into(),
                    )
                })?;
                let rows = cfg.atlas_rows.ok_or_else(|| {
                    CliError::Argument(
                        "manual layout requires both --atlas-cols and --atlas-rows".into(),
                    )
                })?;
                if cols == 0 || rows == 0 {
                    return Err(CliError::Argument(
                        "manual layout cols and rows must be greater than zero".into(),
                    ));
                }
                OutputLayout::Manual { cols, rows }
            }
        };

        let max_dim = cfg.max_atlas_dim.unwrap_or(8192);
        if !matches!(max_dim, 1024 | 2048 | 4096 | 8192) {
            return Err(CliError::Argument(
                "max-atlas-dim must be one of 1024, 2048, 4096, 8192".into(),
            ));
        }

        let mut options = GenerateOptions::default();
        options.atlas_resolution = cfg.atlas_resolution.unwrap_or(options.atlas_resolution);
        options.tile_pixel_width = cfg.tile_width.unwrap_or(options.tile_pixel_width);
        options.output_atlas_max_dim = max_dim;
        options.is_loop = cfg.is_loop.unwrap_or(options.is_loop);
        options.trim_tail_for_exact_output_count = cfg
            .trim_tail
            .unwrap_or(options.trim_tail_for_exact_output_count);
        options.premultiplied_alpha = cfg.premultiply_color.unwrap_or(options.premultiplied_alpha);
        options.stagger_pack = cfg.stagger_pack.unwrap_or(options.stagger_pack);
        options.analyze_skipped_frames = cfg
            .analyze_skipped_frames
            .unwrap_or(options.analyze_skipped_frames);
        options.halve_motion_vector = cfg
            .halve_motion_vector
            .unwrap_or(options.halve_motion_vector);
        options.temporal_smoothing = cfg
            .temporal_smoothing
            .unwrap_or(options.temporal_smoothing)
            .clamp(0.0, 1.0);
        options.extrude = cfg.extrude.unwrap_or(options.extrude);
        options.resize_algorithm = match cfg.resize_algorithm {
            Some(ResizeAlgorithmArg::Nearest) => Interpolation::Nearest,
            Some(ResizeAlgorithmArg::Linear) => Interpolation::Linear,
            Some(ResizeAlgorithmArg::Cubic) | None => Interpolation::Cubic,
            Some(ResizeAlgorithmArg::Lanczos) => Interpolation::Lanczos,
        };
        options.input_atlas_dims = cfg.input_atlas_cols.zip(cfg.input_atlas_rows);
        options.motion_vector_encoding = match cfg.motion_vector_encoding {
            Some(MotionVectorEncodingArg::Staggered) | None => MotionVectorEncoding::R8G8Remap01,
            Some(MotionVectorEncodingArg::Normal) => MotionVectorEncoding::SidefxLabsR8G8,
        };
        options.output_name_format = cfg.output_name_format.unwrap_or(options.output_name_format);
        options.output_name_basename = cfg.output_name_basename.unwrap_or(options.output_name_basename);
        options.output_suffix_color = cfg.output_suffix_color.unwrap_or(options.output_suffix_color);
        options.output_suffix_motion = cfg.output_suffix_motion.unwrap_or(options.output_suffix_motion);
        options.output_suffix_meta = cfg.output_suffix_meta.unwrap_or(options.output_suffix_meta);
        options.start_frame = cfg.start_frame.unwrap_or(options.start_frame);
        options.end_frame = cfg.end_frame.unwrap_or(options.end_frame);

        if options.end_frame != 0 && options.start_frame >= options.end_frame {
            return Err(CliError::Argument(
                "--start must be less than --end".into(),
            ));
        }

        Ok(Self {
            input,
            output,
            overwrite: cfg.overwrite.unwrap_or(false),
            source_kind,
            layout,
            progress: cfg.progress.unwrap_or(ProgressMode::Auto),
            options,
            requested_output_count: cfg.output_count,
        })
    }

    /// Resolve output count and atlas layout after source dimensions are known.
    ///
    /// `effective_frame_count` is the count after the start/end slice has been
    /// applied by `load_frames()`.
    pub fn resolve_after_load(
        &mut self,
        source_width: u32,
        source_height: u32,
        effective_frame_count: u32,
    ) -> Result<(), CliError> {
        if effective_frame_count < 2 {
            return Err(CliError::Argument(
                "effective input frame count must be at least 2".into(),
            ));
        }

        let output_count = self.resolve_output_count(effective_frame_count)?;
        let input_aspect_ratio = source_width as f64 / source_height as f64;
        let atlas_res = self.options.atlas_resolution;

        match self.layout {
            OutputLayout::Auto => {
                // In auto mode, use the new aspect-ratio-aware pick_layout.
                let layout: AtlasLayout = pick_layout(
                    output_count,
                    input_aspect_ratio,
                    atlas_res,
                    self.options.output_atlas_max_dim,
                    DEFAULT_PADDING_BOUND,
                )
                .ok_or_else(|| {
                    CliError::Argument(format!(
                        "auto layout cannot fit {output_count} output frames under atlas_resolution {atlas_res}",
                    ))
                })?;
                self.options.atlas_dims = (layout.cols, layout.rows);
                self.options.tile_pixel_width = layout.tile_width;
            }
            OutputLayout::Manual { cols, rows } => {
                let slots = cols.saturating_mul(rows);
                if slots < output_count {
                    return Err(CliError::Argument(format!(
                        "manual layout has {slots} slots, but output-count requires {output_count} frames"
                    )));
                }
                // For manual layout, compute tile dims from atlas_resolution.
                let (tile_w, tile_h) = compute_tile_dims(atlas_res, cols, rows, input_aspect_ratio);
                if cols.saturating_mul(tile_w) > self.options.output_atlas_max_dim
                    || rows.saturating_mul(tile_h) > self.options.output_atlas_max_dim
                {
                    return Err(CliError::Argument(format!(
                        "manual layout {}x{} exceeds max-atlas-dim {}",
                        cols, rows, self.options.output_atlas_max_dim
                    )));
                }
                self.options.atlas_dims = (cols, rows);
                self.options.tile_pixel_width = tile_w;
            }
        }

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

        Ok(())
    }

    fn resolve_output_count(&mut self, effective_frame_count: u32) -> Result<u32, CliError> {
        let Some(requested) = self.requested_output_count else {
            self.options.frame_skip = 0;
            self.options.output_frames = effective_frame_count;
            return Ok(effective_frame_count);
        };

        let detents = build_output_count_detents(
            effective_frame_count,
            self.options.trim_tail_for_exact_output_count,
        );
        let Some(entry) = detents.iter().find(|entry| entry.output_count == requested) else {
            let nearby = nearby_counts(&detents, requested);
            return Err(CliError::Argument(format!(
                "output-count {requested} is not valid for {effective_frame_count} effective input frames; nearby valid counts: {nearby}"
            )));
        };
        self.options.frame_skip = entry.frame_skip;
        self.options.output_frames = entry.output_count;
        Ok(entry.output_count)
    }
}

fn classify_source(input: &PathBuf) -> Result<SourceKind, CliError> {
    let meta = std::fs::metadata(input).map_err(CliError::Io)?;
    if meta.is_dir() {
        Ok(SourceKind::DirectorySequence)
    } else if meta.is_file() {
        Ok(SourceKind::SingleAtlas)
    } else {
        Err(CliError::Argument(format!(
            "input path is neither a directory nor a file: {}",
            input.display()
        )))
    }
}

fn validate_input_atlas_dims(
    source_kind: SourceKind,
    cols: Option<u32>,
    rows: Option<u32>,
) -> Result<(), CliError> {
    match source_kind {
        SourceKind::DirectorySequence => {
            if cols.is_some() || rows.is_some() {
                return Err(CliError::Argument(
                    "directory input rejects input atlas dimensions".into(),
                ));
            }
        }
        SourceKind::SingleAtlas => {
            let Some(cols) = cols else {
                return Err(CliError::Argument(
                    "single-file input requires --input-atlas-cols and --input-atlas-rows".into(),
                ));
            };
            let Some(rows) = rows else {
                return Err(CliError::Argument(
                    "single-file input requires --input-atlas-cols and --input-atlas-rows".into(),
                ));
            };
            if cols == 0 || rows == 0 {
                return Err(CliError::Argument(
                    "input atlas cols and rows must be greater than zero".into(),
                ));
            }
        }
    }
    Ok(())
}

fn nearby_counts(detents: &[DetentEntry], requested: u32) -> String {
    let mut counts: Vec<u32> = detents.iter().map(|entry| entry.output_count).collect();
    counts.sort_by_key(|count| count.abs_diff(requested));
    counts
        .into_iter()
        .take(3)
        .map(|count| count.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_greater_than_end_returns_error() {
        let cfg = CliConfig {
            start_frame: Some(10),
            end_frame: Some(5),
            input: Some(".".into()),
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
            input: Some(".".into()),
            output: Some("out".into()),
            ..Default::default()
        };
        if let Ok(job) = ConvertJob::from_config(cfg) {
            assert_eq!(job.options.start_frame, 2);
            assert_eq!(job.options.end_frame, 8);
        }
    }
}
