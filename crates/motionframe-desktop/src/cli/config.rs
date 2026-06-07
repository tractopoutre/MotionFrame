use std::path::{Path, PathBuf};

use clap::ValueEnum;
use serde::Deserialize;

use crate::cli::args::{ConvertArgs, ProgressArg};
use crate::cli::CliError;

/// Atlas layout mode: automatic or manual grid dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum LayoutMode {
    Auto,
    Manual,
}

/// Motion vector encoding format for CLI config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum MotionVectorEncodingArg {
    Staggered,
    Normal,
}

/// Resize interpolation algorithm for CLI config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum ResizeAlgorithmArg {
    Nearest,
    Linear,
    Cubic,
    Lanczos,
}

/// Progress output mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProgressMode {
    Auto,
    Plain,
    Json,
    None,
}

impl From<ProgressArg> for ProgressMode {
    fn from(value: ProgressArg) -> Self {
        match value {
            ProgressArg::Auto => Self::Auto,
            ProgressArg::Plain => Self::Plain,
            ProgressArg::Json => Self::Json,
            ProgressArg::None => Self::None,
        }
    }
}

/// TOML config schema for `motionframe convert`.
///
/// All fields are optional; missing values fall back to engine defaults.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CliConfig {
    pub input: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub overwrite: Option<bool>,
    pub output_count: Option<u32>,
    pub tile_width: Option<u32>,
    pub max_atlas_dim: Option<u32>,
    pub layout: Option<LayoutMode>,
    pub atlas_cols: Option<u32>,
    pub atlas_rows: Option<u32>,
    #[serde(rename = "loop")]
    pub is_loop: Option<bool>,
    pub motion_vector_encoding: Option<MotionVectorEncodingArg>,
    pub premultiply_color: Option<bool>,
    pub stagger_pack: Option<bool>,
    pub analyze_skipped_frames: Option<bool>,
    pub halve_motion_vector: Option<bool>,
    pub temporal_smoothing: Option<f32>,
    pub extrude: Option<u32>,
    pub resize_algorithm: Option<ResizeAlgorithmArg>,
    pub input_atlas_cols: Option<u32>,
    pub input_atlas_rows: Option<u32>,
    pub trim_tail: Option<bool>,
    pub progress: Option<ProgressMode>,
}

impl CliConfig {
    pub fn load(path: &Path) -> Result<Self, CliError> {
        let text = std::fs::read_to_string(path).map_err(CliError::Io)?;
        toml::from_str(&text)
            .map_err(|e| CliError::Config(format!("invalid config {}: {e}", path.display())))
    }

    pub fn merge_args(mut self, args: &ConvertArgs) -> Result<Self, CliError> {
        if args.quiet && matches!(args.progress, Some(p) if p != ProgressArg::None) {
            return Err(CliError::Argument(
                "--quiet conflicts with --progress values other than none".into(),
            ));
        }

        self.input = args.input.clone().or(self.input);
        self.output = args.output.clone().or(self.output);
        if args.overwrite {
            self.overwrite = Some(true);
        }
        self.output_count = args.output_count.or(self.output_count);
        self.tile_width = args.tile_width.or(self.tile_width);
        self.max_atlas_dim = args.max_atlas_dim.or(self.max_atlas_dim);
        self.layout = args.layout.or(self.layout);
        self.atlas_cols = args.atlas_cols.or(self.atlas_cols);
        self.atlas_rows = args.atlas_rows.or(self.atlas_rows);
        self.is_loop = args.is_loop().or(self.is_loop);
        self.motion_vector_encoding = args.motion_vector_encoding.or(self.motion_vector_encoding);
        self.premultiply_color = args.premultiply_color().or(self.premultiply_color);
        self.stagger_pack = args.stagger_pack().or(self.stagger_pack);
        self.analyze_skipped_frames = args
            .analyze_skipped_frames()
            .or(self.analyze_skipped_frames);
        self.halve_motion_vector = args.halve_motion_vector().or(self.halve_motion_vector);
        self.temporal_smoothing = args.temporal_smoothing.or(self.temporal_smoothing);
        self.extrude = args.extrude.or(self.extrude);
        self.resize_algorithm = args.resize_algorithm.or(self.resize_algorithm);
        self.input_atlas_cols = args.input_atlas_cols.or(self.input_atlas_cols);
        self.input_atlas_rows = args.input_atlas_rows.or(self.input_atlas_rows);
        self.trim_tail = args.trim_tail().or(self.trim_tail);
        self.progress = if args.quiet {
            Some(ProgressMode::None)
        } else {
            args.progress.map(ProgressMode::from).or(self.progress)
        };
        Ok(self)
    }
}
