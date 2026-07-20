use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use crate::cli::config::{LayoutMode, MotionVectorEncodingArg, ResizeAlgorithmArg};

/// Top-level CLI arguments parsed by clap.
#[derive(Parser, Debug)]
#[command(version, about = "MotionFrame — flipbook motion vector tool")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

/// CLI subcommands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Convert one image sequence or atlas into `MotionFrame` output atlases.
    Convert(ConvertArgs),
}

/// Arguments for the `convert` subcommand.
#[derive(Debug, Clone, Parser)]
#[allow(clippy::struct_excessive_bools)] // clap flag pairs require bool fields
pub struct ConvertArgs {
    /// TOML config file.
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// Input directory sequence or single atlas image.
    #[arg(long)]
    pub input: Option<PathBuf>,
    /// Output filename prefix.
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Replace existing output files.
    #[arg(long)]
    pub overwrite: bool,
    /// Suppress non-error output.
    #[arg(long)]
    pub quiet: bool,
    /// Progress output mode.
    #[arg(long, value_enum)]
    pub progress: Option<ProgressArg>,
    /// Desired output frame count.
    #[arg(long)]
    pub output_count: Option<u32>,
    /// Output tile width in pixels.
    #[arg(long)]
    pub tile_width: Option<u32>,
    /// Target total atlas resolution in pixels (auto-computes tile size).
    #[arg(long)]
    pub atlas_resolution: Option<u32>,
    /// Output atlas layout mode.
    #[arg(long, value_enum)]
    pub layout: Option<LayoutMode>,
    /// Manual output atlas columns.
    #[arg(long)]
    pub atlas_cols: Option<u32>,
    /// Manual output atlas rows.
    #[arg(long)]
    pub atlas_rows: Option<u32>,
    /// Maximum output atlas dimension.
    #[arg(long)]
    pub max_atlas_dim: Option<u32>,
    /// Loop output motion.
    #[arg(long = "loop", conflicts_with = "no_loop")]
    pub loop_flag: bool,
    /// Disable loop output motion.
    #[arg(long = "no-loop")]
    pub no_loop: bool,
    /// Motion vector encoding.
    #[arg(long, value_enum)]
    pub motion_vector_encoding: Option<MotionVectorEncodingArg>,
    /// Premultiply color atlas alpha.
    #[arg(long = "premultiply-color", conflicts_with = "straight_color")]
    pub premultiply_color_flag: bool,
    /// Keep color atlas straight alpha.
    #[arg(long = "straight-color")]
    pub straight_color: bool,
    /// Use staggered motion packing.
    #[arg(long = "stagger-pack", conflicts_with = "flat_pack")]
    pub stagger_pack_flag: bool,
    /// Use flat motion packing.
    #[arg(long = "flat-pack")]
    pub flat_pack: bool,
    /// Compute backend: auto (default), cpu, gpu.
    #[arg(long = "compute-backend")]
    pub compute_backend: Option<String>,
    /// Analyze skipped source frames for accumulated motion.
    #[arg(
        long = "analyze-skipped-frames",
        conflicts_with = "no_analyze_skipped_frames"
    )]
    pub analyze_skipped_frames_flag: bool,
    /// Do not analyze skipped source frames.
    #[arg(long = "no-analyze-skipped-frames")]
    pub no_analyze_skipped_frames: bool,
    /// Halve encoded motion vector magnitude.
    #[arg(long = "halve-motion-vector", conflicts_with = "full_motion_vector")]
    pub halve_motion_vector_flag: bool,
    /// Keep full encoded motion vector magnitude.
    #[arg(long = "full-motion-vector")]
    pub full_motion_vector: bool,
    /// Temporal smoothing strength in [0, 1].
    #[arg(long)]
    pub temporal_smoothing: Option<f32>,
    /// Output atlas extrusion in pixels.
    #[arg(long)]
    pub extrude: Option<u32>,
    /// Resize interpolation algorithm.
    #[arg(long, value_enum)]
    pub resize_algorithm: Option<ResizeAlgorithmArg>,
    /// Input atlas columns for single-file atlas input.
    #[arg(long)]
    pub input_atlas_cols: Option<u32>,
    /// Input atlas rows for single-file atlas input.
    #[arg(long)]
    pub input_atlas_rows: Option<u32>,
    /// Trim source tail to hit exact output count.
    #[arg(long = "trim-tail", conflicts_with = "no_trim_tail")]
    pub trim_tail_flag: bool,
    /// Keep source tail behavior untrimmed.
    #[arg(long = "no-trim-tail")]
    pub no_trim_tail: bool,
    /// Output filename format string (tokens: [basename], [cols], [rows], [suffix], [ext]).
    #[arg(long = "out-format")]
    pub output_name_format: Option<String>,
    /// Override the [basename] token in the output format.
    #[arg(long = "out-base")]
    pub output_name_basename: Option<String>,
    /// Label for [suffix] token in color atlas filename.
    #[arg(long = "suffix-color")]
    pub output_suffix_color: Option<String>,
    /// Label for [suffix] token in motion atlas filename.
    #[arg(long = "suffix-motion")]
    pub output_suffix_motion: Option<String>,
    /// Label for [suffix] token in metadata filename.
    #[arg(long = "suffix-meta")]
    pub output_suffix_meta: Option<String>,
    /// First frame to process (0-based, inclusive).
    #[arg(long = "start")]
    pub start_frame: Option<u32>,
    /// Last frame to process (0-based, exclusive). 0 = all frames.
    #[arg(long = "end")]
    pub end_frame: Option<u32>,
}

/// Progress output mode for the `--progress` flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ProgressArg {
    Auto,
    Plain,
    Json,
    None,
}

impl ConvertArgs {
    pub fn compute_backend(&self) -> Option<motionframe_engine::pipeline::ComputeBackend> {
        self.compute_backend
            .as_ref()
            .and_then(|s| match s.as_str() {
                "cpu" => Some(motionframe_engine::pipeline::ComputeBackend::Cpu),
                "gpu" => Some(motionframe_engine::pipeline::ComputeBackend::Gpu),
                "auto" => Some(motionframe_engine::pipeline::ComputeBackend::Auto),
                _ => None,
            })
    }

    pub const fn is_loop(&self) -> Option<bool> {
        if self.loop_flag {
            Some(true)
        } else if self.no_loop {
            Some(false)
        } else {
            None
        }
    }

    pub const fn premultiply_color(&self) -> Option<bool> {
        if self.premultiply_color_flag {
            Some(true)
        } else if self.straight_color {
            Some(false)
        } else {
            None
        }
    }

    pub const fn stagger_pack(&self) -> Option<bool> {
        if self.stagger_pack_flag {
            Some(true)
        } else if self.flat_pack {
            Some(false)
        } else {
            None
        }
    }

    pub const fn analyze_skipped_frames(&self) -> Option<bool> {
        if self.analyze_skipped_frames_flag {
            Some(true)
        } else if self.no_analyze_skipped_frames {
            Some(false)
        } else {
            None
        }
    }

    pub const fn halve_motion_vector(&self) -> Option<bool> {
        if self.halve_motion_vector_flag {
            Some(true)
        } else if self.full_motion_vector {
            Some(false)
        } else {
            None
        }
    }

    pub const fn trim_tail(&self) -> Option<bool> {
        if self.trim_tail_flag {
            Some(true)
        } else if self.no_trim_tail {
            Some(false)
        } else {
            None
        }
    }
}

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
    fn parse_suffix_flags() {
        let args = ConvertArgs::parse_from([
            "test",
            "--suffix-color",
            "",
            "--suffix-motion",
            "_vec",
            "--suffix-meta",
            "_metadata",
        ]);
        assert_eq!(args.output_suffix_color, Some("".into()));
        assert_eq!(args.output_suffix_motion, Some("_vec".into()));
        assert_eq!(args.output_suffix_meta, Some("_metadata".into()));
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
