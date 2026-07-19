use std::path::PathBuf;

use motionframe_desktop::cli::args::{ConvertArgs, ProgressArg};
use motionframe_desktop::cli::config::{
    CliConfig, LayoutMode, MotionVectorEncodingArg, ProgressMode, ResizeAlgorithmArg,
};
use motionframe_desktop::cli::job::{ConvertJob, SourceKind};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("explosion00")
}

fn fixture_file() -> PathBuf {
    fixture_dir().join("explosion00-frame001.tga")
}

const fn empty_args() -> ConvertArgs {
    ConvertArgs {
        config: None,
        input: None,
        output: None,
        overwrite: false,
        quiet: false,
        progress: None,
        output_count: None,
        tile_width: None,
        layout: None,
        atlas_cols: None,
        atlas_rows: None,
        max_atlas_dim: None,
        loop_flag: false,
        no_loop: false,
        motion_vector_encoding: None,
        premultiply_color_flag: false,
        straight_color: false,
        stagger_pack_flag: false,
        flat_pack: false,
        analyze_skipped_frames_flag: false,
        no_analyze_skipped_frames: false,
        halve_motion_vector_flag: false,
        full_motion_vector: false,
        temporal_smoothing: None,
        extrude: None,
        resize_algorithm: None,
        input_atlas_cols: None,
        input_atlas_rows: None,
        trim_tail_flag: false,
        no_trim_tail: false,
        output_name_format: None,
        output_name_basename: None,
        output_type_color: None,
        output_type_motion: None,
        output_type_meta: None,
        start_frame: None,
        end_frame: None,
    }
}

#[test]
fn toml_config_parses() {
    let cfg: CliConfig = toml::from_str(
        r#"
input = "frames/explosion"
output = "out/explosion"
overwrite = true
output_count = 64
tile_width = 128
max_atlas_dim = 8192
layout = "auto"
loop = false
motion_vector_encoding = "staggered"
premultiply_color = false
stagger_pack = false
analyze_skipped_frames = true
halve_motion_vector = false
temporal_smoothing = 0.0
extrude = 0
resize_algorithm = "cubic"
progress = "plain"
"#,
    )
    .expect("parse config");

    assert_eq!(cfg.input, Some(PathBuf::from("frames/explosion")));
    assert_eq!(cfg.output_count, Some(64));
    assert_eq!(cfg.layout, Some(LayoutMode::Auto));
    assert_eq!(cfg.progress, Some(ProgressMode::Plain));
    assert_eq!(
        cfg.motion_vector_encoding,
        Some(MotionVectorEncodingArg::Staggered)
    );
    assert_eq!(cfg.resize_algorithm, Some(ResizeAlgorithmArg::Cubic));
}

#[test]
fn cli_flags_override_config() {
    let cfg = CliConfig {
        input: Some(PathBuf::from("from-config")),
        output: Some(PathBuf::from("out-config")),
        overwrite: Some(false),
        output_count: Some(32),
        tile_width: Some(128),
        max_atlas_dim: Some(4096),
        layout: Some(LayoutMode::Auto),
        atlas_cols: None,
        atlas_rows: None,
        is_loop: Some(false),
        motion_vector_encoding: Some(MotionVectorEncodingArg::Staggered),
        premultiply_color: Some(false),
        stagger_pack: Some(false),
        analyze_skipped_frames: Some(true),
        halve_motion_vector: Some(false),
        temporal_smoothing: Some(0.0),
        extrude: Some(0),
        resize_algorithm: Some(ResizeAlgorithmArg::Cubic),
        input_atlas_cols: None,
        input_atlas_rows: None,
        trim_tail: Some(false),
        progress: Some(ProgressMode::Plain),
        output_name_format: None,
        output_name_basename: None,
        output_type_color: None,
        output_type_motion: None,
        output_type_meta: None,
        start_frame: None,
        end_frame: None,
    };
    let mut args = empty_args();
    args.input = Some(PathBuf::from("from-cli"));
    args.output_count = Some(64);
    args.overwrite = true;
    args.progress = Some(ProgressArg::Json);

    let merged = cfg.merge_args(&args).expect("merge config and args");

    assert_eq!(merged.input, Some(PathBuf::from("from-cli")));
    assert_eq!(merged.output_count, Some(64));
    assert_eq!(merged.overwrite, Some(true));
    assert_eq!(merged.progress, Some(ProgressMode::Json));
    assert_eq!(merged.output, Some(PathBuf::from("out-config")));
}

#[test]
fn quiet_conflicts_with_explicit_progress() {
    let cfg = CliConfig::default();
    let mut args = empty_args();
    args.quiet = true;
    args.progress = Some(ProgressArg::Plain);

    let err = cfg
        .merge_args(&args)
        .expect_err("quiet plus progress is invalid");
    assert!(err
        .to_string()
        .contains("--quiet conflicts with --progress"));
}

// --- Job validation tests ---

#[test]
fn manual_layout_requires_both_axes() {
    let cfg = CliConfig {
        input: Some(fixture_dir()),
        output: Some(PathBuf::from("target/tmp/out")),
        layout: Some(LayoutMode::Manual),
        atlas_cols: Some(8),
        atlas_rows: None,
        ..CliConfig::default()
    };

    let err = ConvertJob::from_config(cfg).expect_err("manual layout needs both axes");
    assert!(err.to_string().contains("manual layout requires both"));
}

#[test]
fn invalid_max_atlas_dim_is_rejected() {
    let cfg = CliConfig {
        input: Some(fixture_dir()),
        output: Some(PathBuf::from("target/tmp/out")),
        max_atlas_dim: Some(1234),
        ..CliConfig::default()
    };

    let err = ConvertJob::from_config(cfg).expect_err("bad max dim");
    assert!(err.to_string().contains("max-atlas-dim must be one of"));
}

#[test]
fn directory_input_rejects_input_atlas_dims() {
    let cfg = CliConfig {
        input: Some(fixture_dir()),
        output: Some(PathBuf::from("target/tmp/out")),
        input_atlas_cols: Some(8),
        input_atlas_rows: Some(8),
        ..CliConfig::default()
    };

    let err = ConvertJob::from_config(cfg).expect_err("directory atlas dims invalid");
    assert!(err
        .to_string()
        .contains("directory input rejects input atlas dimensions"));
}

#[test]
fn single_file_input_requires_input_atlas_dims() {
    let cfg = CliConfig {
        input: Some(fixture_file()),
        output: Some(PathBuf::from("target/tmp/out")),
        ..CliConfig::default()
    };

    let err = ConvertJob::from_config(cfg).expect_err("file atlas dims required");
    assert!(err.to_string().contains("single-file input requires"));
}

#[test]
fn minimal_directory_job_uses_auto_layout_defaults() {
    let cfg = CliConfig {
        input: Some(fixture_dir()),
        output: Some(PathBuf::from("target/tmp/out")),
        ..CliConfig::default()
    };

    let job = ConvertJob::from_config(cfg).expect("valid job");
    assert_eq!(job.source_kind, SourceKind::DirectorySequence);
    assert_eq!(job.options.tile_pixel_width, 128);
    assert_eq!(job.options.frame_skip, 0);
    assert_eq!(job.options.output_atlas_max_dim, 8192);
}

// --- Post-load resolution tests ---

#[test]
fn manual_layout_rejects_too_few_slots_after_load() {
    let cfg = CliConfig {
        input: Some(fixture_dir()),
        output: Some(PathBuf::from("target/tmp/out")),
        output_count: Some(40),
        layout: Some(LayoutMode::Manual),
        atlas_cols: Some(5),
        atlas_rows: Some(6),
        ..CliConfig::default()
    };
    let mut job = ConvertJob::from_config(cfg).expect("pre-load valid");

    let err = job
        .resolve_after_load(128, 128, 80)
        .expect_err("manual layout too small");
    assert!(err.to_string().contains("manual layout has 30 slots"));
}

#[test]
fn noncanonical_output_count_is_rejected() {
    let cfg = CliConfig {
        input: Some(fixture_dir()),
        output: Some(PathBuf::from("target/tmp/out")),
        output_count: Some(63),
        ..CliConfig::default()
    };
    let mut job = ConvertJob::from_config(cfg).expect("pre-load valid");

    let err = job
        .resolve_after_load(128, 128, 80)
        .expect_err("noncanonical count");
    assert!(err.to_string().contains("output-count 63 is not valid"));
}

#[test]
fn auto_layout_resolves_to_fitting_atlas() {
    let cfg = CliConfig {
        input: Some(fixture_dir()),
        output: Some(PathBuf::from("target/tmp/out")),
        output_count: Some(40),
        tile_width: Some(128),
        max_atlas_dim: Some(8192),
        ..CliConfig::default()
    };
    let mut job = ConvertJob::from_config(cfg).expect("pre-load valid");

    job.resolve_after_load(128, 128, 80)
        .expect("resolve layout");

    assert_eq!(job.options.frame_skip, 1);
    assert!(job.options.atlas_dims.0 * job.options.atlas_dims.1 >= 40);
}
