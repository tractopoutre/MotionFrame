use std::path::{Path, PathBuf};

use motionframe_engine::io::metadata::{self, AtlasMetadata};
use motionframe_engine::io::sequence;
use motionframe_engine::io::tga::{load_rgba, save_tga};
use motionframe_engine::io::{slice_atlas, InMemoryFrames};
use motionframe_engine::pipeline::run::{self, EncodeResult};
use motionframe_engine::pipeline::{ImageRgba8, Progress};

use crate::cli::config::ProgressMode;
use crate::cli::job::{ConvertJob, SourceKind};
use crate::cli::CliError;

/// Execute a validated conversion job: load frames, resolve layout, run pipeline, write outputs.
pub fn run_convert(mut job: ConvertJob) -> Result<(), CliError> {
    let (frames, effective_frame_count) = load_frames(&job)?;
    let first = frames
        .first()
        .ok_or_else(|| CliError::Argument("input produced no frames".into()))?;

    job.resolve_after_load(first.width, first.height, effective_frame_count)?;
    ensure_outputs_writable(&job)?;

    emit_progress(job.progress, "running pipeline");
    let source = InMemoryFrames::new(frames)
        .map_err(|e| CliError::Pipeline(format!("frame source: {e}")))?;
    let progress_fn = |_p: Progress| {};
    let cancel_fn = || false;
    let result: EncodeResult = run::run_pipeline(&source, &job.options, &progress_fn, &cancel_fn)?;

    write_outputs(&job.output, &result)?;
    emit_progress(job.progress, "done");
    Ok(())
}

fn load_frames(job: &ConvertJob) -> Result<(Vec<ImageRgba8>, u32), CliError> {
    match job.source_kind {
        SourceKind::DirectorySequence => {
            let seed = sequence::resolve_seed_file(&job.input).ok_or_else(|| {
                CliError::Pipeline(format!(
                    "could not find image files in '{}'",
                    job.input.display()
                ))
            })?;
            let filename = seed
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| CliError::Pipeline("invalid seed filename".into()))?;
            let (prefix, num_digits, ext) =
                sequence::detect_pattern(filename).ok_or_else(|| {
                    CliError::Pipeline(format!("could not detect frame pattern in '{filename}'"))
                })?;
            let dir = seed.parent().unwrap_or_else(|| Path::new("."));
            let files = sequence::collect_sequence_files(dir, &prefix, num_digits, &ext);
            if files.is_empty() {
                return Err(CliError::Pipeline(format!(
                    "no matching frames found in '{}'",
                    dir.display()
                )));
            }
            let mut frames = Vec::with_capacity(files.len());
            for file in &files {
                frames.push(load_rgba(file)?);
            }
            let count = u32::try_from(frames.len())
                .map_err(|_| CliError::Argument("input sequence has too many frames".into()))?;
            Ok((frames, count))
        }
        SourceKind::SingleAtlas => {
            let image = load_rgba(&job.input)?;
            let (cols, rows) = job.options.input_atlas_dims.ok_or_else(|| {
                CliError::Argument("single-file input requires input atlas dimensions".into())
            })?;
            let frames = slice_atlas(&image, cols, rows)
                .map_err(|e| CliError::Pipeline(format!("atlas slice: {e}")))?;
            let count = u32::try_from(frames.len())
                .map_err(|_| CliError::Argument("input atlas has too many tiles".into()))?;
            Ok((frames, count))
        }
    }
}

fn ensure_outputs_writable(job: &ConvertJob) -> Result<(), CliError> {
    let paths = output_paths(&job.output)?;
    if let Some(parent) = job.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    if !job.overwrite {
        for path in [&paths.color, &paths.motion, &paths.meta] {
            if path.exists() {
                return Err(CliError::Argument(format!(
                    "output already exists: {}; use --overwrite to replace",
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

fn write_outputs(output_prefix: &Path, result: &EncodeResult) -> Result<(), CliError> {
    let paths = output_paths(output_prefix)?;
    save_tga(&paths.color, &result.color_atlas)?;
    save_tga(&paths.motion, &result.motion_atlas)?;
    let meta = AtlasMetadata {
        strength: result.strength,
        total_frames: result.total_frames,
        atlas_width: result.atlas_width,
        atlas_height: result.atlas_height,
        columns: result.columns,
        rows: result.rows,
        pack_mode: result.pack_mode,
        is_loop: result.is_loop,
        premultiplied_alpha: result.premultiplied_alpha,
    };
    metadata::write(&paths.meta, &meta)?;
    Ok(())
}

struct OutputPaths {
    color: PathBuf,
    motion: PathBuf,
    meta: PathBuf,
}

fn output_paths(prefix: &Path) -> Result<OutputPaths, CliError> {
    let out_dir = prefix.parent().unwrap_or_else(|| Path::new("."));
    let stem = prefix.file_name().ok_or_else(|| {
        CliError::Argument(format!(
            "output prefix has no file name: {}",
            prefix.display()
        ))
    })?;
    let stem = stem.to_string_lossy();
    Ok(OutputPaths {
        color: out_dir.join(format!("{stem}_color_atlas.tga")),
        motion: out_dir.join(format!("{stem}_motion_atlas.tga")),
        meta: out_dir.join(format!("{stem}_meta.json")),
    })
}

fn emit_progress(mode: ProgressMode, message: &str) {
    match mode {
        ProgressMode::None => {}
        ProgressMode::Json => {
            eprintln!(r#"{{"event":"progress","message":"{message}"}}"#);
        }
        ProgressMode::Auto | ProgressMode::Plain => {
            eprintln!("{message}");
        }
    }
}
