use std::path::{Path, PathBuf};

use motionframe_engine::io::metadata::{self, AtlasMetadata};
use motionframe_engine::io::sequence;
use motionframe_engine::io::tga::{load_rgba, save_tga};
use motionframe_engine::io::{slice_atlas, InMemoryFrames};
use motionframe_engine::pipeline::output_naming::{interpolate_name_format, NameTokens, OutputFileType};
use motionframe_engine::pipeline::run::{self, EncodeResult};
use motionframe_engine::pipeline::{GenerateOptions, ImageRgba8, Progress};

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
    let result: EncodeResult;

    // Attempt GPU-accelerated pipeline; fall back to CPU on failure.
    let gpu = motionframe_engine::gpu::GpuPipeline::try_init();
    if let Some(gpu) = gpu.as_ref() {
        eprintln!("[gpu] using GPU pipeline (RTX 4090 or equivalent)");
        match run::run_pipeline_with_gpu(&source, &job.options, &progress_fn, &cancel_fn, Some(gpu)) {
            Ok(r) => {
                result = r;
                eprintln!("[gpu] GPU pipeline succeeded");
            }
            Err(e) => {
                eprintln!("[gpu] GPU pipeline failed: {e}; falling back to CPU");
                result = run::run_pipeline(&source, &job.options, &progress_fn, &cancel_fn)?;
            }
        }
    } else {
        eprintln!("[gpu] no GPU device available, using CPU pipeline");
        result = run::run_pipeline(&source, &job.options, &progress_fn, &cancel_fn)?;
    }

    write_outputs(&job.output, &result, &job.options)?;
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
        }
        SourceKind::SingleAtlas => {
            let image = load_rgba(&job.input)?;
            let (cols, rows) = job.options.input_atlas_dims.ok_or_else(|| {
                CliError::Argument("single-file input requires input atlas dimensions".into())
            })?;
            let mut frames = slice_atlas(&image, cols, rows)
                .map_err(|e| CliError::Pipeline(format!("atlas slice: {e}")))?;
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
        }
    }
}

fn ensure_outputs_writable(job: &ConvertJob) -> Result<(), CliError> {
    let color = resolve_output_paths(&job.output, &job.options, OutputFileType::Color);
    let motion = resolve_output_paths(&job.output, &job.options, OutputFileType::Motion);
    let meta = resolve_output_paths(&job.output, &job.options, OutputFileType::Meta);
    if let Some(parent) = job.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    if !job.overwrite {
        for path in [&color, &motion, &meta] {
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

fn write_outputs(
    output_prefix: &Path,
    result: &EncodeResult,
    opts: &GenerateOptions,
) -> Result<(), CliError> {
    let color = resolve_output_paths(output_prefix, opts, OutputFileType::Color);
    let motion = resolve_output_paths(output_prefix, opts, OutputFileType::Motion);
    let meta = resolve_output_paths(output_prefix, opts, OutputFileType::Meta);
    save_tga(&color, &result.color_atlas)?;
    save_tga(&motion, &result.motion_atlas)?;
    let meta_obj = AtlasMetadata {
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
    metadata::write(&meta, &meta_obj)?;
    Ok(())
}

pub fn resolve_output_paths(
    output_prefix: &Path,
    opts: &GenerateOptions,
    file_type: OutputFileType,
) -> PathBuf {
    let out_dir = output_prefix.parent().unwrap_or_else(|| Path::new("."));
    let stem = output_prefix
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    let basename = if opts.output_name_basename.is_empty() {
        stem
    } else {
        &opts.output_name_basename
    };
    let (cols, rows) = opts.atlas_dims;
    let suffix = match file_type {
        OutputFileType::Color => &opts.output_suffix_color,
        OutputFileType::Motion => &opts.output_suffix_motion,
        OutputFileType::Meta => &opts.output_suffix_meta,
    };
    let ext = match file_type {
        OutputFileType::Color | OutputFileType::Motion => "tga",
        OutputFileType::Meta => "json",
    };

    let tokens = NameTokens {
        basename,
        cols,
        rows,
        suffix,
        ext,
    };
    let filename = interpolate_name_format(&opts.output_name_format, &tokens);
    out_dir.join(filename)
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

#[cfg(test)]
mod tests {
    use super::*;
    use motionframe_engine::pipeline::output_naming::OutputFileType;

    #[test]
    fn resolve_output_paths_color_tga() {
        let opts = GenerateOptions {
            output_name_format: "[basename]_[cols]x[rows][suffix].[ext]".into(),
            output_suffix_color: "".into(),
            output_suffix_motion: "_MV".into(),
            output_suffix_meta: "_meta".into(),
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
            output_name_format: "[basename]_[cols]x[rows][suffix].[ext]".into(),
            output_suffix_color: "".into(),
            output_suffix_motion: "_MV".into(),
            output_suffix_meta: "_meta".into(),
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
