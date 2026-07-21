//! Debug dump binary: runs CPU + GPU pipeline on input frames and saves all
//! intermediate stages to `debug_dump/` as TGA images.
//!
//! Usage:
//!   cargo run --bin debugdump -- <input_dir> [output_count]
//!
//! Example:
//!   cargo run --bin debugdump -- ./frames 8

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Enable debug dumps programmatically
    std::env::set_var("MFRAME_DUMP", "1");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input_dir> [output_count]", args[0]);
        std::process::exit(1);
    }
    let dir = PathBuf::from(&args[1]);
    let output_count: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);

    // Load frames from directory
    let pattern = motionframe_engine::io::sequence::detect_pattern(
        &dir.join("frame_0001.png").to_string_lossy(),
    );
    let (base, start, ext) = pattern.unwrap_or_else(|| {
        // Try common patterns
        let entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| matches!(p.extension().and_then(|s| s.to_str()), Some("png" | "jpg" | "tga")))
            .collect();
        let first = entries.first().unwrap();
        let name = first.file_stem().unwrap().to_string_lossy();
        let ext = first.extension().unwrap().to_string_lossy().to_string();
        // Extract numeric suffix
        let digits: String = name.chars().rev().take_while(|c| c.is_ascii_digit()).collect();
        let start: u32 = digits.chars().rev().collect::<String>().parse().unwrap_or(0);
        let base = name.trim_end_matches(&digits);
        (base.to_string(), start, ext)
    });

    let frames: Vec<motionframe_engine::pipeline::ImageRgba8> = (0..output_count + 1)
        .map(|i| {
            let idx = start + i as u32;
            let path = dir.join(format!("{}{:04}.{}", base, idx, ext));
            motionframe_engine::io::decode::decode_image_from_bytes(
                &path.to_string_lossy(),
                &std::fs::read(&path).unwrap(),
            )
            .unwrap()
        })
        .collect();

    eprintln!("Loaded {} frames from {}", frames.len(), dir.display());

    // Run CPU path first
    eprintln!("=== CPU path ===");
    let opts = {
        let mut o = motionframe_engine::pipeline::GenerateOptions::default();
        o.analyze_skipped_frames = true;
        o.output_frames = output_count as u32;
        o.compute_backend = motionframe_engine::pipeline::ComputeBackend::Cpu;
        o.farneback.iterations = 3;
        o.farneback.winsize = 7;
        o.farneback.poly_n = 5;
        o.resize_algorithm = motionframe_engine::pipeline::Interpolation::Cubic;
        o
    };
    let progress = |_| {};
    let cancel = || false;
    let source = motionframe_engine::io::source::InMemoryFrames::new(frames.clone()).unwrap();
    match motionframe_engine::pipeline::run::run_pipeline_with_gpu(
        &source,
        &opts,
        &progress,
        &cancel,
        None,
    ) {
        Ok(_) => eprintln!("CPU path: done"),
        Err(e) => eprintln!("CPU path error (may be OK without fixtures): {e}"),
    }

    // Run GPU path
    eprintln!("=== GPU path ===");
    let gpu = motionframe_engine::gpu::GpuPipeline::try_init();
    let opts = {
        let mut o = motionframe_engine::pipeline::GenerateOptions::default();
        o.analyze_skipped_frames = true;
        o.output_frames = output_count as u32;
        o.compute_backend = motionframe_engine::pipeline::ComputeBackend::Gpu;
        o.farneback.iterations = 3;
        o.farneback.winsize = 7;
        o.farneback.poly_n = 5;
        o.resize_algorithm = motionframe_engine::pipeline::Interpolation::Cubic;
        o
    };
    let source2 = motionframe_engine::io::source::InMemoryFrames::new(frames.clone()).unwrap();
    match motionframe_engine::pipeline::run::run_pipeline_with_gpu(
        &source2,
        &opts,
        &progress,
        &cancel,
        gpu.as_ref(),
    ) {
        Ok(_) => eprintln!("GPU path: done"),
        Err(e) => eprintln!("GPU path error: {e}"),
    }

    eprintln!("\nDebug dumps saved to debug_dump/");
    eprintln!("Compare CPU vs GPU equivalents, e.g.:");
    eprintln!("  cpu_pair0_combined.tga  vs  gpu_pair0_combined.tga");
    eprintln!("  cpu_pair0_fwd.tga       vs  gpu_pair0_fwd.tga");
    eprintln!("  cpu_pair0_acc.tga       (CPU only — GPU equivalent is the final flow)");
    Ok(())
}
