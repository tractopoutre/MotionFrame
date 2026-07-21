//! Debug dump binary: runs CPU + GPU pipeline on frames from the fixtures
//! directory and saves all intermediate stages to `debug_dump/` as TGA images.
//!
//! Usage:  cargo run --bin debugdump
//!
//! Loads frames 20, 40, 60, 80 from tests/fixtures/explosion00.

fn fixture_dir() -> std::path::PathBuf {
    // Resolve relative to the project root (works when running via `cargo run`
    // from the MotionFrame workspace root).
    let exe = std::env::current_exe().ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    // Walk up from target/debug/ to find tests/fixtures/explosion00
    let mut probe = exe.clone();
    for _ in 0..6 {
        let candidate = probe.join("tests").join("fixtures").join("explosion00");
        if candidate.is_dir() {
            return candidate;
        }
        probe.pop();
    }
    // Last resort: relative to CWD (works when `cargo run` is run from workspace root)
    std::path::PathBuf::from("tests/fixtures/explosion00")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("MFRAME_DUMP", "1");

    let dir = fixture_dir();
    let frame_indices = [20usize, 40, 60, 80];

    let mut frames = Vec::new();
    for &idx in &frame_indices {
        let path = dir.join(format!("explosion00-frame{:03}.tga", idx));
        eprintln!("Loading: {}", path.display());
        let bytes = std::fs::read(&path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        let img = motionframe_engine::io::decode::decode_image_from_bytes(
            &path.to_string_lossy(),
            &bytes,
        )
        .map_err(|e| format!("failed to decode {}: {e}", path.display()))?;
        frames.push(img);
    }

    eprintln!("Loaded {} frames from {}", frames.len(), dir.display());
    for (i, &idx) in frame_indices.iter().enumerate() {
        eprintln!("  frame[{}] = idx {}", i, idx);
    }

    // Build options: 4 frames → pairs (0,1), (1,2), (2,3)
    let mut opts = motionframe_engine::pipeline::GenerateOptions::default();
    opts.analyze_skipped_frames = true;
    opts.output_frames = 4;
    opts.farneback.iterations = 3;
    opts.farneback.winsize = 7;
    opts.farneback.poly_n = 5;
    opts.resize_algorithm = motionframe_engine::pipeline::Interpolation::Cubic;

    let progress = |_| {};
    let cancel = || false;

    // === CPU path ===
    eprintln!("\n=== CPU path ===");
    opts.compute_backend = motionframe_engine::pipeline::ComputeBackend::Cpu;
    let source = motionframe_engine::io::source::InMemoryFrames::new(frames.clone())?;
    match motionframe_engine::pipeline::run::run_pipeline_with_gpu(
        &source, &opts, &progress, &cancel, None,
    ) {
        Ok(_) => eprintln!("CPU path: done"),
        Err(e) => eprintln!("CPU path error: {e}"),
    }

    // === GPU path ===
    eprintln!("\n=== GPU path ===");
    let gpu = motionframe_engine::gpu::GpuPipeline::try_init();
    opts.compute_backend = motionframe_engine::pipeline::ComputeBackend::Gpu;
    let source2 = motionframe_engine::io::source::InMemoryFrames::new(frames.clone())?;
    match motionframe_engine::pipeline::run::run_pipeline_with_gpu(
        &source2, &opts, &progress, &cancel, gpu.as_ref(),
    ) {
        Ok(_) => eprintln!("GPU path: done"),
        Err(e) => eprintln!("GPU path error: {e}"),
    }

    eprintln!("\nDebug dumps saved to debug_dump/");
    eprintln!("Compare: cpu_pair0_combined.tga  vs  gpu_pair0_combined.tga");
    eprintln!("         cpu_pair0_fwd.tga       vs  gpu_pair0_fwd.tga");
    eprintln!("         cpu_pair0_acc.tga       (CPU accumulation after pair 0)");
    Ok(())
}
