//! Pipeline analysis: Lagrangian particle accumulator, flow normalization,
//! L∞ clip, and sliding-overlap batch planning.

use crate::flow::farneback::farneback;
use crate::pipeline::{Flow, GenerateOptions, ImageF32, MotionVectorEncoding};
use rayon::prelude::*;

/// Plan of which frames feed which Farneback batch.
///
/// Batches are `frame_skip + 2` frames long with sliding overlaps:
/// consecutive batches share their boundary frame so motion accumulation
/// is continuous across the full sequence.
///
/// Caller drives the work via `process_batch_slice` +
/// `build_tail_flow_from_plan`, which lets progress + cooperative yields
/// land between chunks instead of hiding inside one giant `par_iter`.
pub struct BatchPlan {
    /// Index lists for the main (full-length) batches.
    pub batches: Vec<Vec<usize>>,
    /// Trailing partial batch (shorter than `batch_len`), if any.
    pub tail_batch: Option<Vec<usize>>,
    /// Last full-length batch — used to seed the wrap batch in loop mode
    /// when the trailing partial batch is empty.
    pub last_valid_batch: Option<Vec<usize>>,
}

/// Walk `frames_len` and bucket indices into batches per
/// `opts.frame_skip + 2`-frame sliding overlaps.
pub fn plan_batches(frames_len: usize, opts: &GenerateOptions) -> BatchPlan {
    let batch_len = opts.frame_skip as usize + 2;

    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut tail_batch: Option<Vec<usize>> = None;
    let mut last_valid_batch: Option<Vec<usize>> = None;

    let mut frame_idx: usize = 1;

    while frame_idx < frames_len {
        frame_idx = frame_idx.saturating_sub(1);

        let mut batch_indices: Vec<usize> = Vec::new();
        for _ in 0..batch_len {
            if frame_idx >= frames_len {
                break;
            }
            batch_indices.push(frame_idx);
            frame_idx += 1;
        }

        if batch_indices.len() != batch_len {
            tail_batch = Some(batch_indices);
            break;
        }

        last_valid_batch = Some(batch_indices.clone());
        batches.push(batch_indices);
    }

    BatchPlan {
        batches,
        tail_batch,
        last_valid_batch,
    }
}

/// Run a slice of batches in parallel and return their flows.
///
/// Caller chooses the slice size: pass all batches for maximum rayon
/// parallelism, or smaller chunks to yield between Farneback calls.
pub fn process_batch_slice(
    batches: &[Vec<usize>],
    frames: &[ImageF32],
    opts: &GenerateOptions,
) -> Vec<Flow> {
    process_batch_slice_with_progress(batches, frames, opts, &|| true, &|| {})
}

/// Run a slice of batches in parallel and call `batch_done` after each flow.
///
/// Used by the pipeline to publish low-overhead, throttled progress during the
/// long Farneback stage without splitting Rayon work into many smaller jobs.
pub fn process_batch_slice_with_progress(
    batches: &[Vec<usize>],
    frames: &[ImageF32],
    opts: &GenerateOptions,
    should_continue: &(dyn Fn() -> bool + Sync),
    batch_done: &(dyn Fn() + Sync),
) -> Vec<Flow> {
    batches
        .par_iter()
        .map(|batch_indices| {
            if !should_continue() {
                return None;
            }
            let batch_refs: Vec<&ImageF32> = batch_indices.iter().map(|&i| &frames[i]).collect();
            let mut flow = accumulate_displacement(&batch_refs, opts);
            normalize_flow(&mut flow);
            batch_done();
            Some(flow)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect()
}

/// Build the final motion vector frame using indices from `plan`.
///
/// Loop mode: wrap from the tail (or last-valid-batch seed) into `frames[0]`.
/// Non-loop mode: emit zero MV (hard cut on shader wrap — correct for
/// one-shot effects that come to rest).
pub fn build_tail_flow_from_plan(
    plan: &BatchPlan,
    frames: &[ImageF32],
    opts: &GenerateOptions,
) -> Flow {
    let tail_refs: Option<Vec<&ImageF32>> = plan
        .tail_batch
        .as_ref()
        .map(|indices| indices.iter().map(|&i| &frames[i]).collect());
    let last_valid_refs: Option<Vec<&ImageF32>> = plan
        .last_valid_batch
        .as_ref()
        .map(|indices| indices.iter().map(|&i| &frames[i]).collect());
    build_tail_flow(
        opts.is_loop,
        tail_refs.as_deref().unwrap_or(&[]),
        last_valid_refs.as_deref(),
        &frames[0],
        opts,
    )
}

/// Build the final motion vector frame for the tail.
///
/// Loop mode: wrap from the tail (or last-valid-batch seed) into `frames[0]`.
/// Non-loop mode: zero MV (hard cut on shader wrap).
fn build_tail_flow(
    is_loop: bool,
    tail_batch: &[&ImageF32],
    last_valid_batch: Option<&[&ImageF32]>,
    first_frame: &ImageF32,
    opts: &GenerateOptions,
) -> Flow {
    if !is_loop {
        return Flow::zeros(first_frame.width, first_frame.height);
    }
    let mut wrap_refs: Vec<&ImageF32> = if tail_batch.is_empty() {
        last_valid_batch.map_or_else(|| vec![first_frame], |lvb| vec![lvb[lvb.len() - 1]])
    } else {
        tail_batch.to_vec()
    };
    wrap_refs.push(first_frame);
    let mut flow = accumulate_displacement(&wrap_refs, opts);
    normalize_flow(&mut flow);
    flow
}

/// Accumulate displacement using Lagrangian particle integration (Heun's method).
///
/// Per-pair Farneback flow is sampled at the *current particle position*
/// (not the static origin), so total displacement tracks correctly when
/// sub-frame motion is non-trivial.
fn accumulate_displacement(frames: &[&ImageF32], opts: &GenerateOptions) -> Flow {
    if frames.len() < 2 {
        if let Some(f) = frames.first() {
            return Flow::zeros(f.width, f.height);
        }
        return Flow::zeros(1, 1);
    }

    // If not analyzing skipped frames, collapse to [first, last]
    let effective_frames: Vec<&ImageF32> = if opts.analyze_skipped_frames {
        frames.to_vec()
    } else {
        vec![frames[0], frames[frames.len() - 1]]
    };

    let w = effective_frames[0].width;
    let h = effective_frames[0].height;
    let mut acc = Flow::zeros(w, h);

    for pair in effective_frames.windows(2) {
        // Per-step bidirectional combine: run forward and backward
        // Farneback on the pair and average the two displacement
        // fields before integrating. Doing this per-pair means
        // accumulation operates on already-corrected fields, so
        // per-step Farneback noise can't compound across the batch the
        // way it does when forward and backward are accumulated
        // separately and combined only at the output.
        let f = farneback(pair[0], pair[1], &opts.farneback);
        let b = farneback(pair[1], pair[0], &opts.farneback);
        let flow = crate::pipeline::bidirectional::combine_bidirectional(&f, &b);
        apply_step(&mut acc, &flow);
    }

    acc
}

/// Heun's method (improved Euler) integration step.
///
/// Averages flow at `p_i` and `p_i + f(p_i)`, dropping integration error
/// from O(dt) to O(dt²). Visible improvement on divergent/convergent motion
/// (explosions, implosions).
fn apply_step(acc: &mut Flow, flow: &Flow) {
    let w = acc.width;
    let h = acc.height;
    for y in 0..h {
        for x in 0..w {
            let [adx, ady] = *acc.at(x, y);
            let sx = x as f32 + adx;
            let sy = y as f32 + ady;
            let [k1x, k1y] = sample_remap(flow, sx, sy);
            let [k2x, k2y] = sample_remap(flow, sx + k1x, sy + k1y);
            let pixel = acc.at_mut(x, y);
            pixel[0] += 0.5 * (k1x + k2x);
            pixel[1] += 0.5 * (k1y + k2y);
        }
    }
}

/// Bicubic (Catmull-Rom) sampling of a flow field with zero-fill border.
///
/// Zero-fill border (`BORDER_CONSTANT(0)`): when particles drift out of
/// frame, sampled flow is zero and the trajectory pins to the boundary.
/// Do not substitute `BORDER_REPLICATE` or `BORDER_REFLECT` — those change
/// motion-vector behavior at frame edges.
fn sample_remap(flow: &Flow, x: f32, y: f32) -> [f32; 2] {
    let w = i32::try_from(flow.width).unwrap_or(i32::MAX);
    let h = i32::try_from(flow.height).unwrap_or(i32::MAX);

    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let fetch = |ix: i32, iy: i32| -> [f32; 2] {
        if ix < 0 || ix >= w || iy < 0 || iy >= h {
            [0.0, 0.0]
        } else {
            *flow.at(ix as u32, iy as u32)
        }
    };

    // Catmull-Rom basis function (a = -0.5)
    let cr = |t: f32| -> [f32; 4] {
        let t2 = t * t;
        let t3 = t2 * t;
        [
            (-0.5f32).mul_add(t3, 0.5f32.mul_add(-t, t2)),
            1.5f32.mul_add(t3, (-2.5f32).mul_add(t2, 1.0)),
            (-1.5f32).mul_add(t3, 2.0f32.mul_add(t2, 0.5 * t)),
            0.5f32.mul_add(t3, -0.5 * t2),
        ]
    };

    let wx = cr(fx);
    let wy = cr(fy);

    let mut result = [0.0f32; 2];
    for jj in 0..4i32 {
        let iy = y0 - 1 + jj;
        for ii in 0..4i32 {
            let ix = x0 - 1 + ii;
            let val = fetch(ix, iy);
            let weight = wx[ii as usize] * wy[jj as usize];
            result[0] = val[0].mul_add(weight, result[0]);
            result[1] = val[1].mul_add(weight, result[1]);
        }
    }

    result
}

/// Normalize flow: divide dx by width, dy by height (pixel units → normalized `[-1,1]` units).
///
/// This is load-bearing: encoding, `max_strength` tracking, and JSON metadata
/// all operate in normalized units. Skipping this divide would produce
/// `max_strength` in pixel units, yielding nearly-black motion atlases.
fn normalize_flow(flow: &mut Flow) {
    let w = flow.width as f32;
    let h = flow.height as f32;
    for pixel in &mut flow.data {
        pixel[0] /= w;
        pixel[1] /= h;
    }
}

/// Direction-preserving L∞ clip.
///
/// Uniformly scales each pixel so neither component exceeds `max_per_dim`,
/// preserving the vector's direction. (Independent per-axis clipping would
/// silently rotate fast vectors.)
pub fn clip_flow_linf(flow: &mut Flow, max_per_dim: f32) {
    for pixel in &mut flow.data {
        let comp_max = pixel[0].abs().max(pixel[1].abs());
        if comp_max > max_per_dim {
            let scale = max_per_dim / comp_max;
            pixel[0] *= scale;
            pixel[1] *= scale;
        }
    }
}

/// Update `max_strength` based on encoding method.
///
/// - `R8G8Remap01`: component-wise max across the entire sequence.
/// - `SidefxLabsR8G8`: Euclidean-magnitude max.
///
/// This value becomes the JSON `strength` field; shaders multiply by it to
/// undo normalization.
pub fn update_max_strength(max_strength: &mut f32, flow: &Flow, encoding: MotionVectorEncoding) {
    match encoding {
        MotionVectorEncoding::R8G8Remap01 => {
            // Component-wise max
            for pixel in &flow.data {
                *max_strength = max_strength.max(pixel[0].abs()).max(pixel[1].abs());
            }
        }
        MotionVectorEncoding::SidefxLabsR8G8 => {
            // Euclidean magnitude max
            for pixel in &flow.data {
                let mag = pixel[0].hypot(pixel[1]);
                *max_strength = max_strength.max(mag);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[test]
    fn clip_flow_linf_preserves_direction() {
        // (2.0, 0.5) with max_per_dim=1.0 → s = 1.0/2.0 = 0.5 → (1.0, 0.25)
        let mut flow = Flow::zeros(1, 1);
        flow.at_mut(0, 0)[0] = 2.0;
        flow.at_mut(0, 0)[1] = 0.5;
        clip_flow_linf(&mut flow, 1.0);
        let [dx, dy] = *flow.at(0, 0);
        assert!((dx - 1.0).abs() < 1e-6);
        assert!((dy - 0.25).abs() < 1e-6);
    }

    #[test]
    fn clip_flow_linf_no_clip_when_within_bounds() {
        let mut flow = Flow::zeros(1, 1);
        flow.at_mut(0, 0)[0] = 0.5;
        flow.at_mut(0, 0)[1] = 0.3;
        clip_flow_linf(&mut flow, 1.0);
        let [dx, dy] = *flow.at(0, 0);
        assert!((dx - 0.5).abs() < 1e-6);
        assert!((dy - 0.3).abs() < 1e-6);
    }

    #[test]
    fn normalize_flow_divides_by_dimensions() {
        // 100x200 image, pixel value (50, 100) → (50/100, 100/200) = (0.5, 0.5)
        let mut flow = Flow::zeros(100, 200);
        flow.at_mut(0, 0)[0] = 50.0;
        flow.at_mut(0, 0)[1] = 100.0;
        normalize_flow(&mut flow);
        let [dx, dy] = *flow.at(0, 0);
        assert!((dx - 0.5).abs() < 1e-6);
        assert!((dy - 0.5).abs() < 1e-6);
    }

    #[test]
    fn process_batch_slice_with_progress_stops_dispatch_after_cancel() {
        let frames = vec![
            ImageF32 {
                width: 16,
                height: 16,
                data: vec![0.0; 16 * 16],
            },
            ImageF32 {
                width: 16,
                height: 16,
                data: vec![1.0; 16 * 16],
            },
            ImageF32 {
                width: 16,
                height: 16,
                data: vec![2.0; 16 * 16],
            },
        ];
        let batches = vec![vec![0, 1], vec![1, 2]];
        let opts = GenerateOptions::default();
        let cancelled = AtomicBool::new(false);
        let started = AtomicUsize::new(0);
        let should_continue = || {
            let count = started.fetch_add(1, Ordering::Relaxed);
            if count == 0 {
                cancelled.store(true, Ordering::Relaxed);
                true
            } else {
                !cancelled.load(Ordering::Relaxed)
            }
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("test thread pool");

        let flows = pool.install(|| {
            process_batch_slice_with_progress(&batches, &frames, &opts, &should_continue, &|| {})
        });

        assert_eq!(flows.len(), 1);
    }

    #[test]
    fn normalize_flow_does_not_clip() {
        let mut flow = Flow::zeros(10, 10);
        flow.at_mut(0, 0)[0] = 20.0; // 20/10 = 2.0, must NOT be clipped
        flow.at_mut(0, 0)[1] = -20.0; // -20/10 = -2.0, must NOT be clipped
        normalize_flow(&mut flow);
        let [dx, dy] = *flow.at(0, 0);
        assert!((dx - 2.0).abs() < 1e-6);
        assert!((dy - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn zero_motion_guard() {
        // All-zero flow → max_strength stays 0
        let flow = Flow::zeros(4, 4);
        let mut max_strength = 0.0f32;
        update_max_strength(&mut max_strength, &flow, MotionVectorEncoding::R8G8Remap01);
        assert!((max_strength - 0.0).abs() < 1e-8);
    }

    #[test]
    fn update_max_strength_r8g8() {
        let mut flow = Flow::zeros(2, 2);
        flow.at_mut(0, 0)[0] = 0.3;
        flow.at_mut(1, 0)[1] = -0.7;
        let mut max_strength = 0.0f32;
        update_max_strength(&mut max_strength, &flow, MotionVectorEncoding::R8G8Remap01);
        assert!((max_strength - 0.7).abs() < 1e-6);
    }

    #[test]
    fn update_max_strength_sidefx() {
        let mut flow = Flow::zeros(2, 2);
        flow.at_mut(0, 0)[0] = 3.0;
        flow.at_mut(0, 0)[1] = 4.0; // magnitude = 5.0
        let mut max_strength = 0.0f32;
        update_max_strength(
            &mut max_strength,
            &flow,
            MotionVectorEncoding::SidefxLabsR8G8,
        );
        assert!((max_strength - 5.0).abs() < 1e-6);
    }

    #[test]
    fn sample_remap_oob_returns_zero() {
        let mut flow = Flow::zeros(2, 2);
        flow.at_mut(0, 0)[0] = 10.0;
        let [dx, dy] = sample_remap(&flow, -10.0, -10.0);
        assert!((dx - 0.0).abs() < 1e-6);
        assert!((dy - 0.0).abs() < 1e-6);
    }

    #[test]
    fn sample_remap_lattice_point_returns_value() {
        let mut flow = Flow::zeros(4, 4);
        flow.at_mut(2, 1)[0] = 7.5;
        flow.at_mut(2, 1)[1] = -3.25;
        let [dx, dy] = sample_remap(&flow, 2.0, 1.0);
        assert!((dx - 7.5).abs() < 1e-5);
        assert!((dy - (-3.25)).abs() < 1e-5);
    }
}
