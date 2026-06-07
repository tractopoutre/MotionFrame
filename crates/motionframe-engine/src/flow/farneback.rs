//! Farneback optical flow driver — coarse-to-fine pyramid iteration.
//!
//! Pure-Rust implementation derived from `OpenCV`'s `optflowgf.cpp`
//! (`calcOpticalFlowFarneback`, CPU non-fastPyramids path).
//! See THIRD-PARTY-LICENSES.md for license attribution.

use crate::flow::poly::poly_expansion;
use crate::flow::pyramid::{build_level_image, compute_num_levels};
use crate::flow::update::{update_flow_with_workspace, UpdateWorkspace};
use crate::pipeline::{FarnebackParams, Flow, ImageF32};
use rayon::prelude::*;

/// Compute dense optical flow between two grayscale frames using Farneback's algorithm.
///
/// Matches `OpenCV`'s `calcOpticalFlowFarneback` parameter semantics exactly.
///
/// Algorithm (non-fastPyramids path, matching `OpenCV` CPU code):
/// 1. Determine actual number of levels (clip with `min_size=32`).
/// 2. For each level k from coarsest to finest:
///    a. Blur the ORIGINAL image with `Gaussian(sigma = (1/scale - 1)/2)`.
///    b. Resize blurred image to level dimensions with bilinear interpolation.
///    c. Compute polynomial expansion on the resized image.
///    d. Upsample flow from previous level (or init to zero).
///    e. Run `iterations` update steps.
/// 3. Return flow at finest level (k=0).
///
/// Key: `OpenCV` computes poly expansion once per level and does NOT warp the image
/// between iterations. Instead, the current flow estimate appears in the residual
/// computation inside `update_flow`. The flow is OVERWRITTEN (not accumulated) each
/// iteration.
// Based on OpenCV's optflowgf.cpp: calcOpticalFlowFarneback
pub fn farneback(frame1: &ImageF32, frame2: &ImageF32, params: &FarnebackParams) -> Flow {
    let num_levels =
        compute_num_levels(frame1.width, frame1.height, params.levels, params.pyr_scale);
    let scale_factor = 1.0 / params.pyr_scale; // = 2.0 for pyr_scale=0.5

    let mut flow: Option<Flow> = None;

    // Pre-allocate workspace for the largest (finest) level, reused across all levels.
    let mut ws = UpdateWorkspace::new(frame1.width as usize, frame1.height as usize);

    // Process from coarsest (k=num_levels) to finest (k=0).
    // Based on OpenCV's optflowgf.cpp: for( k = levels; k >= 0; k-- )
    for level_idx in 0..=num_levels {
        let k = num_levels - level_idx; // k goes from num_levels down to 0

        // Build level images by blurring original and resizing (parallel for both frames)
        let (img1, img2) = rayon::join(
            || build_level_image(frame1, k, params.pyr_scale),
            || build_level_image(frame2, k, params.pyr_scale),
        );

        let level_w = img1.width;
        let level_h = img1.height;

        // Initialize or upsample flow
        // Based on OpenCV's optflowgf.cpp: flow init / resize logic
        match flow {
            None => {
                flow = Some(Flow::zeros(level_w, level_h));
            }
            Some(prev_flow) => {
                flow = Some(upsample_flow(&prev_flow, level_w, level_h, scale_factor));
            }
        }

        // can't fail: both match arms above set flow = Some(...)
        let current_flow = flow.as_mut().unwrap();

        // Compute polynomial expansion ONCE per level — parallel for both frames.
        let (poly1, poly2) = rayon::join(
            || poly_expansion(&img1, params.poly_n, params.poly_sigma),
            || poly_expansion(&img2, params.poly_n, params.poly_sigma),
        );

        // Run iterations at this level (workspace reused across iterations and levels).
        ws.ensure_size(level_w as usize, level_h as usize);
        for _iter in 0..params.iterations {
            update_flow_with_workspace(
                current_flow,
                &poly1,
                &poly2,
                params.winsize,
                params.use_gaussian,
                &mut ws,
            );
        }
    }

    flow.unwrap() // can't fail: loop runs at least once (0..=num_levels, num_levels >= 0)
}

/// Upsample a flow field to target dimensions using bilinear interpolation.
/// Flow values are multiplied by `scale_factor` to account for resolution change.
// Based on OpenCV's optflowgf.cpp: flow upsampling between pyramid levels
fn upsample_flow(flow: &Flow, target_w: u32, target_h: u32, scale_factor: f32) -> Flow {
    let mut result = Flow::zeros(target_w, target_h);
    let src_w = flow.width as f32;
    let src_h = flow.height as f32;
    let dst_w = target_w as f32;
    let dst_h = target_h as f32;
    let tw = target_w as usize;

    result
        .data
        .par_chunks_mut(tw)
        .enumerate()
        .for_each(|(y, row)| {
            let sy = (y as f32 + 0.5) * src_h / dst_h - 0.5;
            for (x, elem) in row.iter_mut().enumerate() {
                let sx = (x as f32 + 0.5) * src_w / dst_w - 0.5;
                let [fx, fy] = bilinear_sample_flow(flow, sx, sy);
                *elem = [fx * scale_factor, fy * scale_factor];
            }
        });

    result
}

/// Sample flow field at sub-pixel location using bilinear interpolation.
/// Border policy: clamp (matching `OpenCV`'s resize `BORDER_REPLICATE`).
// allow(cast_possible_wrap): width/height to i32 for signed coord math, always < i32::MAX
#[allow(clippy::cast_possible_wrap)]
fn bilinear_sample_flow(flow: &Flow, x: f32, y: f32) -> [f32; 2] {
    let w = flow.width as i32;
    let h = flow.height as i32;

    // Clamp to valid range
    let x = x.max(0.0).min((w - 1) as f32);
    let y = y.max(0.0).min((h - 1) as f32);

    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let v00 = flow.data[y0 as usize * w as usize + x0 as usize];
    let v10 = flow.data[y0 as usize * w as usize + x1 as usize];
    let v01 = flow.data[y1 as usize * w as usize + x0 as usize];
    let v11 = flow.data[y1 as usize * w as usize + x1 as usize];

    let dx = (v00[0] * (1.0 - fx)).mul_add(
        1.0 - fy,
        (v10[0] * fx).mul_add(
            1.0 - fy,
            (v01[0] * (1.0 - fx)).mul_add(fy, v11[0] * fx * fy),
        ),
    );
    let dy = (v00[1] * (1.0 - fx)).mul_add(
        1.0 - fy,
        (v10[1] * fx).mul_add(
            1.0 - fy,
            (v01[1] * (1.0 - fx)).mul_add(fy, v11[1] * fx * fy),
        ),
    );

    [dx, dy]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn farneback_zero_images_zero_flow() {
        let img1 = ImageF32::zeros(32, 32);
        let img2 = ImageF32::zeros(32, 32);
        let params = FarnebackParams {
            levels: 2,
            iterations: 2,
            winsize: 5,
            poly_n: 3,
            poly_sigma: 1.2,
            pyr_scale: 0.5,
            use_gaussian: false,
        };
        let flow = farneback(&img1, &img2, &params);
        assert_eq!(flow.width, 32);
        assert_eq!(flow.height, 32);
        for v in &flow.data {
            assert!(v[0].abs() < 1e-4);
            assert!(v[1].abs() < 1e-4);
        }
    }

    #[test]
    fn upsample_flow_doubles_size_and_values() {
        let mut flow = Flow::zeros(4, 4);
        // Set a uniform flow of (1.0, 0.5)
        for v in &mut flow.data {
            *v = [1.0, 0.5];
        }
        let up = upsample_flow(&flow, 8, 8, 2.0);
        assert_eq!(up.width, 8);
        assert_eq!(up.height, 8);
        // Interior values should be approximately (2.0, 1.0) after scaling
        let center = up.data[4 * 8 + 4];
        assert!((center[0] - 2.0).abs() < 0.2, "got {}", center[0]);
        assert!((center[1] - 1.0).abs() < 0.2, "got {}", center[1]);
    }
}
