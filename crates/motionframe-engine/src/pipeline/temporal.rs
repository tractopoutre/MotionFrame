//! Temporal smoothing of the per-output-frame motion-vector sequence.
//!
//! Filters `Vec<Flow>` (one entry per output frame, normalized + L∞-clipped)
//! to reduce neighbor-to-neighbor MV jumps that read as "stepping" in the
//! preview and in downstream consumers like Unity.

use crate::pipeline::Flow;
use rayon::prelude::*;

/// Centre-pixel motion² threshold below which `temporal_smooth` leaves the
/// pixel untouched. The smoothing pass exists to soften jumps in motion
/// direction across frames; it must not invent motion in regions that
/// genuinely had none, otherwise destination-side `−B` values from one
/// frame's bidirectional combine bleed into static neighbours and ghost
/// the moving region.
///
/// Normalised flow space — `1e-6` mag² ≈ 1/1024 of a frame dimension, well
/// below any meaningful Farneback signal but above rounding noise.
const NO_MOTION_EPS_MAG2: f32 = 1e-6;

/// Soft gate band on the *neighbour's warped sample magnitude*. Below
/// `WEIGHT_FADE_LO_MAG2` a neighbour contributes nothing; above
/// `WEIGHT_FADE_HI_MAG2` it contributes fully; smoothstep ramp in
/// between. This replaces the old binary bilateral cliff so the
/// blend/no-blend transition at the support boundary becomes a ramp
/// instead of a step.
///
/// `WEIGHT_FADE_LO_MAG2` is intentionally equal to `NO_MOTION_EPS_MAG2`
/// so the two notions of "no motion" agree: a centre below the
/// passthrough threshold and a sampled-neighbour at the gate's lower
/// edge are both at the same noise floor. If you tighten one, tighten
/// the other.
const WEIGHT_FADE_LO_MAG2: f32 = 1e-6;
const WEIGHT_FADE_HI_MAG2: f32 = 1e-4;

fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Bilinear sample with zero-fill border. `(x, y)` are pixel coordinates.
/// Caller must guarantee `width > 0 && height > 0`; `temporal_smooth`'s
/// outer loop is the only caller and structurally never enters the
/// inner sample if either dimension is zero.
fn sample_bilinear_zero(data: &[[f32; 2]], width: u32, height: u32, x: f32, y: f32) -> [f32; 2] {
    debug_assert!(
        width > 0 && height > 0,
        "sample_bilinear_zero: zero dimension"
    );
    debug_assert_eq!(data.len(), (width as usize) * (height as usize));
    let w = i32::try_from(width).unwrap_or(i32::MAX);
    let h = i32::try_from(height).unwrap_or(i32::MAX);
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let fetch = |ix: i32, iy: i32| -> [f32; 2] {
        if ix < 0 || ix >= w || iy < 0 || iy >= h {
            [0.0, 0.0]
        } else {
            data[(iy as usize) * (width as usize) + (ix as usize)]
        }
    };
    let f00 = fetch(x0, y0);
    let f10 = fetch(x1, y0);
    let f01 = fetch(x0, y1);
    let f11 = fetch(x1, y1);
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;
    [
        f00[0].mul_add(w00, f10[0].mul_add(w10, f01[0].mul_add(w01, f11[0] * w11))),
        f00[1].mul_add(w00, f10[1].mul_add(w10, f01[1].mul_add(w01, f11[1] * w11))),
    ]
}

/// Motion-compensated 3-tap binomial smoothing across the temporal axis.
///
/// For each pixel `p` in flow `i`, the centre's motion vector defines
/// where the same particle sat in flow `i−1` and where it'll sit in
/// `i+1`. The kernel samples each neighbour at its trajectory position:
///
/// ```text
///   left  = sample(MV[i-1], p − MV[i](p))
///   right = sample(MV[i+1], p + MV[i](p))
///   smoothed = (0.5·c + 0.25·w_l·left + 0.25·w_r·right) / Σweights
///   target   = smoothed · (|c| / |smoothed|)        // magnitude-preserving
///   result   = c + strength · (target − c)
/// ```
///
/// `MV[i]` is in normalised flow space (±1 = ±frame extent); the warp
/// scales by `(width, height)` to get pixel offsets. Bilinear sampling
/// with zero-fill at borders (so trajectories that exit the frame
/// contribute nothing rather than wrapping or clamping).
///
/// The neighbour weights `w_l, w_r` use a smoothstep magnitude gate on
/// the *sampled* neighbour value: a trajectory that lands on a
/// no-motion texel contributes nothing and the remaining weights are
/// re-normalised. This replaces an earlier binary bilateral exclusion
/// (which produced a hard cliff at the support boundary in expanding
/// regions like an explosion's outer ring).
///
/// **Magnitude-preserving rescale.** After the weighted vector mean,
/// the smoothed value is rescaled to the centre's magnitude before
/// strength-mixing. Without this, the soft gate's transition contour
/// (where neighbour magnitude is fading at the previous frame's
/// support edge) drags the result toward weaker vectors, producing a
/// visible magnitude dip + small step. Smoothing rotates direction;
/// magnitude stays equal to the centre. If the vector mean collapses
/// (e.g. anti-parallel neighbours cancel), the centre is preserved
/// instead.
///
/// Centre pixels with `|c|² < NO_MOTION_EPS_MAG2` pass through
/// unchanged — there's no trajectory to follow and no jump to soften.
///
/// - `strength = 0.0` is a no-op (early-return).
/// - `strength = 1.0` is the full filter.
/// - Values outside `[0, 1]` are clamped.
/// - `is_loop = true` wraps neighbour frame indices.
/// - `is_loop = false` drops the missing tap at the sequence boundary
///   and re-normalises the remaining weights.
///
/// All flows must have identical `width` and `height`. Mixed dimensions are a
/// programmer error and will panic with a descriptive message in debug builds.
#[allow(clippy::many_single_char_names)]
pub fn temporal_smooth(flows: &mut [Flow], strength: f32, is_loop: bool) {
    let strength = strength.clamp(0.0, 1.0);
    if strength == 0.0 || flows.len() < 2 {
        return;
    }

    // Sanity: all flows must share dimensions. Cheap O(N) check.
    let (width, height) = (flows[0].width, flows[0].height);
    debug_assert!(
        flows
            .iter()
            .all(|flow| flow.width == width && flow.height == height),
        "temporal_smooth: all flows must share width/height"
    );

    let count = flows.len();
    debug_assert!(count >= 2, "early-return covers count < 2");
    let w_f = width as f32;
    let h_f = height as f32;

    // Snapshot input once so we can read neighbors while writing each flow.
    let snapshot: Vec<Vec<[f32; 2]>> = flows.iter().map(|flow| flow.data.clone()).collect();

    // None at sequence boundary in non-loop mode (tap is dropped).
    let neighbor_left = |idx: usize| -> Option<usize> {
        if idx == 0 {
            if is_loop {
                Some(count - 1)
            } else {
                None
            }
        } else {
            Some(idx - 1)
        }
    };
    let neighbor_right = |idx: usize| -> Option<usize> {
        if idx + 1 == count {
            if is_loop {
                Some(0)
            } else {
                None
            }
        } else {
            Some(idx + 1)
        }
    };

    let smooth_one = |idx: usize, dst: &mut [[f32; 2]]| {
        let center = &snapshot[idx];
        let left = neighbor_left(idx).map(|i| &snapshot[i]);
        let right = neighbor_right(idx).map(|i| &snapshot[i]);

        for py in 0..height {
            for px in 0..width {
                let i = (py as usize) * (width as usize) + (px as usize);
                let c = center[i];
                let c_mag2 = c[0].mul_add(c[0], c[1] * c[1]);
                if c_mag2 < NO_MOTION_EPS_MAG2 {
                    // No trajectory to follow.
                    dst[i] = c;
                    continue;
                }

                // Convert centre's normalised flow to pixel offsets.
                let dx_pix = c[0] * w_f;
                let dy_pix = c[1] * h_f;
                let cx = px as f32;
                let cy = py as f32;

                // Trajectory-aligned neighbour samples.
                let sample_neighbor = |frame: &[[f32; 2]], wx: f32, wy: f32| -> [f32; 2] {
                    sample_bilinear_zero(frame, width, height, wx, wy)
                };
                let l = left.map(|frame| sample_neighbor(frame, cx - dx_pix, cy - dy_pix));
                let r = right.map(|frame| sample_neighbor(frame, cx + dx_pix, cy + dy_pix));

                // Soft magnitude gate on the sampled neighbour values.
                let weight = |v: [f32; 2]| -> f32 {
                    let mag2 = v[0].mul_add(v[0], v[1] * v[1]);
                    smoothstep(WEIGHT_FADE_LO_MAG2, WEIGHT_FADE_HI_MAG2, mag2)
                };
                let w_l = l.map_or(0.0, weight);
                let w_r = r.map_or(0.0, weight);

                let mut sum_x = 0.5 * c[0];
                let mut sum_y = 0.5 * c[1];
                let mut weight_total = 0.5;
                if let Some(l) = l {
                    let k = 0.25 * w_l;
                    sum_x += k * l[0];
                    sum_y += k * l[1];
                    weight_total += k;
                }
                if let Some(r) = r {
                    let k = 0.25 * w_r;
                    sum_x += k * r[0];
                    sum_y += k * r[1];
                    weight_total += k;
                }
                let inv = 1.0 / weight_total;
                let smoothed_x = sum_x * inv;
                let smoothed_y = sum_y * inv;

                // Magnitude-preserving rescale: smoothing rotates the
                // direction of `c` but never changes its magnitude. This
                // prevents the previous frame's edge fade from dragging
                // the result down — the soft gate's transition contour
                // would otherwise leak weak-magnitude neighbour samples
                // into the centre, producing a visible dip + jump along
                // the previous-MV's support boundary.
                let smoothed_mag2 = smoothed_x.mul_add(smoothed_x, smoothed_y * smoothed_y);
                let target = if smoothed_mag2 < NO_MOTION_EPS_MAG2 {
                    // Vector mean collapsed (e.g. anti-parallel neighbours
                    // cancel). No reliable direction — keep centre.
                    c
                } else {
                    let scale = (c_mag2 / smoothed_mag2).sqrt();
                    [smoothed_x * scale, smoothed_y * scale]
                };
                dst[i][0] = c[0] + strength * (target[0] - c[0]);
                dst[i][1] = c[1] + strength * (target[1] - c[1]);
            }
        }
    };

    flows
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, flow)| smooth_one(idx, &mut flow.data));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flow_with_value(w: u32, h: u32, v: [f32; 2]) -> Flow {
        Flow {
            width: w,
            height: h,
            data: vec![v; (w * h) as usize],
        }
    }

    fn flow_zeros(w: u32, h: u32) -> Flow {
        flow_with_value(w, h, [0.0, 0.0])
    }

    fn set(flow: &mut Flow, x: u32, y: u32, v: [f32; 2]) {
        let i = (y as usize) * (flow.width as usize) + (x as usize);
        flow.data[i] = v;
    }

    fn at(flow: &Flow, x: u32, y: u32) -> [f32; 2] {
        let i = (y as usize) * (flow.width as usize) + (x as usize);
        flow.data[i]
    }

    /// `strength = 0` returns input bit-exactly.
    #[test]
    fn zero_strength_is_identity() {
        let mut flows = vec![
            flow_with_value(2, 2, [0.1, 0.2]),
            flow_with_value(2, 2, [0.5, -0.3]),
            flow_with_value(2, 2, [-0.2, 0.4]),
        ];
        let original = flows.clone();
        temporal_smooth(&mut flows, 0.0, false);
        for (i, (got, want)) in flows.iter().zip(original.iter()).enumerate() {
            assert_eq!(got.data, want.data, "flow {i} mutated at strength 0");
        }
    }

    /// Centre pixel with no motion is passed through, even if neighbours have motion.
    /// No trajectory to follow → don't invent motion.
    #[test]
    #[allow(clippy::float_cmp)]
    fn zero_center_is_preserved() {
        let mut flows = vec![
            flow_with_value(1, 1, [0.5, -0.3]),
            flow_with_value(1, 1, [0.0, 0.0]),
            flow_with_value(1, 1, [-0.4, 0.2]),
        ];
        temporal_smooth(&mut flows, 1.0, false);
        assert_eq!(flows[1].data[0], [0.0, 0.0]);
    }

    /// MCTF samples neighbours along the centre's trajectory. Three frames
    /// of width 4: a particle moves +1 pixel per frame at row 0. The flow
    /// at the particle's location is `[0.25, 0.0]` (= 1/W in normalized
    /// units). MCTF smoothing of the centre frame at the particle pixel
    /// must look up frame 0 at column-1 and frame 2 at column+1, both of
    /// which carry the same flow → smoothed value equals input.
    ///
    /// A naive same-position kernel would instead look at the particle's
    /// current column in the neighbour frames, find zero motion there,
    /// soft-gate them out, and produce only the centre — which happens
    /// to also equal input here, so we additionally check a discriminating
    /// case below.
    #[test]
    fn trajectory_lookup_along_advection() {
        let v = [0.25_f32, 0.0_f32]; // +1 pixel per frame at width 4
        let mut frames = vec![flow_zeros(4, 1), flow_zeros(4, 1), flow_zeros(4, 1)];
        set(&mut frames[0], 0, 0, v);
        set(&mut frames[1], 1, 0, v);
        set(&mut frames[2], 2, 0, v);
        temporal_smooth(&mut frames, 1.0, false);
        // Centre's trajectory landed on the same flow value in both
        // neighbours → smoothed = v exactly.
        let got = at(&frames[1], 1, 0);
        assert!(
            (got[0] - v[0]).abs() < 1e-6 && (got[1] - v[1]).abs() < 1e-6,
            "got {got:?}"
        );
    }

    /// Discriminating test: MCTF rotates direction toward the trajectory
    /// neighbour but preserves the centre's magnitude.
    /// Frame 1 has motion `[0.25, 0]` at (1,0). Frame 0 has motion
    /// `[0, 0.25]` at the trajectory location (0,0) and zero at the same
    /// position (1,0). A naive same-position kernel would see no
    /// neighbour and leave the centre at `[0.25, 0]`. MCTF picks up the
    /// rotated neighbour and rotates the centre toward it; magnitude
    /// stays exactly `0.25`.
    #[test]
    fn trajectory_lookup_rotates_direction_preserves_magnitude() {
        let mut frames = vec![flow_zeros(4, 1), flow_zeros(4, 1)];
        set(&mut frames[0], 0, 0, [0.0, 0.25]); // trajectory neighbour: rotated 90°
        set(&mut frames[1], 1, 0, [0.25, 0.0]); // centre
        temporal_smooth(&mut frames, 1.0, false);

        let got = at(&frames[1], 1, 0);

        // Magnitude must equal centre's exactly.
        let got_mag2 = got[0].mul_add(got[0], got[1] * got[1]);
        assert!(
            (got_mag2 - 0.0625).abs() < 1e-6,
            "magnitude drift: got {got:?}"
        );

        // Direction rotated from pure +x toward +y.
        assert!(
            got[0] > 0.0 && got[1] > 0.0,
            "expected first quadrant: got {got:?}"
        );
        assert!(
            got[0] < 0.25,
            "expected x rotated below centre: got {got:?}"
        );

        // Exact analytical value: sum = (0.125, 0.0625), weight_total = 0.75
        //   smoothed = (1/6, 1/12), |smoothed| = sqrt(5)/12
        //   scale = 0.25 / (sqrt(5)/12) = 3/sqrt(5)
        //   target = (sqrt(5)/10, sqrt(5)/20) ≈ (0.2236, 0.1118)
        let s = 5.0_f32.sqrt();
        let target_x = s / 10.0;
        let target_y = s / 20.0;
        assert!((got[0] - target_x).abs() < 1e-6, "got {}", got[0]);
        assert!((got[1] - target_y).abs() < 1e-6, "got {}", got[1]);
    }

    /// Magnitude of the result equals magnitude of the centre exactly,
    /// regardless of neighbour magnitudes (the dip-and-jump artifact
    /// would manifest as a magnitude < |c|, so this is the load-bearing
    /// invariant).
    #[test]
    fn magnitude_preserving_under_weak_neighbour() {
        // Centre has full magnitude; trajectory neighbour is fading
        // (e.g. previous frame's edge). The fading-magnitude pull-down
        // is precisely the artifact magnitude-preservation cures.
        let mut frames = vec![flow_zeros(4, 1), flow_zeros(4, 1)];
        // Trajectory neighbour: same direction as centre but tiny magnitude
        // (above the soft-gate's HI band so weight is full but value is weak).
        set(&mut frames[0], 0, 0, [0.02, 0.0]);
        set(&mut frames[1], 1, 0, [0.25, 0.0]);
        temporal_smooth(&mut frames, 1.0, false);

        let got = at(&frames[1], 1, 0);
        let got_mag2 = got[0].mul_add(got[0], got[1] * got[1]);
        assert!(
            (got_mag2 - 0.0625).abs() < 1e-6,
            "magnitude must match centre's 0.0625, got {got_mag2} (vec {got:?})",
        );
        // Direction unchanged (centre and neighbour both pure +x).
        assert!((got[1]).abs() < 1e-6);
        assert!((got[0] - 0.25).abs() < 1e-6, "got {}", got[0]);
    }

    /// Trajectory that walks off-frame: bilinear returns zero, soft gate
    /// excludes that tap, remaining weights re-normalize. Edge pixels
    /// gracefully degrade instead of getting stuck on stale values.
    #[test]
    fn trajectory_off_frame_drops_tap() {
        // Width 2, 2 frames, non-loop. Centre flow is [1.0, 0] → pixel
        // offset 2 → trajectory in frame 0 is at (-1, 0), fully OOB.
        let mut frames = vec![flow_zeros(2, 1), flow_zeros(2, 1)];
        set(&mut frames[0], 0, 0, [1.0, 0.0]); // would-be trajectory neighbour, but unreachable
        set(&mut frames[1], 1, 0, [1.0, 0.0]); // centre
        temporal_smooth(&mut frames, 1.0, false);

        // Left lookup OOB → [0,0] → soft gate → w_l = 0 → tap dropped.
        // Right tap also dropped (non-loop boundary).
        // weight_total = 0.5, smoothed = c / 1 (after re-norm with 0.5 only? no:)
        //   sum_x = 0.5*1.0; weight_total = 0.5; smoothed = 1.0. dst = 1.0.
        let got = at(&frames[1], 1, 0);
        assert!((got[0] - 1.0).abs() < 1e-6, "got {got:?}");
    }

    /// Loop mode wraps the neighbour-frame index. The wrapped frame's
    /// sample location lies *outside* the frame in this setup, so the
    /// soft gate zeros it out and only the in-frame left tap
    /// contributes — exercising the wrap-but-OOB combination.
    /// `loop_wrap_in_bounds_sample` below covers the wrap-and-sample
    /// case.
    #[test]
    fn trajectory_loop_wrap() {
        // Width 4, 4 frames, loop. Particle with v=[0.25,0] cycles
        // (0,0) → (1,0) → (2,0) → (3,0) → wrap to (0,0).
        let v = [0.25_f32, 0.0_f32];
        let mut frames = vec![
            flow_zeros(4, 1),
            flow_zeros(4, 1),
            flow_zeros(4, 1),
            flow_zeros(4, 1),
        ];
        set(&mut frames[0], 0, 0, v);
        set(&mut frames[1], 1, 0, v);
        set(&mut frames[2], 2, 0, v);
        set(&mut frames[3], 3, 0, v);
        temporal_smooth(&mut frames, 1.0, true);

        // Frame 3 (index 3) at (3, 0): trajectory points to (4, 0) on
        // frame 0 (right wrap). (4,0) is OOB so right contribution is
        // bilinear-fetched as zero → soft gate excludes it. Left tap on
        // frame 2 at (2, 0) = v. Centre + left only.
        // smoothed = (0.5*v + 0.25*v) / 0.75 = v. ✓
        let got = at(&frames[3], 3, 0);
        assert!((got[0] - v[0]).abs() < 1e-6, "got {got:?}");
    }

    /// Loop mode actually pulls a sample from the wrap-target frame.
    /// Centre at frame 0 (column 0) has motion in `−x`; trajectory left
    /// warps to `(+1, 0)` (in-bounds) on frame `count−1`. The wrap
    /// target carries a 90°-rotated flow, so the smoothed direction
    /// must rotate into the second component — proving the wrap-and-
    /// sample path actually executed (no rotation would happen if the
    /// wrap were dropped).
    #[test]
    fn trajectory_loop_wrap_in_bounds_sample() {
        let mut frames = vec![
            flow_zeros(4, 1),
            flow_zeros(4, 1),
            flow_zeros(4, 1),
            flow_zeros(4, 1),
        ];
        // Centre at frame 0 (0, 0): motion `−0.25 x`; trajectory left
        // wraps to frame 3 at sample position (0 − (−0.25)·4, 0) = (1, 0).
        set(&mut frames[0], 0, 0, [-0.25, 0.0]);
        // Wrap-target sample: rotated 90° from centre direction.
        set(&mut frames[3], 1, 0, [0.0, -0.25]);
        temporal_smooth(&mut frames, 1.0, true);

        let got = at(&frames[0], 0, 0);

        // Magnitude preserved.
        let got_mag2 = got[0].mul_add(got[0], got[1] * got[1]);
        assert!((got_mag2 - 0.0625).abs() < 1e-6, "got {got:?}");

        // Direction rotated into −y (proves the wrapped-sample path ran).
        assert!(got[1] < -1e-3, "expected y rotated negative, got {got:?}");
        // x still negative (centre direction preserved partially).
        assert!(got[0] < 0.0, "got {got:?}");
    }

    /// Width/height/data length unchanged by the filter.
    #[test]
    fn preserves_dimensions() {
        let mut flows = vec![flow_with_value(7, 5, [0.3, -0.4]); 6];
        temporal_smooth(&mut flows, 0.7, false);
        for f in &flows {
            assert_eq!(f.width, 7);
            assert_eq!(f.height, 5);
            assert_eq!(f.data.len(), 35);
        }
    }

    /// Strength values outside `[0, 1]` are clamped (no NaN, no over-smoothing).
    #[test]
    fn strength_clamped() {
        let make = || {
            let mut frames = vec![flow_zeros(4, 1), flow_zeros(4, 1)];
            set(&mut frames[0], 0, 0, [0.5, 0.0]);
            set(&mut frames[1], 1, 0, [0.25, 0.0]);
            frames
        };
        let mut neg = make();
        let mut zero = make();
        let mut over = make();
        temporal_smooth(&mut neg, -0.5, false);
        temporal_smooth(&mut zero, 0.0, false);
        temporal_smooth(&mut over, 1.5, false);

        // strength <= 0 should match strength = 0 (identity early-return).
        for px in 0..4 {
            #[allow(clippy::float_cmp)] // both branches early-return → bit-exact
            {
                assert_eq!(at(&neg[1], px, 0), at(&zero[1], px, 0));
            }
        }
        // strength >= 1 should match strength = 1 exactly.
        let mut full = make();
        temporal_smooth(&mut full, 1.0, false);
        for px in 0..4 {
            let a = at(&over[1], px, 0);
            let b = at(&full[1], px, 0);
            assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6);
        }
    }
}
