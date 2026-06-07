//! Bake-time bidirectional flow combine for motion vectors.
//!
//! For each output frame, combines forward (F0→F1) and backward (F1→F0)
//! Farneback flows into a single MV channel by averaging the
//! displacement vectors:
//!
//! ```text
//! V = 0.5 · (F + (−B))
//! ```
//!
//! Alpha is not consulted. Where the source frames are transparent and
//! premultiplied, the gray-frame input to Farneback is zero, so F and
//! −B are zero and the average is zero naturally — no artificial gate,
//! no inside/outside transition line.
//!
//! Both flows are produced by the same Farneback implementation, just
//! with the input frame pair reversed for B. Output is in the same
//! normalized `[-1, 1]` coordinate space as the standard forward flow.

use crate::pipeline::Flow;

/// Combine forward and backward flows into a single MV by averaging
/// the displacement vectors. See module docs.
///
/// `forward` and `backward` must share dimensions.
pub fn combine_bidirectional(forward: &Flow, backward: &Flow) -> Flow {
    let width = forward.width;
    let height = forward.height;
    let pixels = (width as usize) * (height as usize);
    debug_assert_eq!(forward.data.len(), pixels);
    debug_assert_eq!(backward.width, width);
    debug_assert_eq!(backward.height, height);
    debug_assert_eq!(backward.data.len(), pixels);

    let mut data = vec![[0.0f32; 2]; pixels];
    for (idx, out) in data.iter_mut().enumerate() {
        let fwd = forward.data[idx];
        let bwd = backward.data[idx];
        *out = [0.5 * (fwd[0] - bwd[0]), 0.5 * (fwd[1] - bwd[1])];
    }

    Flow {
        width,
        height,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flow_with(value: [f32; 2], w: u32, h: u32) -> Flow {
        Flow {
            width: w,
            height: h,
            data: vec![value; (w * h) as usize],
        }
    }

    #[test]
    fn parallel_same_direction_averages() {
        // F=[1, 0], -B=[3, 0] → V = 0.5·(1+3, 0) = (2, 0).
        let f = flow_with([1.0, 0.0], 1, 1);
        let b = flow_with([-3.0, 0.0], 1, 1);
        let v = combine_bidirectional(&f, &b);
        assert!((v.data[0][0] - 2.0).abs() < 1e-5, "x={}", v.data[0][0]);
        assert!(v.data[0][1].abs() < 1e-5, "y={}", v.data[0][1]);
    }

    #[test]
    fn perpendicular_averages_componentwise() {
        // F=[1, 0], -B=[0, 1] → V = (0.5, 0.5).
        let f = flow_with([1.0, 0.0], 1, 1);
        let b = flow_with([0.0, -1.0], 1, 1);
        let v = combine_bidirectional(&f, &b);
        assert!((v.data[0][0] - 0.5).abs() < 1e-5, "x={}", v.data[0][0]);
        assert!((v.data[0][1] - 0.5).abs() < 1e-5, "y={}", v.data[0][1]);
    }

    #[test]
    fn antiparallel_partially_cancels() {
        // F=[1, 0], -B=[-2, 0] → V = 0.5·(1−2, 0) = (−0.5, 0).
        let f = flow_with([1.0, 0.0], 1, 1);
        let b = flow_with([2.0, 0.0], 1, 1);
        let v = combine_bidirectional(&f, &b);
        assert!((v.data[0][0] - (-0.5)).abs() < 1e-5, "x={}", v.data[0][0]);
        assert!(v.data[0][1].abs() < 1e-5, "y={}", v.data[0][1]);
    }

    #[test]
    fn near_zero_side_halves_other() {
        // F=[0.5, 0], -B≈0 → V ≈ (0.25, 0).
        let f = flow_with([0.5, 0.0], 1, 1);
        let b = flow_with([1e-9, 0.0], 1, 1);
        let v = combine_bidirectional(&f, &b);
        assert!((v.data[0][0] - 0.25).abs() < 1e-5, "x={}", v.data[0][0]);
        assert!(v.data[0][1].abs() < 1e-5, "y={}", v.data[0][1]);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn zero_inputs_yield_zero() {
        let f = flow_with([0.0, 0.0], 1, 1);
        let b = flow_with([0.0, 0.0], 1, 1);
        let v = combine_bidirectional(&f, &b);
        assert_eq!(v.data[0][0], 0.0);
        assert_eq!(v.data[0][1], 0.0);
    }
}
