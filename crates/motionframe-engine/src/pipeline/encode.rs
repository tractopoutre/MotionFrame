//! Motion vector encoding: R8G8 symmetric remap and `SideFX` Labs polar format.
//!
//! Both paths short-circuit when `max_strength < 1e-8` (zero-motion guard
//! prevents NaN from `1/max_strength` on all-static or fully-masked sequences).
//! The Y axis is flipped before encoding for shader convention; the flip is
//! local so the caller's flow is never mutated.

use crate::pipeline::{Flow, ImageRgba8};

/// Encode flow as R8G8 remap [0,1] format.
///
/// Formula: `byte = (component / (2 * max_strength) + 0.5) * 255`.
/// Zero motion encodes as mid-gray (127).
pub fn encode_r8g8_remap(flow: &Flow, max_strength: f32) -> ImageRgba8 {
    let w = flow.width;
    let h = flow.height;

    // Zero-motion guard
    if max_strength < 1e-8 {
        return ImageRgba8::zeros(w, h);
    }

    let mut out = ImageRgba8 {
        width: w,
        height: h,
        data: vec![0u8; (w as usize) * (h as usize) * 4],
    };

    let two_max = 2.0 * max_strength;
    for y in 0..h {
        for x in 0..w {
            let [dx, dy] = *flow.at(x, y);
            // Y-flip: negate dy (not in-place; flow is &Flow)
            let neg_dy = -dy;
            // Formula: (flow / (2*max) + 0.5) * 255, then clip+truncate
            let r_f = (dx / two_max + 0.5) * 255.0;
            let g_f = (neg_dy / two_max + 0.5) * 255.0;
            let r = r_f.clamp(0.0, 255.0) as u8;
            let g = g_f.clamp(0.0, 255.0) as u8;
            let idx = ((y as usize) * (w as usize) + (x as usize)) * 4;
            out.data[idx] = r;
            out.data[idx + 1] = g;
            out.data[idx + 2] = 0;
            out.data[idx + 3] = 255;
        }
    }

    out
}

/// Encode flow as `SideFX` Labs R8G8 polar format.
///
/// Red = 9-bit angle (low 8 bits visible, top bit packed as flip into green).
/// Green = 7-bit magnitude | (polar-flip << 7). Magnitude-zero zeroes the
/// angle to avoid decoding artifacts.
pub fn encode_sidefx_labs(flow: &Flow, max_strength: f32) -> ImageRgba8 {
    let w = flow.width;
    let h = flow.height;

    // Zero-motion guard
    if max_strength < 1e-8 {
        return ImageRgba8::zeros(w, h);
    }

    let mut out = ImageRgba8 {
        width: w,
        height: h,
        data: vec![0u8; (w as usize) * (h as usize) * 4],
    };

    for y in 0..h {
        for x in 0..w {
            let [dx, dy] = *flow.at(x, y);
            // Y-flip: negate dy for angle computation
            let neg_dy = -dy;

            let magnitude = dx.hypot(neg_dy);

            let (angle_byte, g_byte) = if magnitude < 1e-8 {
                (0u8, 0u8)
            } else {
                let normalized_magnitude = magnitude / max_strength;

                // atan2(dy_flipped, dx) → range [-π, π]; normalize to [0, 2π) then [0, 1)
                let angle = neg_dy.atan2(dx);
                let normalized_angle =
                    angle.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU;

                // Map to [0, 511]
                let scaled_angle = (normalized_angle * 511.0).round() as u32;
                let flip_bit = (scaled_angle >> 8) & 1;
                let angle_bits = (scaled_angle & 0xFF) as u8;

                let magnitude_bits = (normalized_magnitude * 127.0).round().min(127.0) as u8;
                let g = magnitude_bits | ((flip_bit as u8) << 7);

                (angle_bits, g)
            };

            let idx = ((y as usize) * (w as usize) + (x as usize)) * 4;
            out.data[idx] = angle_byte;
            out.data[idx + 1] = g_byte;
            out.data[idx + 2] = 0;
            out.data[idx + 3] = 255;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_r8g8_remap_known_values() {
        // max_strength=1.0
        // (0,0) → r = (0/2+0.5)*255 = 127.5 → 127 (truncated), g = 127
        // (1,0) → r = (1/2+0.5)*255 = 255, g = 127
        // (-1,0) → r = (-1/2+0.5)*255 = 0, g = 127
        let mut flow = Flow::zeros(3, 1);
        flow.at_mut(0, 0)[0] = 0.0;
        flow.at_mut(0, 0)[1] = 0.0;
        flow.at_mut(1, 0)[0] = 1.0;
        flow.at_mut(1, 0)[1] = 0.0;
        flow.at_mut(2, 0)[0] = -1.0;
        flow.at_mut(2, 0)[1] = 0.0;

        let img = encode_r8g8_remap(&flow, 1.0);

        // pixel 0: (127, 127, 0, 255) — zero motion → 127.5 truncated to 127
        assert_eq!(img.data[0], 127); // r
        assert_eq!(img.data[1], 127); // g
        assert_eq!(img.data[2], 0);
        assert_eq!(img.data[3], 255);

        // pixel 1: (255, 127, 0, 255) — max positive X
        assert_eq!(img.data[4], 255); // r
        assert_eq!(img.data[5], 127); // g

        // pixel 2: (0, 127, 0, 255) — max negative X
        assert_eq!(img.data[8], 0); // r
        assert_eq!(img.data[9], 127); // g
    }

    #[test]
    fn encode_r8g8_remap_covers_y_axis_and_diagonal_motion() {
        let mut flow = Flow::zeros(4, 1);
        flow.at_mut(0, 0)[0] = 0.0;
        flow.at_mut(0, 0)[1] = 1.0;
        flow.at_mut(1, 0)[0] = 0.0;
        flow.at_mut(1, 0)[1] = -1.0;
        flow.at_mut(2, 0)[0] = 1.0;
        flow.at_mut(2, 0)[1] = 1.0;
        flow.at_mut(3, 0)[0] = -1.0;
        flow.at_mut(3, 0)[1] = -1.0;

        let img = encode_r8g8_remap(&flow, 1.0);

        assert_eq!(&img.data[0..4], &[127, 0, 0, 255]);
        assert_eq!(&img.data[4..8], &[127, 255, 0, 255]);
        assert_eq!(&img.data[8..12], &[255, 0, 0, 255]);
        assert_eq!(&img.data[12..16], &[0, 255, 0, 255]);
    }

    #[test]
    fn encode_r8g8_remap_zero_motion_guard() {
        let flow = Flow::zeros(2, 2);
        let img = encode_r8g8_remap(&flow, 0.0);
        assert!(img.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn encode_sidefx_labs_known_values() {
        // Pure rightward motion: dx=1, dy=0 → angle=atan2(0,1)=0, magnitude=1.0/1.0=1.0
        // normalized_angle = 0/(2π) = 0, scaled_angle = 0*511 = 0
        // flip_bit = 0, angle_bits = 0
        // magnitude_bits = (1.0*127).round().min(127) = 127
        // g = 127 | (0<<7) = 127
        let mut flow = Flow::zeros(1, 1);
        flow.at_mut(0, 0)[0] = 1.0;
        flow.at_mut(0, 0)[1] = 0.0;

        let img = encode_sidefx_labs(&flow, 1.0);
        assert_eq!(img.data[0], 0); // R = angle_byte
        assert_eq!(img.data[1], 127); // G = magnitude | flip
        assert_eq!(img.data[2], 0);
        assert_eq!(img.data[3], 255);
    }

    #[test]
    fn encode_sidefx_labs_zero_motion_guard() {
        let flow = Flow::zeros(2, 2);
        let img = encode_sidefx_labs(&flow, 0.0);
        assert!(img.data.iter().all(|&b| b == 0));
    }
}
