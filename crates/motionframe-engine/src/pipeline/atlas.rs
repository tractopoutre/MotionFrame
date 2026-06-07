//! Atlas blitting, extrusion, and Nyquist-safe resize.

use crate::pipeline::{ImageRgba8, Interpolation};

/// Copy tile into atlas and extrude edges outward.
///
/// Each tile gets `extrude` pixels of edge-replicate padding (top, bottom,
/// left, right, corners). Prevents UV bleed when shaders sample near tile
/// boundaries with linear filtering.
pub fn blit_extrude(
    atlas: &mut ImageRgba8,
    tile: &ImageRgba8,
    tile_x: u32,
    tile_y: u32,
    extrude: u32,
) {
    let tw = tile.width;
    let th = tile.height;
    let aw = atlas.width;

    // Copy tile body into atlas at offset (tile_x + extrude, tile_y + extrude)
    let ox = tile_x + extrude;
    let oy = tile_y + extrude;
    for row in 0..th {
        for col in 0..tw {
            let src_idx = ((row as usize) * (tw as usize) + (col as usize)) * 4;
            let dst_x = ox + col;
            let dst_y = oy + row;
            let dst_idx = ((dst_y as usize) * (aw as usize) + (dst_x as usize)) * 4;
            atlas.data[dst_idx..dst_idx + 4].copy_from_slice(&tile.data[src_idx..src_idx + 4]);
        }
    }

    if extrude == 0 {
        return;
    }

    // Top extrusion: replicate first row of tile upward
    for ey in 0..extrude {
        for col in 0..tw {
            let src_idx = (col as usize) * 4; // row 0 of tile
            let dst_x = ox + col;
            let dst_y = tile_y + ey;
            let dst_idx = ((dst_y as usize) * (aw as usize) + (dst_x as usize)) * 4;
            atlas.data[dst_idx..dst_idx + 4].copy_from_slice(&tile.data[src_idx..src_idx + 4]);
        }
    }

    // Bottom extrusion: replicate last row of tile downward
    for ey in 0..extrude {
        for col in 0..tw {
            let src_idx = (((th - 1) as usize) * (tw as usize) + (col as usize)) * 4;
            let dst_x = ox + col;
            let dst_y = oy + th + ey;
            let dst_idx = ((dst_y as usize) * (aw as usize) + (dst_x as usize)) * 4;
            atlas.data[dst_idx..dst_idx + 4].copy_from_slice(&tile.data[src_idx..src_idx + 4]);
        }
    }

    // Left extrusion: replicate first column leftward
    for row in 0..th {
        let src_idx = ((row as usize) * (tw as usize)) * 4; // col 0
        for ex in 0..extrude {
            let dst_x = tile_x + ex;
            let dst_y = oy + row;
            let dst_idx = ((dst_y as usize) * (aw as usize) + (dst_x as usize)) * 4;
            atlas.data[dst_idx..dst_idx + 4].copy_from_slice(&tile.data[src_idx..src_idx + 4]);
        }
    }

    // Right extrusion: replicate last column rightward
    for row in 0..th {
        let src_idx = ((row as usize) * (tw as usize) + ((tw - 1) as usize)) * 4;
        for ex in 0..extrude {
            let dst_x = ox + tw + ex;
            let dst_y = oy + row;
            let dst_idx = ((dst_y as usize) * (aw as usize) + (dst_x as usize)) * 4;
            atlas.data[dst_idx..dst_idx + 4].copy_from_slice(&tile.data[src_idx..src_idx + 4]);
        }
    }

    // Corner extrusions (extrude×extrude regions)
    // Top-left corner: tile[0,0]
    let tl: [u8; 4] = tile.data[0..4].try_into().unwrap(); // can't fail: slice is exactly 4 bytes
    for ey in 0..extrude {
        for ex in 0..extrude {
            let dst_x = tile_x + ex;
            let dst_y = tile_y + ey;
            let dst_idx = ((dst_y as usize) * (aw as usize) + (dst_x as usize)) * 4;
            atlas.data[dst_idx..dst_idx + 4].copy_from_slice(&tl);
        }
    }

    // Top-right corner: tile[0, tw-1]
    let tr_idx = ((tw - 1) as usize) * 4;
    let tr: [u8; 4] = tile.data[tr_idx..tr_idx + 4].try_into().unwrap(); // can't fail: slice is exactly 4 bytes
    for ey in 0..extrude {
        for ex in 0..extrude {
            let dst_x = ox + tw + ex;
            let dst_y = tile_y + ey;
            let dst_idx = ((dst_y as usize) * (aw as usize) + (dst_x as usize)) * 4;
            atlas.data[dst_idx..dst_idx + 4].copy_from_slice(&tr);
        }
    }

    // Bottom-left corner: tile[th-1, 0]
    let bl_idx = ((th - 1) as usize) * (tw as usize) * 4;
    let bl: [u8; 4] = tile.data[bl_idx..bl_idx + 4].try_into().unwrap(); // can't fail: slice is exactly 4 bytes
    for ey in 0..extrude {
        for ex in 0..extrude {
            let dst_x = tile_x + ex;
            let dst_y = oy + th + ey;
            let dst_idx = ((dst_y as usize) * (aw as usize) + (dst_x as usize)) * 4;
            atlas.data[dst_idx..dst_idx + 4].copy_from_slice(&bl);
        }
    }

    // Bottom-right corner: tile[th-1, tw-1]
    let br_idx = ((th - 1) as usize) * (tw as usize) * 4 + ((tw - 1) as usize) * 4;
    let br: [u8; 4] = tile.data[br_idx..br_idx + 4].try_into().unwrap(); // can't fail: slice is exactly 4 bytes
    for ey in 0..extrude {
        for ex in 0..extrude {
            let dst_x = ox + tw + ex;
            let dst_y = oy + th + ey;
            let dst_idx = ((dst_y as usize) * (aw as usize) + (dst_x as usize)) * 4;
            atlas.data[dst_idx..dst_idx + 4].copy_from_slice(&br);
        }
    }
}

/// Nyquist-safe iterative-halving resize.
///
/// For large downscales (2–8×), iteratively halves until close to target,
/// then a single final resize to exact dimensions. Avoids aliasing that a
/// single large-ratio resize would introduce.
pub fn resize_nyquist(src: &ImageRgba8, new_width: u32, interp: Interpolation) -> ImageRgba8 {
    if src.width == 0 || src.height == 0 || new_width == 0 {
        return ImageRgba8::zeros(0, 0);
    }

    if src.width == new_width {
        return src.clone();
    }

    if src.width < new_width {
        // Upscale: single-shot resize — dispatch on interp
        let new_height =
            (u64::from(src.height) * u64::from(new_width) / u64::from(src.width)) as u32;
        return resize_dispatch(src, new_width, new_height, interp);
    }

    // Downscale: iterative halving to respect Nyquist.
    // Use Option to avoid cloning `src` — first iteration reads from the reference directly.
    let mut cur_w = src.width;
    let mut cur_h = src.height;
    let mut owned: Option<ImageRgba8> = None;

    while cur_w > new_width {
        let half_w = cur_w.div_ceil(2);
        let half_h = cur_h.div_ceil(2);
        let source: &ImageRgba8 = owned.as_ref().unwrap_or(src);

        if half_w <= new_width {
            // Final resize to exact target — dispatch on interp
            let final_h =
                (f64::from(cur_h) * (f64::from(new_width) / f64::from(cur_w))).ceil() as u32;
            owned = Some(resize_dispatch(source, new_width, final_h, interp));
            cur_w = new_width;
            cur_h = final_h;
        } else {
            // Intermediate halving step (always bilinear per Nyquist policy)
            owned = Some(resize_bilinear(source, half_w, half_h));
            cur_w = half_w;
            cur_h = half_h;
        }
    }

    owned.unwrap_or_else(|| src.clone())
}

/// Dispatch resize to the appropriate algorithm.
fn resize_dispatch(src: &ImageRgba8, dst_w: u32, dst_h: u32, interp: Interpolation) -> ImageRgba8 {
    match interp {
        Interpolation::Nearest => resize_nearest(src, dst_w, dst_h),
        Interpolation::Linear => resize_bilinear(src, dst_w, dst_h),
        Interpolation::Cubic => resize_bicubic(src, dst_w, dst_h),
        Interpolation::Lanczos => resize_lanczos(src, dst_w, dst_h),
    }
}

/// Bilinear resize of RGBA image.
fn resize_bilinear(src: &ImageRgba8, dst_w: u32, dst_h: u32) -> ImageRgba8 {
    if dst_w == 0 || dst_h == 0 {
        return ImageRgba8::zeros(0, 0);
    }

    let sw = src.width as f32;
    let sh = src.height as f32;
    let dw = dst_w as f32;
    let dh = dst_h as f32;

    let mut out = ImageRgba8 {
        width: dst_w,
        height: dst_h,
        data: vec![0u8; (dst_w as usize) * (dst_h as usize) * 4],
    };

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            // Map destination pixel center to source coordinates
            let sx = (dx as f32 + 0.5) * sw / dw - 0.5;
            let sy = (dy as f32 + 0.5) * sh / dh - 0.5;

            let x0 = sx.floor() as i32;
            let y0 = sy.floor() as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;

            let fetch = |ix: i32, iy: i32| -> [u8; 4] {
                let sw_i32 = i32::try_from(src.width).unwrap_or(i32::MAX);
                let sh_i32 = i32::try_from(src.height).unwrap_or(i32::MAX);
                let cx = ix.clamp(0, sw_i32 - 1) as usize;
                let cy = iy.clamp(0, sh_i32 - 1) as usize;
                let idx = (cy * src.width as usize + cx) * 4;
                [
                    src.data[idx],
                    src.data[idx + 1],
                    src.data[idx + 2],
                    src.data[idx + 3],
                ]
            };

            let f00 = fetch(x0, y0);
            let f10 = fetch(x1, y0);
            let f01 = fetch(x0, y1);
            let f11 = fetch(x1, y1);

            let w00 = (1.0 - fx) * (1.0 - fy);
            let w10 = fx * (1.0 - fy);
            let w01 = (1.0 - fx) * fy;
            let w11 = fx * fy;

            let dst_idx = ((dy as usize) * (dst_w as usize) + (dx as usize)) * 4;
            for c in 0..4 {
                let val = f32::from(f00[c]).mul_add(
                    w00,
                    f32::from(f10[c])
                        .mul_add(w10, f32::from(f01[c]).mul_add(w01, f32::from(f11[c]) * w11)),
                );
                out.data[dst_idx + c] = val.round() as u8;
            }
        }
    }

    out
}

/// Nearest-neighbor resize of RGBA image.
fn resize_nearest(src: &ImageRgba8, dst_w: u32, dst_h: u32) -> ImageRgba8 {
    if dst_w == 0 || dst_h == 0 {
        return ImageRgba8::zeros(0, 0);
    }

    let sw = f64::from(src.width);
    let sh = f64::from(src.height);
    let dw = f64::from(dst_w);
    let dh = f64::from(dst_h);
    let sw_i = src.width as usize;
    let sh_i = src.height as usize;

    let mut out = ImageRgba8 {
        width: dst_w,
        height: dst_h,
        data: vec![0u8; (dst_w as usize) * (dst_h as usize) * 4],
    };

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = (f64::from(dx) + 0.5).mul_add(sw / dw, -0.5).round();
            let sy = (f64::from(dy) + 0.5).mul_add(sh / dh, -0.5).round();

            let cx = (sx as usize).min(sw_i - 1);
            let cy = (sy as usize).min(sh_i - 1);
            let src_idx = (cy * sw_i + cx) * 4;
            let dst_idx = (dy as usize * dst_w as usize + dx as usize) * 4;
            out.data[dst_idx..dst_idx + 4].copy_from_slice(&src.data[src_idx..src_idx + 4]);
        }
    }

    out
}

/// Bicubic (Catmull-Rom, a=-0.5) resize of RGBA image.
fn resize_bicubic(src: &ImageRgba8, dst_w: u32, dst_h: u32) -> ImageRgba8 {
    if dst_w == 0 || dst_h == 0 {
        return ImageRgba8::zeros(0, 0);
    }

    let sw = f64::from(src.width);
    let sh = f64::from(src.height);
    let dw = f64::from(dst_w);
    let dh = f64::from(dst_h);
    let src_w = src.width.cast_signed();
    let src_h = src.height.cast_signed();

    let mut out = ImageRgba8 {
        width: dst_w,
        height: dst_h,
        data: vec![0u8; (dst_w as usize) * (dst_h as usize) * 4],
    };

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = (f64::from(dx) + 0.5).mul_add(sw / dw, -0.5);
            let sy = (f64::from(dy) + 0.5).mul_add(sh / dh, -0.5);

            let ix = sx.floor() as i32;
            let iy = sy.floor() as i32;
            let fx = sx - f64::from(ix);
            let fy = sy - f64::from(iy);

            let mut accum = [0.0f64; 4];
            let mut weight_sum = 0.0f64;

            for ky in -1..=2i32 {
                let wy = cubic_weight(fy - f64::from(ky));
                let py = (iy + ky).clamp(0, src_h - 1) as usize;
                for kx in -1..=2i32 {
                    let wx = cubic_weight(fx - f64::from(kx));
                    let px = (ix + kx).clamp(0, src_w - 1) as usize;
                    let w = wx * wy;
                    weight_sum += w;
                    let idx = (py * src.width as usize + px) * 4;
                    for (c, acc) in accum.iter_mut().enumerate() {
                        *acc += f64::from(src.data[idx + c]) * w;
                    }
                }
            }

            let dst_idx = (dy as usize * dst_w as usize + dx as usize) * 4;
            if weight_sum.abs() > 1e-12 {
                for (c, acc) in accum.iter().enumerate() {
                    out.data[dst_idx + c] = (*acc / weight_sum).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    out
}

/// Catmull-Rom cubic weight function (a = -0.5).
#[inline]
fn cubic_weight(t: f64) -> f64 {
    const A: f64 = -0.5;
    let t_abs = t.abs();
    if t_abs <= 1.0 {
        ((A + 2.0).mul_add(t_abs, -(A + 3.0)) * t_abs).mul_add(t_abs, 1.0)
    } else if t_abs <= 2.0 {
        A.mul_add(t_abs, -5.0 * A)
            .mul_add(t_abs, 8.0 * A)
            .mul_add(t_abs, -4.0 * A)
    } else {
        0.0
    }
}

/// Lanczos3 resize of RGBA image (6×6 kernel, sinc windowed with 3 lobes).
fn resize_lanczos(src: &ImageRgba8, dst_w: u32, dst_h: u32) -> ImageRgba8 {
    if dst_w == 0 || dst_h == 0 {
        return ImageRgba8::zeros(0, 0);
    }

    let sw = f64::from(src.width);
    let sh = f64::from(src.height);
    let dw = f64::from(dst_w);
    let dh = f64::from(dst_h);
    let src_w = src.width.cast_signed();
    let src_h = src.height.cast_signed();

    let mut out = ImageRgba8 {
        width: dst_w,
        height: dst_h,
        data: vec![0u8; (dst_w as usize) * (dst_h as usize) * 4],
    };

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = (f64::from(dx) + 0.5).mul_add(sw / dw, -0.5);
            let sy = (f64::from(dy) + 0.5).mul_add(sh / dh, -0.5);

            let ix = sx.floor() as i32;
            let iy = sy.floor() as i32;
            let fx = sx - f64::from(ix);
            let fy = sy - f64::from(iy);

            let mut accum = [0.0f64; 4];
            let mut weight_sum = 0.0f64;

            for ky in -2..=3i32 {
                let wy = lanczos3_weight(fy - f64::from(ky));
                let py = (iy + ky).clamp(0, src_h - 1) as usize;
                for kx in -2..=3i32 {
                    let wx = lanczos3_weight(fx - f64::from(kx));
                    let px = (ix + kx).clamp(0, src_w - 1) as usize;
                    let w = wx * wy;
                    weight_sum += w;
                    let idx = (py * src.width as usize + px) * 4;
                    for (c, acc) in accum.iter_mut().enumerate() {
                        *acc += f64::from(src.data[idx + c]) * w;
                    }
                }
            }

            let dst_idx = (dy as usize * dst_w as usize + dx as usize) * 4;
            if weight_sum.abs() > 1e-12 {
                for (c, acc) in accum.iter().enumerate() {
                    out.data[dst_idx + c] = (*acc / weight_sum).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    out
}

/// Lanczos3 weight: sinc(t) * sinc(t/3) for |t| < 3, else 0.
#[inline]
fn lanczos3_weight(t: f64) -> f64 {
    let t_abs = t.abs();
    if t_abs < 1e-12 {
        1.0
    } else if t_abs < 3.0 {
        let pi_t = std::f64::consts::PI * t;
        let pi_t_over_3 = pi_t / 3.0;
        (pi_t.sin() / pi_t) * (pi_t_over_3.sin() / pi_t_over_3)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blit_extrude_no_extrude() {
        let mut atlas = ImageRgba8::zeros(4, 4);
        let mut tile = ImageRgba8::zeros(2, 2);
        // Fill tile with known values
        for i in 0..16 {
            tile.data[i] = (i as u8) + 1;
        }
        blit_extrude(&mut atlas, &tile, 1, 1, 0);
        // Check pixel at atlas (1,1) = tile (0,0) = [1,2,3,4]
        let idx = (4 + 1) * 4; // row=1 * stride=4 + col=1
        assert_eq!(&atlas.data[idx..idx + 4], &[1, 2, 3, 4]);
    }

    #[test]
    fn blit_extrude_with_padding() {
        let mut atlas = ImageRgba8::zeros(6, 6);
        let mut tile = ImageRgba8::zeros(2, 2);
        // tile pixels: row0=[R,G], row1=[B,W]
        tile.data[0..4].copy_from_slice(&[255, 0, 0, 255]); // (0,0) Red
        tile.data[4..8].copy_from_slice(&[0, 255, 0, 255]); // (1,0) Green
        tile.data[8..12].copy_from_slice(&[0, 0, 255, 255]); // (0,1) Blue
        tile.data[12..16].copy_from_slice(&[255, 255, 255, 255]); // (1,1) White

        blit_extrude(&mut atlas, &tile, 0, 0, 2);

        // Tile body at (2,2) and (3,2)
        let idx_body = (2 * 6 + 2) * 4;
        assert_eq!(&atlas.data[idx_body..idx_body + 4], &[255, 0, 0, 255]);

        // Top-left corner (0,0) should be tile[0,0] = Red
        assert_eq!(&atlas.data[0..4], &[255, 0, 0, 255]);
    }

    #[test]
    fn resize_nyquist_identity() {
        let mut src = ImageRgba8::zeros(4, 4);
        for (i, b) in src.data.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
        let result = resize_nyquist(&src, 4, Interpolation::Linear);
        assert_eq!(result.width, 4);
        assert_eq!(result.height, 4);
        assert_eq!(result.data, src.data);
    }

    #[test]
    fn resize_nyquist_halve() {
        // 4x4 → 2 wide
        let src = ImageRgba8 {
            width: 4,
            height: 4,
            data: vec![128u8; 4 * 4 * 4],
        };
        let result = resize_nyquist(&src, 2, Interpolation::Linear);
        assert_eq!(result.width, 2);
        // All pixels should remain 128
        assert!(result.data.iter().all(|&b| b == 128));
    }

    #[test]
    fn resize_cubic_differs_from_bilinear() {
        // 6×6 image with a single bright spot — non-integer scale factor (6→4)
        // produces subpixel samples where cubic's wider kernel gives different results
        let mut src = ImageRgba8::zeros(6, 6);
        // Place a single bright pixel at (3,3) surrounded by dark
        let idx = (3 * 6 + 3) * 4;
        src.data[idx] = 255;
        src.data[idx + 1] = 255;
        src.data[idx + 2] = 255;
        src.data[idx + 3] = 255;
        // Fill rest with alpha=255 so alpha channel doesn't mask differences
        for y in 0..6u32 {
            for x in 0..6u32 {
                let i = (y as usize * 6 + x as usize) * 4 + 3;
                src.data[i] = 255;
            }
        }
        let bilinear = resize_nyquist(&src, 4, Interpolation::Linear);
        let cubic = resize_nyquist(&src, 4, Interpolation::Cubic);
        assert_eq!(bilinear.width, 4);
        assert_eq!(cubic.width, 4);
        assert_eq!(bilinear.height, cubic.height);
        // A point source sampled at non-integer positions produces different spread
        assert_ne!(bilinear.data, cubic.data);
    }

    #[test]
    fn resize_nearest_preserves_hard_pixels() {
        // 2×2 with distinct pixels, resize to 4×4 — nearest should produce exact blocks
        let mut src = ImageRgba8::zeros(2, 2);
        src.data[0..4].copy_from_slice(&[10, 20, 30, 255]);
        src.data[4..8].copy_from_slice(&[40, 50, 60, 255]);
        src.data[8..12].copy_from_slice(&[70, 80, 90, 255]);
        src.data[12..16].copy_from_slice(&[100, 110, 120, 255]);
        let result = resize_nyquist(&src, 4, Interpolation::Nearest);
        assert_eq!(result.width, 4);
        assert_eq!(result.height, 4);
        // Top-left 2×2 block should all be [10, 20, 30, 255]
        let p00 = &result.data[0..4];
        let p10 = &result.data[4..8];
        assert_eq!(p00, &[10, 20, 30, 255]);
        assert_eq!(p10, &[10, 20, 30, 255]);
    }

    #[test]
    fn resize_lanczos_differs_from_bilinear() {
        // Same point-source pattern — Lanczos3 with negative lobes differs from bilinear
        let mut src = ImageRgba8::zeros(6, 6);
        let idx = (3 * 6 + 3) * 4;
        src.data[idx] = 255;
        src.data[idx + 1] = 255;
        src.data[idx + 2] = 255;
        src.data[idx + 3] = 255;
        for y in 0..6u32 {
            for x in 0..6u32 {
                let i = (y as usize * 6 + x as usize) * 4 + 3;
                src.data[i] = 255;
            }
        }
        let bilinear = resize_nyquist(&src, 4, Interpolation::Linear);
        let lanczos = resize_nyquist(&src, 4, Interpolation::Lanczos);
        assert_eq!(bilinear.width, lanczos.width);
        assert_eq!(bilinear.height, lanczos.height);
        assert_ne!(bilinear.data, lanczos.data);
    }

    #[test]
    fn resize_uniform_unchanged_all_methods() {
        // Uniform image should produce same result regardless of interpolation method
        let src = ImageRgba8 {
            width: 8,
            height: 8,
            data: vec![100u8; 8 * 8 * 4],
        };
        for interp in &[
            Interpolation::Nearest,
            Interpolation::Linear,
            Interpolation::Cubic,
            Interpolation::Lanczos,
        ] {
            let result = resize_nyquist(&src, 4, *interp);
            assert_eq!(result.width, 4);
            assert!(
                result.data.iter().all(|&b| b == 100),
                "uniform image changed under resize"
            );
        }
    }
}
