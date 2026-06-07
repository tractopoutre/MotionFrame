//! Atlas input loading helpers.
//!
//! These helpers normalize sequence and atlas inputs into the shared source
//! representation used by the engine pipeline.

use crate::pipeline::ImageRgba8;

const MIN_TILE_SIZE: u32 = 4;
const MAX_DETECTED_TILES_PER_AXIS: u32 = 64;
/// Below this absolute mean-discontinuity score, the image is too smooth
/// to detect any structure — return `None`. Sized to reject smooth
/// gradients (per-pixel diff ~4 per channel, sum ~12 across RGB) while
/// admitting any image with real tile boundaries (typically ≥ 50).
const MIN_ABSOLUTE_SCORE: f64 = 16.0;

/// Errors produced when slicing an atlas image into tiles.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AtlasError {
    #[error("atlas needs at least 2 tiles (cols and rows must each be >= 1); got {cols}x{rows}")]
    TooFewTiles { cols: u32, rows: u32 },
    #[error("atlas tile size rounded to zero for image {w}x{h} and grid {cols}x{rows}")]
    TileSizeZero {
        w: u32,
        h: u32,
        cols: u32,
        rows: u32,
    },
}

/// Slice an atlas image into row-major sub-tiles (origin top-left, left to
/// right, top to bottom). Tile index `i` lives at row = `i / cols`, col =
/// `i % cols`.
///
/// Non-divisible image dimensions are valid: tiles are sized via
/// ceiling-division so the UV partition is continuous (no dropped pixels).
pub fn slice_atlas(src: &ImageRgba8, cols: u32, rows: u32) -> Result<Vec<ImageRgba8>, AtlasError> {
    let Some(tile_count) = cols.checked_mul(rows) else {
        return Err(AtlasError::TooFewTiles { cols, rows });
    };
    if cols == 0 || rows == 0 || tile_count < 2 {
        return Err(AtlasError::TooFewTiles { cols, rows });
    }

    let tile_w = div_round(src.width, cols);
    let tile_h = div_round(src.height, rows);
    if tile_w == 0 || tile_h == 0 {
        return Err(AtlasError::TileSizeZero {
            w: src.width,
            h: src.height,
            cols,
            rows,
        });
    }

    let mut tiles = Vec::with_capacity(tile_count as usize);
    for row in 0..rows {
        for col in 0..cols {
            let x0 = f64::from(col) * f64::from(src.width) / f64::from(cols);
            let x1 = f64::from(col + 1) * f64::from(src.width) / f64::from(cols);
            let y0 = f64::from(row) * f64::from(src.height) / f64::from(rows);
            let y1 = f64::from(row + 1) * f64::from(src.height) / f64::from(rows);
            let scale_x = (x1 - x0) / f64::from(tile_w);
            let scale_y = (y1 - y0) / f64::from(tile_h);
            let mut data = Vec::with_capacity((tile_w * tile_h * 4) as usize);

            for ty in 0..tile_h {
                for tx in 0..tile_w {
                    let sx = x0 + (f64::from(tx) + 0.5) * scale_x - 0.5;
                    let sy = y0 + (f64::from(ty) + 0.5) * scale_y - 0.5;
                    data.extend_from_slice(&sample_bilinear_rgba(src, sx, sy));
                }
            }

            tiles.push(ImageRgba8 {
                width: tile_w,
                height: tile_h,
                data,
            });
        }
    }
    Ok(tiles)
}

fn div_round(n: u32, d: u32) -> u32 {
    n / d + u32::from(n % d >= d.div_ceil(2))
}

fn sample_bilinear_rgba(src: &ImageRgba8, x: f64, y: f64) -> [u8; 4] {
    let max_x = f64::from(src.width.saturating_sub(1));
    let max_y = f64::from(src.height.saturating_sub(1));
    let x = x.clamp(0.0, max_x);
    let y = y.clamp(0.0, max_y);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(src.width.saturating_sub(1));
    let y1 = (y0 + 1).min(src.height.saturating_sub(1));
    let fx = x - f64::from(x0);
    let fy = y - f64::from(y0);

    let p00 = pixel_rgba(src, x0, y0);
    let p10 = pixel_rgba(src, x1, y0);
    let p01 = pixel_rgba(src, x0, y1);
    let p11 = pixel_rgba(src, x1, y1);

    let mut out = [0u8; 4];
    for i in 0..4 {
        let top = f64::from(p00[i]) * (1.0 - fx) + f64::from(p10[i]) * fx;
        let bottom = f64::from(p01[i]) * (1.0 - fx) + f64::from(p11[i]) * fx;
        out[i] = (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
    }
    out
}

fn pixel_rgba(src: &ImageRgba8, x: u32, y: u32) -> [u8; 4] {
    let i = ((y as usize) * (src.width as usize) + (x as usize)) * 4;
    [
        src.data[i],
        src.data[i + 1],
        src.data[i + 2],
        src.data[i + 3],
    ]
}

fn counts_capped(max: u32) -> Vec<u32> {
    (1..=max).collect()
}

/// Heuristic best guess at tile count. Returns `None` when no candidate
/// clearly dominates (typically: a smooth or near-uniform image).
///
/// Candidate grids do not need to divide the source dimensions evenly
/// (`1 × 1` is not a valid atlas). Scores candidates by mean RGB seam
/// contrast, rejects images without enough raw contrast, then chooses
/// the highest-scoring candidate weighted by tile count.
pub fn detect_tile_count(src: &ImageRgba8) -> Option<(u32, u32)> {
    let max_cols = (src.width / MIN_TILE_SIZE).min(MAX_DETECTED_TILES_PER_AXIS);
    let max_rows = (src.height / MIN_TILE_SIZE).min(MAX_DETECTED_TILES_PER_AXIS);
    let cols_candidates = counts_capped(max_cols);
    let rows_candidates = counts_capped(max_rows);

    let candidates: Vec<(u32, u32)> = cols_candidates
        .iter()
        .flat_map(|&c| rows_candidates.iter().map(move |&r| (c, r)))
        .filter(|&(c, r)| c.saturating_mul(r) >= 2)
        .collect();
    if candidates.is_empty() {
        return None;
    }

    let scored: Vec<((u32, u32), f64, f64)> = candidates
        .iter()
        .map(|&dims| {
            let raw = score_seams(src, dims);
            let tile_count = dims.0 * dims.1;
            let weighted = raw * f64::from(tile_count + 1).log2();
            (dims, raw, weighted)
        })
        .collect();

    let raw_max = scored
        .iter()
        .map(|&(_, raw, _)| raw)
        .fold(f64::NEG_INFINITY, f64::max);
    if raw_max < MIN_ABSOLUTE_SCORE {
        return None;
    }

    scored
        .into_iter()
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .map(|(dims, _, _)| dims)
}

fn score_seams(src: &ImageRgba8, (cols, rows): (u32, u32)) -> f64 {
    let mut total = 0.0;
    let mut count: u64 = 0;

    // Vertical seams.
    for c in 1..cols {
        let seam = f64::from(c) * f64::from(src.width) / f64::from(cols);
        let left = (seam.floor() - 1.0).max(0.0) as u32;
        let right = seam.ceil().min(f64::from(src.width.saturating_sub(1))) as u32;
        for y in 0..src.height {
            total += seam_contrast_rgb(src, left, y, right, y, true);
            count += 1;
        }
    }
    // Horizontal seams.
    for r in 1..rows {
        let seam = f64::from(r) * f64::from(src.height) / f64::from(rows);
        let top = (seam.floor() - 1.0).max(0.0) as u32;
        let bottom = seam.ceil().min(f64::from(src.height.saturating_sub(1))) as u32;
        for x in 0..src.width {
            total += seam_contrast_rgb(src, x, top, x, bottom, false);
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total / (count as f64)
    }
}

fn seam_contrast_rgb(src: &ImageRgba8, ax: u32, ay: u32, bx: u32, by: u32, vertical: bool) -> f64 {
    let seam = abs_diff_rgb(src, ax, ay, bx, by) as f64;
    let mut baseline = 0.0;
    let mut count = 0.0;

    if vertical {
        if ax > 0 {
            baseline += abs_diff_rgb(src, ax - 1, ay, ax, ay) as f64;
            count += 1.0;
        }
        if bx + 1 < src.width {
            baseline += abs_diff_rgb(src, bx, by, bx + 1, by) as f64;
            count += 1.0;
        }
    } else {
        if ay > 0 {
            baseline += abs_diff_rgb(src, ax, ay - 1, ax, ay) as f64;
            count += 1.0;
        }
        if by + 1 < src.height {
            baseline += abs_diff_rgb(src, bx, by, bx, by + 1) as f64;
            count += 1.0;
        }
    }

    let local = if count == 0.0 { 0.0 } else { baseline / count };
    (seam - local).max(0.0)
}

fn abs_diff_rgb(src: &ImageRgba8, ax: u32, ay: u32, bx: u32, by: u32) -> u64 {
    let stride = (src.width as usize) * 4;
    let ai = (ay as usize) * stride + (ax as usize) * 4;
    let bi = (by as usize) * stride + (bx as usize) * 4;
    let dr = i32::from(src.data[ai]) - i32::from(src.data[bi]);
    let dg = i32::from(src.data[ai + 1]) - i32::from(src.data[bi + 1]);
    let db = i32::from(src.data[ai + 2]) - i32::from(src.data[bi + 2]);
    u64::from(dr.unsigned_abs() + dg.unsigned_abs() + db.unsigned_abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an `ImageRgba8` of `cols × rows` tiles, each `tile_w × tile_h` px,
    /// where every pixel of tile `(tx, ty)` has color `(tx as u8, ty as u8, 0, 255)`.
    #[allow(clippy::many_single_char_names)]
    fn synth_atlas(cols: u32, rows: u32, tile_w: u32, tile_h: u32) -> ImageRgba8 {
        let w = cols * tile_w;
        let h = rows * tile_h;
        let mut data = vec![0u8; (w * h * 4) as usize];
        for ty in 0..rows {
            for tx in 0..cols {
                for dy in 0..tile_h {
                    for dx in 0..tile_w {
                        let x = tx * tile_w + dx;
                        let y = ty * tile_h + dy;
                        let i = ((y * w + x) * 4) as usize;
                        data[i] = tx as u8;
                        data[i + 1] = ty as u8;
                        data[i + 2] = 0;
                        data[i + 3] = 255;
                    }
                }
            }
        }
        ImageRgba8 {
            width: w,
            height: h,
            data,
        }
    }

    #[test]
    fn slice_atlas_row_major_order() {
        let src = synth_atlas(4, 2, 3, 5); // 12×10
        let tiles = slice_atlas(&src, 4, 2).unwrap();
        assert_eq!(tiles.len(), 8);
        for (i, tile) in tiles.iter().enumerate() {
            let tx = (i as u32) % 4;
            let ty = (i as u32) / 4;
            assert_eq!(tile.width, 3);
            assert_eq!(tile.height, 5);
            // first pixel of every tile encodes (tx, ty, 0, 255)
            assert_eq!(tile.data[0], tx as u8);
            assert_eq!(tile.data[1], ty as u8);
            assert_eq!(tile.data[2], 0);
            assert_eq!(tile.data[3], 255);
            // last pixel too — confirms full tile pixels carried over
            let last = tile.data.len() - 4;
            assert_eq!(tile.data[last], tx as u8);
            assert_eq!(tile.data[last + 1], ty as u8);
        }
    }

    #[test]
    fn slice_atlas_divisible_dims_preserve_pixels() {
        let src = synth_atlas(2, 2, 4, 4);
        let tiles = slice_atlas(&src, 2, 2).unwrap();

        assert_eq!(tiles[0].width, 4);
        assert_eq!(tiles[0].height, 4);
        assert_eq!(tiles[0].data[0], 0);
        assert_eq!(tiles[0].data[1], 0);
        assert_eq!(tiles[1].data[0], 1);
        assert_eq!(tiles[1].data[1], 0);
        assert_eq!(tiles[2].data[0], 0);
        assert_eq!(tiles[2].data[1], 1);
        assert_eq!(tiles[3].data[0], 1);
        assert_eq!(tiles[3].data[1], 1);
    }

    #[test]
    fn slice_atlas_accepts_non_divisible_dims() {
        // Non-divisible dimensions use ceiling-division for continuous UV partition.
        let src = synth_atlas(3, 2, 5, 4);
        let tiles = slice_atlas(&src, 2, 2).unwrap();

        assert_eq!(tiles.len(), 4);
        assert_eq!(tiles[0].width, 8);
        assert_eq!(tiles[0].height, 4);
        assert!(tiles.iter().all(|tile| tile.width == 8 && tile.height == 4));
    }

    #[test]
    fn slice_atlas_rejects_too_few_tiles() {
        let src = ImageRgba8::zeros(8, 8);
        let err = slice_atlas(&src, 1, 1).unwrap_err();
        assert_eq!(err, AtlasError::TooFewTiles { cols: 1, rows: 1 });
    }

    #[test]
    fn slice_atlas_rejects_one_by_one() {
        let src = synth_atlas(1, 2, 4, 4);
        let err = slice_atlas(&src, 1, 1).unwrap_err();
        assert_eq!(err, AtlasError::TooFewTiles { cols: 1, rows: 1 });
    }

    #[test]
    fn slice_atlas_minimal_two_tiles() {
        // 1 column × 2 rows is the smallest legal layout.
        let src = synth_atlas(1, 2, 4, 4);
        let tiles = slice_atlas(&src, 1, 2).unwrap();
        assert_eq!(tiles.len(), 2);
        assert_eq!(tiles[0].data[1], 0); // ty=0
        assert_eq!(tiles[1].data[1], 1); // ty=1
    }

    #[test]
    fn detect_tile_count_recovers_true_dims() {
        // 4×2 atlas of 16×16 tiles, each tile a unique solid color.
        // True grid is (4, 2). Sub-grids (2, 2) etc. score the same on real
        // boundaries; the largest-product tie-break must pick (4, 2).
        let src = colorful_atlas(4, 2, 16, 16);
        let dims = detect_tile_count(&src).expect("should detect");
        assert_eq!(dims, (4, 2));
    }

    #[test]
    fn detect_tile_count_recovers_npot_dims() {
        // 5×3 atlas of 16×16 tiles. True grid is (5, 3) — both NPOT.
        // Image is 80×48. Divisors of 80 capped at 20: [1,2,4,5,8,10,16,20].
        // Divisors of 48 capped at 12: [1,2,3,4,6,8,12].
        let src = colorful_atlas(5, 3, 16, 16);
        let dims = detect_tile_count(&src).expect("should detect NPOT");
        assert_eq!(dims, (5, 3));
    }

    #[test]
    fn detect_tile_count_recovers_non_divisor_dims() {
        // Source is cropped by one pixel on each axis after fixture generation.
        // True grid is still 5 x 3, but 79 and 47 are not divisible by 5 or 3.
        let full = colorful_atlas(5, 3, 16, 16);
        let src = crop_rgba(&full, 79, 47);
        let dims = detect_tile_count(&src).expect("should detect non-divisor grid");
        assert_eq!(dims, (5, 3));
    }

    #[test]
    fn detect_tile_count_never_returns_one_by_one() {
        let src = colorful_atlas(1, 2, 16, 16);
        assert_ne!(detect_tile_count(&src), Some((1, 1)));
    }

    #[test]
    fn detect_tile_count_smooth_image_returns_none() {
        // Pure 64×64 mid-grey: no structure at all.
        let mut data = vec![128u8; 64 * 64 * 4];
        for px in data.chunks_exact_mut(4) {
            px[3] = 255;
        }
        let src = ImageRgba8 {
            width: 64,
            height: 64,
            data,
        };
        assert_eq!(detect_tile_count(&src), None);
    }

    #[test]
    fn detect_tile_count_horizontal_gradient_returns_none() {
        // Smooth horizontal gradient — no tile boundaries, score below
        // MIN_ABSOLUTE_SCORE because per-pixel diffs are ≤ 1.
        let w = 64u32;
        let h = 64u32;
        let mut data = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                data[i] = (x * 255 / (w - 1)) as u8;
                data[i + 1] = data[i];
                data[i + 2] = data[i];
                data[i + 3] = 255;
            }
        }
        let src = ImageRgba8 {
            width: w,
            height: h,
            data,
        };
        assert_eq!(detect_tile_count(&src), None);
    }

    /// Like `synth_atlas` but with multiplicatively-spaced colors so
    /// adjacent tiles always have a large RGB delta.
    #[allow(clippy::many_single_char_names)]
    fn colorful_atlas(cols: u32, rows: u32, tile_w: u32, tile_h: u32) -> ImageRgba8 {
        let w = cols * tile_w;
        let h = rows * tile_h;
        let mut data = vec![0u8; (w * h * 4) as usize];
        for ty in 0..rows {
            for tx in 0..cols {
                let r = ((tx * 67) & 0xff) as u8;
                let g = ((ty * 113) & 0xff) as u8;
                let b = ((tx * ty * 41 + 31) & 0xff) as u8;
                for dy in 0..tile_h {
                    for dx in 0..tile_w {
                        let x = tx * tile_w + dx;
                        let y = ty * tile_h + dy;
                        let i = ((y * w + x) * 4) as usize;
                        data[i] = r;
                        data[i + 1] = g;
                        data[i + 2] = b;
                        data[i + 3] = 255;
                    }
                }
            }
        }
        ImageRgba8 {
            width: w,
            height: h,
            data,
        }
    }

    fn crop_rgba(src: &ImageRgba8, width: u32, height: u32) -> ImageRgba8 {
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            let row_start = ((y * src.width) * 4) as usize;
            let row_end = row_start + (width * 4) as usize;
            data.extend_from_slice(&src.data[row_start..row_end]);
        }
        ImageRgba8 {
            width,
            height,
            data,
        }
    }
}
