//! Atlas grid layout selection: pick (cols, rows) for an output frame count
//! that best preserves the input aspect ratio while fitting the GPU texture
//! limit.

/// Result of `pick_layout`: a chosen atlas grid plus the number of padding
/// tiles added beyond the requested frame count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtlasLayout {
    pub cols: u32,
    pub rows: u32,
    /// Number of empty tiles appended past `n` to obtain a fitting grid.
    /// `cols * rows == n + padding`.
    pub padding: u32,
    /// Computed tile pixel width.
    pub tile_width: u32,
    /// Computed tile pixel height.
    pub tile_height: u32,
}

/// The minimum per-tile pixel dimension (both width and height).
pub const MIN_TILE_DIM: u32 = 4;

/// Default cap on the number of padding tiles `pick_layout` will try.
pub const DEFAULT_PADDING_BOUND: u32 = 8;

/// Compute tile dimensions for a given atlas grid and input aspect ratio.
///
/// Returns `(tile_width, tile_height)` clamped so the total atlas fits within
/// `atlas_resolution` on both axes.
#[must_use]
pub fn compute_tile_dims(
    atlas_resolution: u32,
    cols: u32,
    rows: u32,
    input_aspect_ratio: f64,
) -> (u32, u32) {
    let (cols, rows) = (cols.max(1), rows.max(1));
    // Start with width filling the atlas resolution horizontally.
    let mut tile_w = (f64::from(atlas_resolution) / f64::from(cols)).floor() as u32;
    let mut tile_h = (f64::from(tile_w) / input_aspect_ratio).round() as u32;

    // If rows * tile_h exceeds atlas_resolution, clamp height and recompute width.
    if rows.saturating_mul(tile_h) > atlas_resolution {
        tile_h = (f64::from(atlas_resolution) / f64::from(rows)).floor() as u32;
        tile_w = (f64::from(tile_h) * input_aspect_ratio).round() as u32;
    }

    // Ensure minimum tile dimension and within atlas_resolution.
    tile_w = tile_w.clamp(MIN_TILE_DIM, atlas_resolution);
    tile_h = tile_h.clamp(MIN_TILE_DIM, atlas_resolution);

    (tile_w, tile_h)
}

/// Pick the best (cols, rows) layout for `frame_count` frames that preserves
/// input aspect ratio while fitting the GPU texture limit.
///
/// Iterates (cols, rows) pairs where `cols * rows >= frame_count` and scores
/// each by (1) aspect ratio match and (2) wasted tiles. Returns `None` if no
/// layout fits.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn pick_layout(
    frame_count: u32,
    input_aspect_ratio: f64,
    atlas_resolution: u32,
    max_dim: u32,
    padding_bound: u32,
) -> Option<AtlasLayout> {
    if frame_count == 0 || atlas_resolution == 0 || max_dim == 0 || input_aspect_ratio <= 0.0 {
        return None;
    }

    let max_dim = max_dim.min(atlas_resolution);
    if max_dim < MIN_TILE_DIM {
        return None;
    }

    let mut best: Option<AtlasLayout> = None;
    let mut best_aspect_diff = f64::MAX;
    let mut best_waste = u32::MAX;
    let mut best_cols = 0u32;
    let mut best_rows = 0u32;

    // Try up to padding_bound extra tiles beyond frame_count.
    for k in 0..=padding_bound {
        let m = frame_count + k;
        // Generate candidate (cols, rows) pairs where cols*rows >= m.
        for cols in 1..=max_dim.min(m) {
            let rows = m.div_ceil(cols);
            if cols.saturating_mul(rows) > max_dim.saturating_mul(max_dim) {
                continue;
            }

            let (tile_w, tile_h) =
                compute_tile_dims(atlas_resolution, cols, rows, input_aspect_ratio);

            if tile_w < MIN_TILE_DIM || tile_h < MIN_TILE_DIM {
                continue;
            }

            // Check total atlas fits within max_dim.
            if cols.saturating_mul(tile_w) > max_dim || rows.saturating_mul(tile_h) > max_dim {
                continue;
            }

            let tile_aspect = f64::from(tile_w) / f64::from(tile_h);
            let aspect_diff = (tile_aspect - input_aspect_ratio).abs();
            let wasted = cols.saturating_mul(rows) - frame_count;

            let replace = match &best {
                None => true,
                Some(_) => {
                    // Primary sort: aspect ratio match.
                    // Secondary: wasted tiles.
                    // Tertiary: squarest grid (minimize cols-rows diff).
                    if aspect_diff < best_aspect_diff - 1e-9 {
                        true
                    } else if (aspect_diff - best_aspect_diff).abs() > 1e-9 {
                        false
                    } else if wasted < best_waste {
                        true
                    } else if wasted > best_waste {
                        false
                    } else {
                        // Equal aspect and waste: prefer squarest grid.
                        let diff = cols.abs_diff(rows);
                        let best_diff = best_cols.abs_diff(best_rows);
                        diff < best_diff
                    }
                }
            };

            if replace {
                best_aspect_diff = aspect_diff;
                best_waste = wasted;
                best_cols = cols;
                best_rows = rows;
                best = Some(AtlasLayout {
                    cols,
                    rows,
                    padding: k,
                    tile_width: tile_w,
                    tile_height: tile_h,
                });
            }
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn wide_aspect() -> f64 {
        16.0 / 9.0
    }
    #[allow(dead_code)]
    fn tall_aspect() -> f64 {
        9.0 / 16.0
    }

    #[test]
    fn pick_layout_zero_inputs_return_none() {
        assert!(pick_layout(0, 1.0, 2048, 8192, 8).is_none());
        assert!(pick_layout(10, 0.0, 2048, 8192, 8).is_none());
        assert!(pick_layout(10, 1.0, 0, 8192, 8).is_none());
    }

    #[test]
    fn pick_layout_single_frame() {
        let l = pick_layout(1, 1.0, 2048, 8192, 8).unwrap();
        assert_eq!(l.cols, 1);
        assert_eq!(l.rows, 1);
        assert_eq!(l.padding, 0);
        assert!(l.tile_width >= MIN_TILE_DIM);
        assert!(l.tile_height >= MIN_TILE_DIM);
    }

    #[test]
    fn pick_layout_square_aspect_uses_squareish_grid() {
        let l = pick_layout(64, 1.0, 2048, 8192, 8).unwrap();
        assert_eq!(l.cols * l.rows, 64);
        assert_eq!(l.padding, 0);
        // Square tiles for square aspect.
        let diff = (f64::from(l.tile_width) / f64::from(l.tile_height) - 1.0).abs();
        assert!(diff < 0.1, "tile aspect diff too large: {diff}");
    }

    #[test]
    fn pick_layout_wide_aspect_prefers_wide_grid() {
        let aspect = wide_aspect();
        let l = pick_layout(64, aspect, 2048, 8192, 8).unwrap();
        // Tiles should be wider than tall to match input aspect
        let tile_aspect = l.tile_width as f64 / l.tile_height as f64;
        let diff = (tile_aspect - aspect).abs();
        assert!(
            diff < 0.3,
            "tile aspect {tile_aspect} too far from input {aspect}: diff {diff}"
        );
    }

    #[test]
    fn pick_layout_tall_aspect_prefers_tall_grid() {
        let aspect = tall_aspect();
        let l = pick_layout(64, aspect, 2048, 8192, 8).unwrap();
        let tile_aspect = f64::from(l.tile_width) / f64::from(l.tile_height);
        let diff = (tile_aspect - aspect).abs();
        assert!(
            diff < 0.3,
            "tile aspect {tile_aspect} too far from input {aspect}: diff {diff}"
        );
    }

    #[test]
    fn pick_layout_waste_minimized_when_aspect_equal() {
        // 50 frames: 10x5 (waste 0) vs 7x8 (waste 6) — should pick 10x5.
        let l = pick_layout(50, 1.0, 2048, 8192, 8).unwrap();
        assert_eq!(l.cols * l.rows, 50, "should pick exact-fit layout");
        assert_eq!(l.padding, 0);
    }

    #[test]
    fn pick_layout_fits_within_max_dim() {
        let l = pick_layout(100, 1.0, 2048, 4096, 8).unwrap();
        assert!(l.cols * l.tile_width <= 4096);
        assert!(l.rows * l.tile_height <= 4096);
    }

    #[test]
    fn pick_layout_padding_bound_respected() {
        // With k_max=0 and an odd prime count that can't be factored, expect
        // cols*rows == frame_count but may still work with 1xN.
        let l = pick_layout(7, 1.0, 2048, 8192, 0);
        assert!(l.is_some(), "7 frames should fit in 7x1 or 1x7");
        if let Some(layout) = l {
            assert!(layout.cols * layout.rows >= 7);
            assert_eq!(layout.padding, 0);
        }
    }

    #[test]
    fn compute_tile_dims_square_grid_square_aspect() {
        let (tw, th) = compute_tile_dims(2048, 4, 4, 1.0);
        assert_eq!(tw, 512);
        assert_eq!(th, 512);
    }

    #[test]
    fn compute_tile_dims_wide_aspect() {
        let (tw, th) = compute_tile_dims(2048, 4, 4, 16.0 / 9.0);
        // tw = 2048/4 = 512, th = 512 / (16/9) = 288
        assert_eq!(tw, 512);
        assert_eq!(th, 288);
    }

    #[test]
    fn compute_tile_dims_tall_aspect() {
        let (tw, th) = compute_tile_dims(2048, 4, 4, 9.0 / 16.0);
        // tw = 2048/4 = 512, th = 512 / (9/16) = 910.2 -> 910
        // But 4 * 910 = 3640 > 2048, so clamp: th = 2048/4 = 512, tw = 512 * 9/16 = 288
        assert_eq!(tw, 288);
        assert_eq!(th, 512);
    }

    #[test]
    fn compute_tile_dims_caps_height_when_rows_overflow() {
        // 2048 res, 2 cols, 50 rows = tight vertically
        let (tw, th) = compute_tile_dims(2048, 2, 50, 1.0);
        // tw = 2048/2 = 1024, th = 1024
        // But 50 * 1024 = 51200 > 2048, so: th = 2048/50 = 40, tw = 40*1 = 40
        assert_eq!(th, 40);
        assert_eq!(tw, 40);
    }
}
