//! Arrow-based optical flow visualization.

use crate::pipeline::{Flow, GenerateOptions, ImageRgba8};

/// Build the full visualization atlas (all flow pairs composited into a grid).
///
/// Built lazily (on demand when the Visualization tab is first opened) to
/// avoid ~512 MB of unconditional allocation during generation.
///
/// Renders each tile at atlas-cell resolution. R/G channels are BT.601 luma
/// of the premultiplied RGB. Arrow displacement is scaled by tile dims so a
/// normalized flow of ±1 spans a full tile.
// allow(cast_possible_wrap): tile offsets and pixel positions, always < i32::MAX
#[allow(clippy::cast_possible_wrap)]
pub fn draw_optical_flow_atlas(
    flows: &[Flow],
    color_atlas: &ImageRgba8,
    total_color_tiles: u32,
    frame_width: u32,
    frame_height: u32,
    opts: &GenerateOptions,
) -> ImageRgba8 {
    const TILE_STEP: u32 = 16;

    let (atlas_cols, _atlas_rows) = opts.atlas_dims;
    let atlas_w = color_atlas.width;
    let atlas_h = color_atlas.height;
    let mut atlas = ImageRgba8::zeros(atlas_w, atlas_h);
    let premul = opts.premultiplied_alpha;

    for (flow_idx, flow) in flows.iter().enumerate() {
        // Tile flow_idx in the color atlas is frame1; tile flow_idx+1 is frame2.
        // The +1 lookup mirrors the previous gray-frame indexing: skip flows
        // whose pair frame would be past the last filled tile.
        let f1 = flow_idx as u32;
        let f2 = f1 + 1;
        if f2 >= total_color_tiles {
            break;
        }

        let ox1 = (f1 % atlas_cols) * frame_width;
        let oy1 = (f1 / atlas_cols) * frame_height;
        let ox2 = (f2 % atlas_cols) * frame_width;
        let oy2 = (f2 / atlas_cols) * frame_height;

        // R = frame1 luma, G = frame2 luma, B = 0, A = 255.
        for dy in 0..frame_height {
            for dx in 0..frame_width {
                let f1_pix = ((oy1 + dy) as usize * atlas_w as usize + (ox1 + dx) as usize) * 4;
                let f2_pix = ((oy2 + dy) as usize * atlas_w as usize + (ox2 + dx) as usize) * 4;
                let dst_idx = f1_pix;
                atlas.data[dst_idx] = sample_luma(&color_atlas.data, f1_pix, premul);
                atlas.data[dst_idx + 1] = sample_luma(&color_atlas.data, f2_pix, premul);
                atlas.data[dst_idx + 2] = 0;
                atlas.data[dst_idx + 3] = 255;
            }
        }

        // Arrows on a tile-space grid; flow sampled at the corresponding native pixel.
        let src_w = flow.width;
        let src_h = flow.height;
        let half_step = TILE_STEP / 2;
        let mut ty = half_step;
        while ty < frame_height {
            let mut tx = half_step;
            while tx < frame_width {
                let sx = tile_to_src(tx, frame_width, src_w);
                let sy = tile_to_src(ty, frame_height, src_h);
                let [fx, fy] = *flow.at(sx, sy);
                // Flow is normalized to [-1,1]; scale by tile dims for tile-space pixels.
                let arrow_dx = fx * frame_width as f32;
                let arrow_dy = fy * frame_height as f32;
                let x1 = (ox1 + tx) as i32;
                let y1 = (oy1 + ty) as i32;
                let x2 = x1 + arrow_dx as i32;
                let y2 = y1 + arrow_dy as i32;
                draw_arrowed_line(&mut atlas, x1, y1, x2, y2, [255, 0, 255, 255], 0.2);
                tx += TILE_STEP;
            }
            ty += TILE_STEP;
        }
    }

    atlas
}

/// BT.601 luma of an RGB triple — fixed-point ×256 with rounding.
fn bt601_luma(r: u8, g: u8, b: u8) -> u8 {
    let y = (u32::from(r) * 76 + u32::from(g) * 150 + u32::from(b) * 30 + 128) >> 8;
    y.min(255) as u8
}

/// Sample BT.601 luma of `rgb·α` at `pix` in `data`. When the atlas is stored
/// straight, RGB is multiplied by α on the fly; when stored premul, RGB
/// already encodes `rgb·α` (so transparent pixels are zero by construction).
fn sample_luma(data: &[u8], pix: usize, premultiplied: bool) -> u8 {
    let r = data[pix];
    let g = data[pix + 1];
    let b = data[pix + 2];
    if premultiplied {
        return bt601_luma(r, g, b);
    }
    let a = u32::from(data[pix + 3]);
    let premul = |c: u8| ((u32::from(c) * a + 127) / 255) as u8;
    bt601_luma(premul(r), premul(g), premul(b))
}

/// Map a tile-space coordinate to the nearest source-image coordinate.
fn tile_to_src(tile_coord: u32, tile_size: u32, src_size: u32) -> u32 {
    let s = ((tile_coord as f32 + 0.5) * src_size as f32 / tile_size as f32) as u32;
    s.min(src_size.saturating_sub(1))
}

/// Draw an arrowed line from (x1,y1) to (x2,y2) with the given RGBA color.
/// Uses Bresenham's algorithm for the shaft and two tip lines at ±30° from reverse direction.
/// `tip_length` is the ratio of tip line length to shaft length (0.2 = 20%).
fn draw_arrowed_line(
    img: &mut ImageRgba8,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    color: [u8; 4],
    tip_length: f32,
) {
    // Skip zero-length arrows entirely. `draw_line` paints the start pixel
    // before its termination check, so without this guard every still-region
    // grid point gets a single magenta pixel — visible as dotted patterns.
    let dx = (x2 - x1) as f32;
    let dy = (y2 - y1) as f32;
    let length = dx.hypot(dy);
    if length < 1.0 {
        return;
    }

    // Draw the shaft
    draw_line(img, x1, y1, x2, y2, color);

    let tip_len = length * tip_length;
    // Unit vector from tip back toward start
    let ux = -dx / length;
    let uy = -dy / length;

    // ±30 degrees rotation for the two tip arms
    let cos30: f32 = 0.866_025_4;
    let sin30: f32 = 0.5;

    // Left tip arm
    let lx = ux.mul_add(cos30, -(uy * sin30));
    let ly = ux.mul_add(sin30, uy * cos30);
    let tip_lx = x2 + (lx * tip_len) as i32;
    let tip_ly = y2 + (ly * tip_len) as i32;
    draw_line(img, x2, y2, tip_lx, tip_ly, color);

    // Right tip arm
    let rx = ux.mul_add(cos30, uy * sin30);
    let ry = (-ux).mul_add(sin30, uy * cos30);
    let tip_rx = x2 + (rx * tip_len) as i32;
    let tip_ry = y2 + (ry * tip_len) as i32;
    draw_line(img, x2, y2, tip_rx, tip_ry, color);
}

/// Bresenham's line algorithm. Draws a 1-pixel-wide line from (x0,y0) to (x1,y1).
/// Skips pixels outside image bounds.
// allow(cast_possible_wrap): line coords are i32 for signed arithmetic
#[allow(clippy::cast_possible_wrap)]
fn draw_line(img: &mut ImageRgba8, x0: i32, y0: i32, x1: i32, y1: i32, color: [u8; 4]) {
    let w = img.width as i32;
    let h = img.height as i32;

    let mut x = x0;
    let mut y = y0;
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if x >= 0 && x < w && y >= 0 && y < h {
            let idx = (y as usize * img.width as usize + x as usize) * 4;
            img.data[idx..idx + 4].copy_from_slice(&color);
        }

        if x == x1 && y == y1 {
            break;
        }

        let e2 = 2 * err;
        if e2 >= dy {
            if x == x1 {
                break;
            }
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            if y == y1 {
                break;
            }
            err += dx;
            y += sy;
        }
    }
}
