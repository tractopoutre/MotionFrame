//! Atlas packing: stagger pack and flat pack.
//!
//! Stagger pack is a cross-process contract with the preview shader:
//! even tiles' R→B, G→G; odd tiles' R→R, G→A. Output is half-height.
//! Any change must land in encoder and shader simultaneously.

use crate::pipeline::{ImageRgba8, MotionVectorEncoding};

/// Stagger-pack: interleave pairs of motion tiles into a half-height 4-channel atlas.
///
/// | Input tile `i` | Source R → | Source G → |
/// |---|---|---|
/// | Even (`i % 2 == 0`) | B (ch 2) | G (ch 1) |
/// | Odd  (`i % 2 == 1`) | R (ch 0) | A (ch 3) |
///
/// The shader decodes `mv_tex.bg` for even frames and `mv_tex.ra` for odd.
pub fn stagger_pack(
    motion_atlas: &ImageRgba8,
    atlas_w: u32,
    atlas_h: u32,
    encoding: MotionVectorEncoding,
) -> ImageRgba8 {
    let tile_w = motion_atlas.width / atlas_w;
    let tile_h = motion_atlas.height / atlas_h;
    let num_tiles = atlas_w * atlas_h;
    let src_stride = motion_atlas.width as usize;

    // Output dimensions: same width, half the tile rows (ceil(num_tiles/2) / atlas_w rows)
    let output_tile_rows = num_tiles.div_ceil(2).div_ceil(atlas_w);
    let out_h = output_tile_rows * tile_h;
    let out_w = motion_atlas.width;

    let mut out = neutral_stagger_atlas(out_w, out_h, encoding);
    let dst_stride = out_w as usize;

    for i in 0..num_tiles {
        // Source tile position in input grid
        let src_tile_col = i % atlas_w;
        let src_tile_row = i / atlas_w;
        let src_x0 = (src_tile_col * tile_w) as usize;
        let src_y0 = (src_tile_row * tile_h) as usize;

        // Destination tile position in output grid
        let dst_tile_idx = i / 2;
        let dst_tile_col = dst_tile_idx % atlas_w;
        let dst_tile_row = dst_tile_idx / atlas_w;
        let dst_x0 = (dst_tile_col * tile_w) as usize;
        let dst_y0 = (dst_tile_row * tile_h) as usize;

        for row in 0..tile_h as usize {
            for col in 0..tile_w as usize {
                let src_idx = ((src_y0 + row) * src_stride + (src_x0 + col)) * 4;
                let src_r = motion_atlas.data[src_idx];
                let src_g = motion_atlas.data[src_idx + 1];

                let dst_idx = ((dst_y0 + row) * dst_stride + (dst_x0 + col)) * 4;

                if i % 2 == 0 {
                    // Even tiles: source R → output B (ch2), source G → output G (ch1)
                    out.data[dst_idx + 2] = src_r; // B channel
                    out.data[dst_idx + 1] = src_g; // G channel
                } else {
                    // Odd tiles: source R → output R (ch0), source G → output A (ch3)
                    out.data[dst_idx] = src_r; // R channel
                    out.data[dst_idx + 3] = src_g; // A channel
                }
            }
        }
    }

    out
}

fn neutral_stagger_atlas(width: u32, height: u32, encoding: MotionVectorEncoding) -> ImageRgba8 {
    let neutral = match encoding {
        MotionVectorEncoding::R8G8Remap01 => [127, 127, 127, 127],
        MotionVectorEncoding::SidefxLabsR8G8 => [0, 0, 0, 0],
    };
    let pixel_count = (width as usize) * (height as usize);
    let mut data = Vec::with_capacity(pixel_count * 4);
    for _ in 0..pixel_count {
        data.extend_from_slice(&neutral);
    }
    ImageRgba8 {
        width,
        height,
        data,
    }
}

/// Flat-pack: simple channel remapping — `B=0, G=motion.G, R=motion.R`.
/// No tile combining; output is same dimensions as input.
pub fn flat_pack(motion_atlas: &ImageRgba8) -> ImageRgba8 {
    let w = motion_atlas.width;
    let h = motion_atlas.height;

    let mut out = ImageRgba8 {
        width: w,
        height: h,
        data: vec![0u8; (w as usize) * (h as usize) * 4],
    };

    let pixel_count = (w as usize) * (h as usize);
    for i in 0..pixel_count {
        let src_idx = i * 4;
        let src_r = motion_atlas.data[src_idx];
        let src_g = motion_atlas.data[src_idx + 1];

        let dst_idx = i * 4;
        // RGBA byte order: R=motion.R, G=motion.G, B=0, A=255
        out.data[dst_idx] = src_r; // R
        out.data[dst_idx + 1] = src_g; // G
        out.data[dst_idx + 2] = 0; // B
        out.data[dst_idx + 3] = 255; // A
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stagger_pack_channel_mapping() {
        // 2×2 grid of tiles, each tile 2×2 pixels
        // Input: 4x4 atlas (2 tiles wide × 2 tiles tall, each tile 2×2)
        let atlas_w = 2u32;
        let atlas_h = 2u32;
        let tile_w = 2u32;
        let tile_h = 2u32;
        let input_w = atlas_w * tile_w; // 4
        let input_h = atlas_h * tile_h; // 4

        let mut input = ImageRgba8::zeros(input_w, input_h);

        // Tile 0 (top-left): R=10, G=20
        // Tile 1 (top-right): R=30, G=40
        // Tile 2 (bottom-left): R=50, G=60
        // Tile 3 (bottom-right): R=70, G=80
        let tiles = [(10u8, 20u8), (30u8, 40u8), (50u8, 60u8), (70u8, 80u8)];

        for tile_idx in 0..4u32 {
            let tc = tile_idx % atlas_w;
            let tr = tile_idx / atlas_w;
            let (r_val, g_val) = tiles[tile_idx as usize];
            for row in 0..tile_h {
                for col in 0..tile_w {
                    let x = tc * tile_w + col;
                    let y = tr * tile_h + row;
                    let idx = ((y as usize) * (input_w as usize) + (x as usize)) * 4;
                    input.data[idx] = r_val;
                    input.data[idx + 1] = g_val;
                    input.data[idx + 2] = 0;
                    input.data[idx + 3] = 255;
                }
            }
        }

        let result = stagger_pack(&input, atlas_w, atlas_h, MotionVectorEncoding::R8G8Remap01);

        // num_tiles=4, output_tile_rows = ceil(4/2)/2 = ceil(2/2)=1... let's compute:
        // (4+1)/2 = 2 (integer), then (2 + 2 - 1)/2 = 1
        // out_h = 1 * 2 = 2, out_w = 4
        assert_eq!(result.width, 4);
        assert_eq!(result.height, 2);

        // Tile 0 (even, i=0) → dst_tile_idx=0 at (0,0): B=10, G=20
        let idx00 = 0usize; // pixel (0,0)
        assert_eq!(result.data[idx00 * 4 + 2], 10); // B = src R
        assert_eq!(result.data[idx00 * 4 + 1], 20); // G = src G

        // Tile 1 (odd, i=1) → dst_tile_idx=0 at (0,0): R=30, A=40
        assert_eq!(result.data[idx00 * 4], 30); // R = src R
        assert_eq!(result.data[idx00 * 4 + 3], 40); // A = src G

        // Tile 2 (even, i=2) → dst_tile_idx=1 at (tile_w, 0) = (2, 0): B=50, G=60
        let idx_tile1 = 2usize; // pixel (2,0)
        assert_eq!(result.data[idx_tile1 * 4 + 2], 50); // B
        assert_eq!(result.data[idx_tile1 * 4 + 1], 60); // G

        // Tile 3 (odd, i=3) → dst_tile_idx=1 at (2, 0): R=70, A=80
        assert_eq!(result.data[idx_tile1 * 4], 70); // R
        assert_eq!(result.data[idx_tile1 * 4 + 3], 80); // A
    }

    #[test]
    fn stagger_pack_pads_unwritten_output_tiles_with_r8g8_neutral() {
        let atlas_w = 4u32;
        let atlas_h = 3u32;
        let tile_w = 1u32;
        let tile_h = 1u32;
        let mut input = ImageRgba8::zeros(atlas_w * tile_w, atlas_h * tile_h);

        for pixel in input.data.chunks_exact_mut(4) {
            pixel.copy_from_slice(&[127, 127, 0, 255]);
        }

        let result = stagger_pack(&input, atlas_w, atlas_h, MotionVectorEncoding::R8G8Remap01);
        let packed_tile_idx = 6usize;
        let idx = packed_tile_idx * 4;

        assert_eq!(&result.data[idx..idx + 4], &[127, 127, 127, 127]);
    }

    #[test]
    fn stagger_pack_pads_unwritten_output_tiles_with_sidefx_neutral() {
        let atlas_w = 4u32;
        let atlas_h = 3u32;
        let input = ImageRgba8::zeros(atlas_w, atlas_h);

        let result = stagger_pack(
            &input,
            atlas_w,
            atlas_h,
            MotionVectorEncoding::SidefxLabsR8G8,
        );
        let packed_tile_idx = 6usize;
        let idx = packed_tile_idx * 4;

        assert_eq!(&result.data[idx..idx + 4], &[0, 0, 0, 0]);
    }

    #[test]
    fn flat_pack_channel_remap() {
        let mut input = ImageRgba8::zeros(2, 2);
        // pixel 0: R=100, G=200
        input.data[0] = 100;
        input.data[1] = 200;
        input.data[2] = 0;
        input.data[3] = 255;

        let result = flat_pack(&input);

        // RGBA byte order: R=motion.R, G=motion.G, B=0, A=255
        assert_eq!(result.data[0], 100); // R = motion.R
        assert_eq!(result.data[1], 200); // G = motion.G
        assert_eq!(result.data[2], 0); // B = 0
        assert_eq!(result.data[3], 255); // A
    }
}
