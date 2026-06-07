//! Atlas-mode and sequence-mode produce identical outputs when the atlas
//! tiles equal the sequence frames.

use motionframe_engine::io::{slice_atlas, InMemoryFrames};
use motionframe_engine::pipeline::run::run_pipeline;
use motionframe_engine::pipeline::{GenerateOptions, ImageRgba8, Progress};

/// Build N synthetic frames that have small-but-real motion between them
/// so the pipeline produces non-trivial flow.
fn build_frames(n: u32, w: u32, h: u32) -> Vec<ImageRgba8> {
    (0..n)
        .map(|i| {
            let mut data = vec![0u8; (w * h * 4) as usize];
            // A small bright square translates by `i` px per frame.
            let sq = 8u32;
            let cx = 4 + i;
            let cy = 4 + i;
            for y in cy..(cy + sq).min(h) {
                for x in cx..(cx + sq).min(w) {
                    let idx = ((y * w + x) * 4) as usize;
                    data[idx] = 255;
                    data[idx + 1] = 255;
                    data[idx + 2] = 255;
                    data[idx + 3] = 255;
                }
            }
            ImageRgba8 {
                width: w,
                height: h,
                data,
            }
        })
        .collect()
}

/// Stitch frames into a `cols × rows` atlas (row-major, top-left origin).
fn build_atlas(frames: &[ImageRgba8], cols: u32, rows: u32) -> ImageRgba8 {
    assert_eq!(frames.len() as u32, cols * rows);
    let tile_w = frames[0].width;
    let tile_h = frames[0].height;
    let w = cols * tile_w;
    let h = rows * tile_h;
    let mut data = vec![0u8; (w * h * 4) as usize];
    for (i, f) in frames.iter().enumerate() {
        let i = i as u32;
        let tx = i % cols;
        let ty = i / cols;
        for dy in 0..tile_h {
            let dst_y = ty * tile_h + dy;
            let src_row = (dy * tile_w * 4) as usize;
            let dst_row = ((dst_y * w + tx * tile_w) * 4) as usize;
            data[dst_row..dst_row + (tile_w * 4) as usize]
                .copy_from_slice(&f.data[src_row..src_row + (tile_w * 4) as usize]);
        }
    }
    ImageRgba8 {
        width: w,
        height: h,
        data,
    }
}

#[test]
fn atlas_mode_matches_sequence_mode() {
    let cols = 2u32;
    let rows = 2u32;
    let frames = build_frames(cols * rows, 32, 32);
    let atlas = build_atlas(&frames, cols, rows);

    // Round-trip atlas → tiles must equal the original frames.
    let tiles = slice_atlas(&atlas, cols, rows).unwrap();
    assert_eq!(tiles.len(), frames.len());
    for (a, b) in tiles.iter().zip(frames.iter()) {
        assert_eq!(a.width, b.width);
        assert_eq!(a.height, b.height);
        assert_eq!(a.data, b.data);
    }

    // Run the pipeline twice with identical options. atlas_dims (output) is
    // tiny so generation is fast.
    let opts = GenerateOptions {
        atlas_dims: (2, 2),
        tile_pixel_width: 32,
        output_frames: 4,
        ..Default::default()
    };

    let progress = |_: Progress| {};
    let cancel = || false;

    let src_seq = InMemoryFrames::new(frames).unwrap();
    let r_seq = run_pipeline(&src_seq, &opts, &progress, &cancel).expect("sequence run");

    let src_atlas = InMemoryFrames::new(tiles).unwrap();
    let r_atlas = run_pipeline(&src_atlas, &opts, &progress, &cancel).expect("atlas run");

    assert_eq!(r_seq.color_atlas.data, r_atlas.color_atlas.data);
    assert_eq!(r_seq.motion_atlas.data, r_atlas.motion_atlas.data);
    assert!((r_seq.strength - r_atlas.strength).abs() < 1e-9);
}
