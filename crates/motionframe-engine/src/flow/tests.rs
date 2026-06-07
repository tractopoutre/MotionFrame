//! Internal unit tests for flow module.

use crate::pipeline::Flow;
use std::path::Path;

/// Read a .flo or .flo.gz file (Middlebury format) into a Flow struct (unit test helper).
pub fn read_flo(path: &Path) -> Flow {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("Cannot open {path:?}: {e}"));
    let mut buf = Vec::new();
    if path.extension().is_some_and(|e| e == "gz") {
        GzDecoder::new(file)
            .read_to_end(&mut buf)
            .unwrap_or_else(|e| panic!("Cannot decompress {path:?}: {e}"));
    } else {
        std::io::BufReader::new(file)
            .read_to_end(&mut buf)
            .unwrap_or_else(|e| panic!("Cannot read {path:?}: {e}"));
    }

    assert!(buf.len() >= 12, "File too small for .flo header");
    assert_eq!(&buf[0..4], b"PIEH", "Invalid .flo magic");

    let w = i32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as u32;
    let h = i32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as u32;

    let expected_size = 12 + (w as usize) * (h as usize) * 2 * 4;
    assert_eq!(buf.len(), expected_size, "File size mismatch");

    let mut data = vec![[0.0f32; 2]; (w as usize) * (h as usize)];
    let mut offset = 12;
    for pixel in &mut data {
        pixel[0] = f32::from_le_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
        ]);
        offset += 4;
        pixel[1] = f32::from_le_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
        ]);
        offset += 4;
    }

    Flow {
        width: w,
        height: h,
        data,
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::flow::farneback::farneback;
    use crate::pipeline::{FarnebackParams, ImageF32};

    #[test]
    fn read_flo_header() {
        let path = Path::new("tests/fixtures/explosion00_001_002.flo.gz");
        if path.exists() {
            let flow = read_flo(path);
            assert_eq!(flow.width, 400);
            assert_eq!(flow.height, 400);
            assert_eq!(flow.data.len(), 400 * 400);
        }
    }

    #[test]
    fn use_gaussian_produces_nonzero_flow() {
        // Verify that use_gaussian: true produces equivalent flow to use_gaussian: false.
        // Without the scale fix, Gaussian path would divide by an extra factor of N²,
        // making flow magnitude drastically smaller than box path.
        // With the fix, both paths should produce similar results since the kernels
        // differ only in weighting (box vs Gaussian shape), not in overall scale.
        let w: u32 = 64;
        let h: u32 = 64;

        let mut img1 = ImageF32::zeros(w, h);
        for y in 0..h as usize {
            for x in 0..w as usize {
                // 2D sinusoidal pattern provides non-zero curvature in both directions
                let val = (0.5 * (2.0 * std::f32::consts::PI * (x as f32) / 8.0).sin())
                    .mul_add((2.0 * std::f32::consts::PI * (y as f32) / 12.0).cos(), 0.5);
                img1.data[y * w as usize + x] = val;
            }
        }

        // img2 is img1 shifted right by 2 pixels
        let mut img2 = ImageF32::zeros(w, h);
        // allow(cast_possible_wrap): test image 64×64, trivially fits i32
        #[allow(clippy::cast_possible_wrap)]
        for y in 0..h as usize {
            for x in 2..w as usize {
                img2.data[y * w as usize + x] = img1.data[y * w as usize + x - 2];
            }
        }

        let base_params = FarnebackParams {
            levels: 1,
            iterations: 5,
            winsize: 5,
            poly_n: 5,
            poly_sigma: 1.1,
            pyr_scale: 0.5,
            use_gaussian: false,
        };

        let flow_box = farneback(&img1, &img2, &base_params);

        let gauss_params = FarnebackParams {
            use_gaussian: true,
            ..base_params
        };
        let flow_gauss = farneback(&img1, &img2, &gauss_params);

        // Compute mean magnitude for both
        let margin = 10usize;
        let mut sum_box = 0.0f64;
        let mut sum_gauss = 0.0f64;
        let mut count = 0u32;
        for y in margin..(h as usize - margin) {
            for x in margin..(w as usize - margin) {
                let [bx, by] = flow_box.data[y * w as usize + x];
                let [gx, gy] = flow_gauss.data[y * w as usize + x];
                sum_box += f64::from(bx).hypot(f64::from(by));
                sum_gauss += f64::from(gx).hypot(f64::from(gy));
                count += 1;
            }
        }
        let mean_box = sum_box / f64::from(count);
        let mean_gauss = sum_gauss / f64::from(count);

        // Both should produce non-zero flow
        assert!(
            mean_box > 1e-6,
            "box filter should produce non-zero flow, got {mean_box}"
        );
        assert!(
            mean_gauss > 1e-6,
            "Gaussian filter should produce non-zero flow, got {mean_gauss}"
        );

        // Key assertion: Gaussian and box should produce similar magnitude.
        // Without the fix, Gaussian would be ~N² = 25× smaller (for winsize=5).
        // With the fix, they should be within 2× of each other (shape difference only).
        let ratio = mean_gauss / mean_box;
        assert!(
            ratio > 0.3 && ratio < 3.0,
            "Gaussian/box flow ratio should be near 1.0, got {ratio:.4} \
             (gauss={mean_gauss:.6}, box={mean_box:.6})"
        );
    }
}
