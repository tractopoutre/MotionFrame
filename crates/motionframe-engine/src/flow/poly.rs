//! Per-pixel polynomial expansion for Farneback optical flow.
//!
//! Pure-Rust implementation derived from `OpenCV`'s `optflowgf.cpp`
//! (`FarnebackPolyExp`, `FarnebackPrepareGaussian`).
//! See THIRD-PARTY-LICENSES.md for license attribution.

use crate::pipeline::ImageF32;
use rayon::prelude::*;
use wide::f32x4;

/// Stores 5 polynomial expansion coefficients per pixel.
/// Layout per pixel: [r4, r6, r5, r2, r3] representing:
///   - r4, r5: diagonal quadratic coefficients (x², y²)
///   - r6: cross quadratic coefficient (xy)
///   - r2, r3: linear coefficients (x, y)
pub struct PolyImage {
    pub width: u32,
    pub height: u32,
    /// 5 coefficients per pixel: [r4, r6, r5, r2, r3]
    pub data: Vec<[f32; 5]>,
}

/// Precomputed 1D kernels for polynomial expansion.
/// Computed in f64, stored as f32 for application.
// Based on OpenCV's optflowgf.cpp: FarnebackPrepareGaussian
struct PolyKernels {
    /// Gaussian weights kernel (size = 2*n+1)
    g: Vec<f32>,
    /// x * Gaussian kernel
    xg: Vec<f32>,
    /// x² * Gaussian kernel
    xxg: Vec<f32>,
    /// Inverse of Gram matrix diagonal entries needed for coefficient extraction
    ig11: f64,
    ig03: f64,
    ig33: f64,
    ig55: f64,
}

/// Compute polynomial expansion kernels from (`poly_n`, `poly_sigma`).
/// Kernels computed in f64 for precision, stored as f32.
// Based on OpenCV's optflowgf.cpp: FarnebackPrepareGaussian
// allow(cast_possible_wrap): poly_n is typically 5 or 7, always fits i32
#[allow(clippy::cast_possible_wrap)]
fn prepare_kernels(poly_n: u32, poly_sigma: f32) -> PolyKernels {
    let n = poly_n as i32;
    let sigma = f64::from(poly_sigma);
    let size = 2 * n + 1;

    let mut g_raw = vec![0.0f64; size as usize];

    // Compute raw Gaussian
    let mut s = 0.0f64;
    for i in -n..=n {
        let idx = (i + n) as usize;
        let x = f64::from(i);
        let w = (-x * x / (2.0 * sigma * sigma)).exp();
        g_raw[idx] = w;
        s += w;
    }

    // Normalize g (sum = 1), compute xg and xxg from normalized g
    // Based on OpenCV's optflowgf.cpp: g gets normalized first, then xg=x*g, xxg=x²*g
    let mut g_norm = vec![0.0f64; size as usize];
    let mut xg = vec![0.0f64; size as usize];
    let mut xxg = vec![0.0f64; size as usize];

    for i in -n..=n {
        let idx = (i + n) as usize;
        let x = f64::from(i);
        g_norm[idx] = g_raw[idx] / s;
        // OpenCV's FarnebackPrepareGaussian uses xg[k] = -k * g[k] (negative convention)
        // This sign convention ensures linear coefficients match OpenCV's output.
        xg[idx] = -x * g_norm[idx];
        xxg[idx] = x * x * g_norm[idx];
    }

    // Compute the 2D Gram matrix elements for basis [1, x, y, x², y², xy]
    // with separable 2D weight g_norm[x]*g_norm[y].
    // Due to symmetry, only 4 unique values matter:
    //   G(0,0) = 1 (sum of weights = 1)
    //   G(1,1) = G(2,2) = sum(x²*g_norm) = m2
    //   G(3,3) = G(4,4) = sum(x⁴*g_norm) = m4
    //   G(5,5) = m2²
    //   G(0,3) = G(3,0) = G(0,4) = G(4,0) = m2
    //   G(3,4) = G(4,3) = m2²
    // Based on OpenCV's optflowgf.cpp: 6×6 Gram matrix construction and inversion
    let mut m2 = 0.0f64;
    let mut m4 = 0.0f64;
    for i in -n..=n {
        let idx = (i + n) as usize;
        let x = f64::from(i);
        m2 += x * x * g_norm[idx];
        m4 += x * x * x * x * g_norm[idx];
    }

    // Inverse Gram matrix elements (derived from block structure):
    // Linear block {1,2}: diagonal with G[1][1] = m2 → inv = 1/m2
    let ig11 = 1.0 / m2;
    // Quadratic block {0,3,4}: 3×3 with det = (m4 - m2²)²
    // inv[0][1] = inv[0][2] = -m2/(m4-m2²)
    // inv[1][1] = inv[2][2] = 1/(m4-m2²)
    let ig03 = -m2 / m2.mul_add(-m2, m4);
    let ig33 = 1.0 / m2.mul_add(-m2, m4);
    // Cross block {5}: scalar G[5][5] = m2² → inv = 1/m2²
    let ig55 = 1.0 / (m2 * m2);

    let g_f32: Vec<f32> = g_norm.iter().map(|&v| v as f32).collect();
    let xg_f32: Vec<f32> = xg.iter().map(|&v| v as f32).collect();
    let xxg_f32: Vec<f32> = xxg.iter().map(|&v| v as f32).collect();

    PolyKernels {
        g: g_f32,
        xg: xg_f32,
        xxg: xxg_f32,
        ig11,
        ig03,
        ig33,
        ig55,
    }
}

/// Scalar convolution of a single pixel in the horizontal pass.
/// Applies `BORDER_REPLICATE` clamping for pixels near edges.
/// Uses explicit multiply+add (no FMA) to match the SIMD path bit-for-bit.
// Based on OpenCV's optflowgf.cpp: FarnebackPolyExp horizontal pass (per-pixel)
#[inline]
// allow(cast_possible_wrap): kernel offset and pixel coords, always small
#[allow(clippy::cast_possible_wrap)]
fn convolve_row_pixel_scalar(
    img: &ImageF32,
    kernels: &PolyKernels,
    row_offset: usize,
    x: usize,
    w: usize,
    n: i32,
    ksize: usize,
) -> (f32, f32, f32) {
    let mut sg = 0.0f32;
    let mut sxg = 0.0f32;
    let mut sxxg = 0.0f32;
    for k in 0..ksize {
        let sx = x as i32 + k as i32 - n;
        // BORDER_REPLICATE (clamp) matching OpenCV's horizontal border extension
        let sx_clamped = sx.max(0).min(w as i32 - 1) as usize;
        let val = img.data[row_offset + sx_clamped];
        sg += val * kernels.g[k];
        sxg += val * kernels.xg[k];
        sxxg += val * kernels.xxg[k];
    }
    (sg, sxg, sxxg)
}

/// Compute per-pixel polynomial expansion for a single-channel image.
///
/// For each pixel, fits a quadratic polynomial in a local neighborhood
/// and returns the 5 expansion coefficients.
///
/// Border policy: `BORDER_REPLICATE` (clamp to edge), matching `OpenCV`'s `FarnebackPolyExp`.
///
/// Uses SIMD (`wide::f32x4`) for the interior of the horizontal convolution pass,
/// with scalar fallback for border and tail pixels.
// Based on OpenCV's optflowgf.cpp: FarnebackPolyExp
// allow(cast_possible_wrap): kernel offset arithmetic, always small values
// allow(too_many_lines): single logical convolution algorithm, splitting would obscure flow
#[allow(clippy::cast_possible_wrap, clippy::too_many_lines)]
pub fn poly_expansion(img: &ImageF32, poly_n: u32, poly_sigma: f32) -> PolyImage {
    let kernels = prepare_kernels(poly_n, poly_sigma);
    let w = img.width as usize;
    let h = img.height as usize;
    let n = poly_n as i32;
    let ksize = (2 * n + 1) as usize;

    // Step 1: Row-wise convolutions to get intermediate buffers (parallelized by row).
    // For each row, convolve with g, xg, xxg to get:
    //   row_g[y][x] = sum_k img[y][x+k] * g[k]    (smoothed in x)
    //   row_xg[y][x] = sum_k img[y][x+k] * xg[k]  (x-derivative-like)
    //   row_xxg[y][x] = sum_k img[y][x+k] * xxg[k] (x²-moment-like)
    let mut row_g = vec![0.0f32; w * h];
    let mut row_xg = vec![0.0f32; w * h];
    let mut row_xxg = vec![0.0f32; w * h];

    let n_usize = n as usize;

    row_g
        .par_chunks_mut(w)
        .zip(row_xg.par_chunks_mut(w))
        .zip(row_xxg.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, ((rg_row, rxg_row), rxxg_row))| {
            let row_offset = y * w;

            // Left border: scalar with clamping (x in 0..n_usize)
            for x in 0..n_usize.min(w) {
                let (sg, sxg, sxxg) =
                    convolve_row_pixel_scalar(img, &kernels, row_offset, x, w, n, ksize);
                rg_row[x] = sg;
                rxg_row[x] = sxg;
                rxxg_row[x] = sxxg;
            }

            // Interior: SIMD path — no border clamping needed.
            let interior_start = n_usize;
            let interior_end = if w > n_usize { w - n_usize } else { n_usize };
            let interior_len = interior_end.saturating_sub(interior_start);
            let simd_end = interior_start + (interior_len / 4) * 4;

            let mut x = interior_start;
            while x < simd_end {
                let mut sg = f32x4::ZERO;
                let mut sxg = f32x4::ZERO;
                let mut sxxg = f32x4::ZERO;
                for k in 0..ksize {
                    let base = row_offset + x + k - n_usize;
                    // can't fail: interior SIMD loop guarantees base..base+4 within row bounds
                    let arr: [f32; 4] = img.data[base..base + 4].try_into().unwrap();
                    let vals = f32x4::new(arr);
                    let g_k = f32x4::splat(kernels.g[k]);
                    let xg_k = f32x4::splat(kernels.xg[k]);
                    let xxg_k = f32x4::splat(kernels.xxg[k]);
                    sg += vals * g_k;
                    sxg += vals * xg_k;
                    sxxg += vals * xxg_k;
                }
                let sg_arr: [f32; 4] = sg.into();
                let sxg_arr: [f32; 4] = sxg.into();
                let sxxg_arr: [f32; 4] = sxxg.into();
                rg_row[x..x + 4].copy_from_slice(&sg_arr);
                rxg_row[x..x + 4].copy_from_slice(&sxg_arr);
                rxxg_row[x..x + 4].copy_from_slice(&sxxg_arr);
                x += 4;
            }

            // Scalar tail for remaining interior pixels + right border
            for x in simd_end..w {
                let (sg, sxg, sxxg) =
                    convolve_row_pixel_scalar(img, &kernels, row_offset, x, w, n, ksize);
                rg_row[x] = sg;
                rxg_row[x] = sxg;
                rxxg_row[x] = sxxg;
            }
        });

    // Step 2: Column-wise convolutions to get the final 5 coefficients.
    // Apply g, xg, xxg in the y-direction on the row-convolved buffers.
    // Parallelized by row — each output row is independent.
    // Uses k-outer/x-inner loop order for cache-friendly sequential access.
    let mut result = PolyImage {
        width: img.width,
        height: img.height,
        data: vec![[0.0f32; 5]; w * h],
    };

    // f32 ig values + accumulators so LLVM can vectorize the vertical pass.
    let ig11 = kernels.ig11 as f32;
    let ig03 = kernels.ig03 as f32;
    let ig33 = kernels.ig33 as f32;
    let ig55 = kernels.ig55 as f32;
    let g_kern = &kernels.g;
    let xg_kern = &kernels.xg;
    let xxg_kern = &kernels.xxg;

    result
        .data
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, dst_row)| {
            // Precompute clamped source row indices for this output row
            let sy_indices: Vec<usize> = (0..ksize)
                .map(|k| {
                    let sy = y as i32 + k as i32 - n;
                    sy.max(0).min(h as i32 - 1) as usize
                })
                .collect();

            let mut b1_row = vec![0.0f32; w];
            let mut b2_row = vec![0.0f32; w];
            let mut b3_row = vec![0.0f32; w];
            let mut b4_row = vec![0.0f32; w];
            let mut b5_row = vec![0.0f32; w];
            let mut b6_row = vec![0.0f32; w];

            // k-outer loop: sequential access through row buffers for each kernel tap
            for k in 0..ksize {
                let row_base = sy_indices[k] * w;
                let g_k = g_kern[k];
                let xg_k = xg_kern[k];
                let xxg_k = xxg_kern[k];

                for x in 0..w {
                    let rg = row_g[row_base + x];
                    let rxg = row_xg[row_base + x];
                    let rxxg = row_xxg[row_base + x];

                    b1_row[x] = g_k.mul_add(rg, b1_row[x]);
                    b2_row[x] = xg_k.mul_add(rg, b2_row[x]);
                    b4_row[x] = xxg_k.mul_add(rg, b4_row[x]);
                    b3_row[x] = g_k.mul_add(rxg, b3_row[x]);
                    b5_row[x] = g_k.mul_add(rxxg, b5_row[x]);
                    b6_row[x] = xg_k.mul_add(rxg, b6_row[x]);
                }
            }

            // Final coefficient computation
            for (x, dst_elem) in dst_row.iter_mut().enumerate() {
                let r4 = b5_row[x].mul_add(ig33, b1_row[x] * ig03);
                let r5 = b4_row[x].mul_add(ig33, b1_row[x] * ig03);
                let r6 = b6_row[x] * ig55;
                let r2 = b3_row[x] * ig11;
                let r3 = b2_row[x] * ig11;

                *dst_elem = [r4, r6, r5, r2, r3];
            }
        });

    result
}

/// Scalar-only polynomial expansion (no SIMD) for parity testing.
/// Produces bit-identical results to the SIMD path for interior pixels,
/// and uses the same clamping logic for border pixels.
// Based on OpenCV's optflowgf.cpp: FarnebackPolyExp (scalar reference)
#[cfg(test)]
// allow(cast_possible_wrap): same arithmetic as poly_expansion, test-only code
#[allow(clippy::cast_possible_wrap)]
pub(crate) fn poly_expansion_scalar(img: &ImageF32, poly_n: u32, poly_sigma: f32) -> PolyImage {
    let kernels = prepare_kernels(poly_n, poly_sigma);
    let w = img.width as usize;
    let h = img.height as usize;
    let n = poly_n as i32;
    let ksize = (2 * n + 1) as usize;

    let mut row_g = vec![0.0f32; w * h];
    let mut row_xg = vec![0.0f32; w * h];
    let mut row_xxg = vec![0.0f32; w * h];

    for y in 0..h {
        let row_offset = y * w;
        for x in 0..w {
            let (sg, sxg, sxxg) =
                convolve_row_pixel_scalar(img, &kernels, row_offset, x, w, n, ksize);
            row_g[row_offset + x] = sg;
            row_xg[row_offset + x] = sxg;
            row_xxg[row_offset + x] = sxxg;
        }
    }

    let mut result = PolyImage {
        width: img.width,
        height: img.height,
        data: vec![[0.0f32; 5]; w * h],
    };

    let ig11 = kernels.ig11 as f32;
    let ig03 = kernels.ig03 as f32;
    let ig33 = kernels.ig33 as f32;
    let ig55 = kernels.ig55 as f32;

    for y in 0..h {
        for x in 0..w {
            let mut b1 = 0.0f32;
            let mut b2 = 0.0f32;
            let mut b3 = 0.0f32;
            let mut b4 = 0.0f32;
            let mut b5 = 0.0f32;
            let mut b6 = 0.0f32;

            for k in 0..ksize {
                let sy = y as i32 + k as i32 - n;
                let sy_clamped = sy.max(0).min(h as i32 - 1) as usize;
                let idx = sy_clamped * w + x;
                let g_k = kernels.g[k];
                let xg_k = kernels.xg[k];
                let xxg_k = kernels.xxg[k];
                let rg = row_g[idx];
                let rxg = row_xg[idx];
                let rxxg = row_xxg[idx];

                b1 = g_k.mul_add(rg, b1);
                b2 = xg_k.mul_add(rg, b2);
                b4 = xxg_k.mul_add(rg, b4);
                b3 = g_k.mul_add(rxg, b3);
                b5 = g_k.mul_add(rxxg, b5);
                b6 = xg_k.mul_add(rxg, b6);
            }

            let r4 = b5.mul_add(ig33, b1 * ig03);
            let r5 = b4.mul_add(ig33, b1 * ig03);
            let r6 = b6 * ig55;
            let r2 = b3 * ig11;
            let r3 = b2 * ig11;

            result.data[y * w + x] = [r4, r6, r5, r2, r3];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poly_expansion_zero_image_is_zero() {
        let img = ImageF32::zeros(16, 16);
        let poly = poly_expansion(&img, 5, 1.5);
        for coeffs in &poly.data {
            for &c in coeffs {
                assert!((c).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn poly_expansion_constant_image_zero_linear() {
        // A constant image should have zero linear and quadratic terms
        let mut img = ImageF32::zeros(32, 32);
        for v in &mut img.data {
            *v = 42.0;
        }
        let poly = poly_expansion(&img, 5, 1.5);
        // Check interior pixel (far from borders)
        let idx = 16 * 32 + 16;
        let [r4, r6, r5, r2, r3] = poly.data[idx];
        // Linear terms should be ~0
        assert!(r2.abs() < 1e-4, "r2 = {r2}");
        assert!(r3.abs() < 1e-4, "r3 = {r3}");
        // Cross term should be ~0
        assert!(r6.abs() < 1e-4, "r6 = {r6}");
        // Quadratic terms should be ~0 (constant has no curvature)
        assert!(r4.abs() < 1e-3, "r4 = {r4}");
        assert!(r5.abs() < 1e-3, "r5 = {r5}");
    }

    #[test]
    fn prepare_kernels_sizes_correct() {
        let k = prepare_kernels(5, 1.5);
        assert_eq!(k.g.len(), 11); // 2*5+1
        assert_eq!(k.xg.len(), 11);
        assert_eq!(k.xxg.len(), 11);
    }

    #[test]
    fn simd_matches_scalar_poly_expansion() {
        // Synthetic image with varied content to exercise all code paths.
        // Width chosen to NOT be divisible by 4, exercising the scalar tail.
        let w: u32 = 67;
        let h: u32 = 53;
        let mut img = ImageF32::zeros(w, h);
        for y in 0..h as usize {
            for x in 0..w as usize {
                img.data[y * w as usize + x] =
                    (x as f32).mul_add(0.1, (y as f32) * 0.07).sin() * 100.0;
            }
        }

        let simd_result = poly_expansion(&img, 5, 1.5);
        let scalar_result = poly_expansion_scalar(&img, 5, 1.5);

        assert_eq!(simd_result.data.len(), scalar_result.data.len());
        for (i, (s, r)) in simd_result
            .data
            .iter()
            .zip(scalar_result.data.iter())
            .enumerate()
        {
            for c in 0..5 {
                assert_eq!(
                    s[c].to_bits(),
                    r[c].to_bits(),
                    "Mismatch at pixel {i} coeff {c}: SIMD={}, scalar={}",
                    s[c],
                    r[c]
                );
            }
        }
    }
}
