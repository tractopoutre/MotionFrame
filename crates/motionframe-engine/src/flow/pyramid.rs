//! Gaussian pyramid construction for Farneback optical flow.
//!
//! Derived from `OpenCV`'s `optflowgf.cpp` (non-fastPyramids path).
//! See THIRD-PARTY-NOTICES.md for license attribution.

use crate::pipeline::ImageF32;
use rayon::prelude::*;
use wide::f32x4;

/// Compute actual number of pyramid levels, matching `OpenCV`'s `min_size=32` check.
/// Based on `OpenCV`'s optflowgf.cpp: level-clipping loop in `calcOpticalFlowFarneback`
// allow(cast_possible_wrap): level count and image dims always small enough for i32
#[allow(clippy::cast_possible_wrap)]
pub fn compute_num_levels(width: u32, height: u32, levels: u32, pyr_scale: f32) -> u32 {
    let min_size = 32;
    let mut k = 0u32;
    let mut scale = 1.0f64;
    while k < levels {
        scale *= f64::from(pyr_scale);
        if (f64::from(width) * scale) < f64::from(min_size)
            || (f64::from(height) * scale) < f64::from(min_size)
        {
            break;
        }
        k += 1;
    }
    k.max(1)
}

/// Build a blurred + resized image for a specific pyramid level.
///
/// Matches `OpenCV`'s non-fastPyramids path:
///   `GaussianBlur(original, sigma=(1/scale-1)*0.5, ksize=...)`
///   `resize(blurred, target_size, INTER_LINEAR)`
///
/// Based on `OpenCV`'s optflowgf.cpp: blur-then-resize in `calcOpticalFlowFarneback`
// allow(cast_possible_wrap): target dimensions from scale < original, always < i32::MAX
#[allow(clippy::cast_possible_wrap)]
pub fn build_level_image(img: &ImageF32, level_k: u32, pyr_scale: f32) -> ImageF32 {
    let scale = f64::from(pyr_scale).powi(level_k as i32);
    let target_w = (f64::from(img.width) * scale).round() as u32;
    let target_h = (f64::from(img.height) * scale).round() as u32;

    // Compute sigma and kernel size matching OpenCV
    let sigma = ((1.0 / scale) - 1.0) * 0.5;
    let smooth_sz_raw = (sigma * 5.0).round() as i32 | 1;
    let smooth_sz = smooth_sz_raw.max(3) as usize;

    // When sigma=0 (finest level, scale=1), OpenCV computes sigma from ksize:
    // sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    let effective_sigma = if sigma <= 0.0 {
        (smooth_sz as f64 - 1.0)
            .mul_add(0.5, -1.0)
            .mul_add(0.3, 0.8)
    } else {
        sigma
    };

    // Apply Gaussian blur to original image
    let blurred = gaussian_blur_separable(img, smooth_sz, effective_sigma);

    // Resize to target dimensions
    if target_w == img.width && target_h == img.height {
        blurred
    } else {
        resize_bilinear(&blurred, target_w, target_h)
    }
}

/// Separable Gaussian blur with `BORDER_REFLECT_101`.
/// Based on `OpenCV`'s `GaussianBlur` (default `borderType` = `BORDER_REFLECT_101`)
// allow(cast_possible_wrap): kernel half-size used as offset, always small
#[allow(clippy::cast_possible_wrap)]
fn gaussian_blur_separable(img: &ImageF32, ksize: usize, sigma: f64) -> ImageF32 {
    let kernel = make_gaussian_kernel(ksize, sigma);
    let w = img.width as usize;
    let h = img.height as usize;
    let half = (ksize / 2) as i32;
    let half_u = ksize / 2;

    // Precompute f32 kernel for SIMD path
    let kernel_f32: Vec<f32> = kernel.iter().map(|&v| v as f32).collect();

    // Horizontal pass — parallelized by row with SIMD interior
    let mut horiz = vec![0.0f32; w * h];
    horiz.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        // Left border: scalar with reflection
        for (x, row_elem) in row[..half_u.min(w)].iter_mut().enumerate() {
            let mut sum = 0.0f64;
            for (k, &kval) in kernel.iter().enumerate().take(ksize) {
                let sx = reflect_101(x as i32 + k as i32 - half, w as i32) as usize;
                sum = f64::from(img.data[y * w + sx]).mul_add(kval, sum);
            }
            *row_elem = sum as f32;
        }

        // Interior: SIMD path (no reflection needed)
        let interior_start = half_u;
        let interior_end = w.saturating_sub(half_u);
        let interior_len = interior_end.saturating_sub(interior_start);
        let simd_end = interior_start + (interior_len / 4) * 4;

        let mut x = interior_start;
        while x < simd_end {
            let mut acc = f32x4::ZERO;
            for (k, &kval) in kernel_f32.iter().enumerate().take(ksize) {
                let base = y * w + x + k - half_u;
                let arr: [f32; 4] = img.data[base..base + 4].try_into().unwrap();
                acc += f32x4::new(arr) * f32x4::splat(kval);
            }
            let result: [f32; 4] = acc.into();
            row[x..x + 4].copy_from_slice(&result);
            x += 4;
        }

        // Scalar tail + right border
        for (xi, row_elem) in row[simd_end..].iter_mut().enumerate() {
            let x = simd_end + xi;
            let mut sum = 0.0f64;
            for (k, &kval) in kernel.iter().enumerate().take(ksize) {
                let sx = reflect_101(x as i32 + k as i32 - half, w as i32) as usize;
                sum = f64::from(img.data[y * w + sx]).mul_add(kval, sum);
            }
            *row_elem = sum as f32;
        }
    });

    // Vertical pass — parallelized by row with SIMD interior
    let mut result = ImageF32::zeros(img.width, img.height);
    result
        .data
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            // Precompute reflected source row indices
            let src_rows: Vec<usize> = (0..ksize)
                .map(|k| reflect_101(y as i32 + k as i32 - half, h as i32) as usize)
                .collect();

            // SIMD interior
            let simd_end = (w / 4) * 4;
            let mut x = 0;
            while x < simd_end {
                let mut acc = f32x4::ZERO;
                for (k, &kval) in kernel_f32.iter().enumerate().take(ksize) {
                    let base = src_rows[k] * w + x;
                    let arr: [f32; 4] = horiz[base..base + 4].try_into().unwrap();
                    acc += f32x4::new(arr) * f32x4::splat(kval);
                }
                let out: [f32; 4] = acc.into();
                row[x..x + 4].copy_from_slice(&out);
                x += 4;
            }

            // Scalar tail
            for (xi, row_elem) in row[simd_end..].iter_mut().enumerate() {
                let x = simd_end + xi;
                let mut sum = 0.0f64;
                for (k, &kval) in kernel.iter().enumerate().take(ksize) {
                    let sy = src_rows[k];
                    sum = f64::from(horiz[sy * w + x]).mul_add(kval, sum);
                }
                *row_elem = sum as f32;
            }
        });

    result
}

/// Build a 1D Gaussian kernel (normalized, f64).
/// Based on `OpenCV`'s `getGaussianKernel`.
fn make_gaussian_kernel(ksize: usize, sigma: f64) -> Vec<f64> {
    let half = (ksize as f64 - 1.0) / 2.0;
    let scale = -0.5 / (sigma * sigma);
    let mut kernel = Vec::with_capacity(ksize);
    let mut sum = 0.0f64;
    for i in 0..ksize {
        let x = i as f64 - half;
        let v = (scale * x * x).exp();
        kernel.push(v);
        sum += v;
    }
    let inv_sum = 1.0 / sum;
    for v in &mut kernel {
        *v *= inv_sum;
    }
    kernel
}

/// Bilinear resize of a single-channel image.
/// Based on `OpenCV`'s resize with `INTER_LINEAR`.
/// Uses the standard coordinate mapping: `src_coord = (dst_coord + 0.5) * src_size/dst_size - 0.5`
pub fn resize_bilinear(img: &ImageF32, target_w: u32, target_h: u32) -> ImageF32 {
    let src_w = img.width as usize;
    let src_h = img.height as usize;
    let dst_w = target_w as usize;
    let dst_h = target_h as usize;
    let mut result = ImageF32::zeros(target_w, target_h);

    result
        .data
        .par_chunks_mut(dst_w)
        .enumerate()
        .for_each(|(dy, row)| {
            let sy = (dy as f64 + 0.5) * (src_h as f64) / (dst_h as f64) - 0.5;
            let sy = sy.max(0.0).min((src_h - 1) as f64);
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(src_h - 1);
            let fy = (sy - y0 as f64) as f32;

            for (dx, row_elem) in row.iter_mut().enumerate() {
                let sx = (dx as f64 + 0.5) * (src_w as f64) / (dst_w as f64) - 0.5;
                let sx = sx.max(0.0).min((src_w - 1) as f64);
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(src_w - 1);
                let fx = (sx - x0 as f64) as f32;

                let v00 = img.data[y0 * src_w + x0];
                let v10 = img.data[y0 * src_w + x1];
                let v01 = img.data[y1 * src_w + x0];
                let v11 = img.data[y1 * src_w + x1];

                *row_elem = (v00 * (1.0 - fx)).mul_add(
                    1.0 - fy,
                    (v10 * fx).mul_add(1.0 - fy, (v01 * (1.0 - fx)).mul_add(fy, v11 * fx * fy)),
                );
            }
        });

    result
}

/// Build a Gaussian pyramid for a single-channel f32 image (legacy, used by unit tests).
///
/// Returns pyramid levels from coarsest (index 0) to finest (last index).
/// Uses 5-tap binomial filter `[1, 4, 6, 4, 1] / 16` applied separably,
/// then downsamples by 2 between levels.
pub fn build_pyramid(img: &ImageF32, levels: u32, pyr_scale: f32) -> Vec<ImageF32> {
    let min_dim = f64::from(img.width.min(img.height));
    let log_base = (1.0_f64 / f64::from(pyr_scale)).ln();
    let max_levels_from_size = (min_dim.ln() / log_base).floor() as u32;
    let num_levels = levels.min(max_levels_from_size).max(1);

    let mut images = Vec::with_capacity(num_levels as usize);
    images.push(img.clone());

    for _ in 1..num_levels {
        let prev = images.last().unwrap(); // can't fail: images is non-empty (pushed before loop starts)
        let smoothed = smooth_binomial5(prev);
        let downsampled = downsample_2x(&smoothed);
        images.push(downsampled);
    }

    images.reverse();
    images
}

/// 5-tap binomial filter `[1, 4, 6, 4, 1] / 16` applied separably.
/// Border policy: `BORDER_REFLECT_101` (`gfedcb|abcdefgh|gfedcba`).
/// Uses SIMD (f32x4) for interior pixels and rayon parallelism by row.
// Based on OpenCV's optflowgf.cpp: pyrDown uses this kernel with reflect borders
// allow(cast_possible_wrap): small border offsets used in reflect logic
#[allow(clippy::cast_possible_wrap)]
fn smooth_binomial5(img: &ImageF32) -> ImageF32 {
    const INV16: f32 = 1.0 / 16.0;
    const WEIGHTS: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];

    let w = img.width as usize;
    let h = img.height as usize;

    // Horizontal pass — parallelized by row with SIMD interior
    let mut horiz = vec![0.0f32; w * h];
    horiz.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let src_row = &img.data[y * w..(y + 1) * w];

        // Left border (2 pixels)
        for (x, row_elem) in row[..2usize.min(w)].iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (k, &weight) in WEIGHTS.iter().enumerate() {
                let sx = reflect_101(x as i32 + k as i32 - 2, w as i32) as usize;
                sum = weight.mul_add(src_row[sx], sum);
            }
            *row_elem = sum * INV16;
        }

        // Interior: SIMD path (x in 2..w-2)
        let interior_start = 2usize.min(w);
        let interior_end = w.saturating_sub(2);
        let interior_len = interior_end.saturating_sub(interior_start);
        let simd_end = interior_start + (interior_len / 4) * 4;

        let mut x = interior_start;
        while x < simd_end {
            let mut acc = f32x4::ZERO;
            for (k, &weight) in WEIGHTS.iter().enumerate() {
                let base = x + k - 2;
                let arr: [f32; 4] = src_row[base..base + 4].try_into().unwrap();
                acc += f32x4::new(arr) * f32x4::splat(weight);
            }
            let result: [f32; 4] = (acc * f32x4::splat(INV16)).into();
            row[x..x + 4].copy_from_slice(&result);
            x += 4;
        }

        // Scalar tail + right border
        for (xi, row_elem) in row[simd_end..].iter_mut().enumerate() {
            let x = simd_end + xi;
            let mut sum = 0.0f32;
            for (k, &weight) in WEIGHTS.iter().enumerate() {
                let sx = reflect_101(x as i32 + k as i32 - 2, w as i32) as usize;
                sum = weight.mul_add(src_row[sx], sum);
            }
            *row_elem = sum * INV16;
        }
    });

    // Vertical pass — parallelized by row with SIMD interior
    let mut result = ImageF32::zeros(img.width, img.height);
    result
        .data
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            // Precompute source row indices
            let sy: [usize; 5] =
                std::array::from_fn(|k| reflect_101(y as i32 + k as i32 - 2, h as i32) as usize);

            // SIMD interior
            let simd_end = (w / 4) * 4;
            let mut x = 0;
            while x < simd_end {
                let mut acc = f32x4::ZERO;
                for (k, &weight) in WEIGHTS.iter().enumerate() {
                    let base = sy[k] * w + x;
                    let arr: [f32; 4] = horiz[base..base + 4].try_into().unwrap();
                    acc += f32x4::new(arr) * f32x4::splat(weight);
                }
                let out: [f32; 4] = (acc * f32x4::splat(INV16)).into();
                row[x..x + 4].copy_from_slice(&out);
                x += 4;
            }

            // Scalar tail
            for (xi, row_elem) in row[simd_end..].iter_mut().enumerate() {
                let x = simd_end + xi;
                let mut sum = 0.0f32;
                for (k, &weight) in WEIGHTS.iter().enumerate() {
                    sum = weight.mul_add(horiz[sy[k] * w + x], sum);
                }
                *row_elem = sum * INV16;
            }
        });

    result
}

/// Downsample an image by factor 2 (take every other pixel).
/// Output size = `((width+1)/2, (height+1)/2)` matching `OpenCV`'s `pyrDown`.
fn downsample_2x(img: &ImageF32) -> ImageF32 {
    let new_w = img.width.div_ceil(2);
    let new_h = img.height.div_ceil(2);
    let w = img.width as usize;
    let mut result = ImageF32::zeros(new_w, new_h);

    for y in 0..new_h as usize {
        for x in 0..new_w as usize {
            result.data[y * new_w as usize + x] = img.data[(y * 2) * w + (x * 2)];
        }
    }

    result
}

/// `BORDER_REFLECT_101`: `gfedcb|abcdefgh|gfedcba`
#[inline]
const fn reflect_101(idx: i32, len: i32) -> i32 {
    if idx >= 0 && idx < len {
        return idx;
    }
    let border = len - 1;
    if border == 0 {
        return 0;
    }
    let mut i = idx;
    if i < 0 {
        i = -i;
    }
    i %= 2 * border;
    if i >= len {
        i = 2 * border - i;
    }
    i
}

impl Clone for ImageF32 {
    fn clone(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            data: self.data.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramid_levels_decrease_in_size() {
        let img = ImageF32::zeros(64, 64);
        let pyr = build_pyramid(&img, 4, 0.5);
        // Coarsest first
        assert!(pyr[0].width < pyr[pyr.len() - 1].width);
        assert_eq!(pyr[pyr.len() - 1].width, 64);
        assert_eq!(pyr[pyr.len() - 1].height, 64);
    }

    #[test]
    fn pyramid_respects_max_levels_from_size() {
        let img = ImageF32::zeros(8, 8);
        // 8 -> 4 -> 2 -> 1: max 3 levels from log(8)/log(2) = 3
        let pyr = build_pyramid(&img, 10, 0.5);
        assert!(pyr.len() <= 3);
    }

    #[test]
    fn smooth_binomial5_constant_image_unchanged() {
        let mut img = ImageF32::zeros(10, 10);
        for v in &mut img.data {
            *v = 5.0;
        }
        let smoothed = smooth_binomial5(&img);
        // Interior pixels should be exactly 5.0 (kernel sums to 1 for interior)
        let val = smoothed.data[5 * 10 + 5];
        assert!((val - 5.0).abs() < 1e-5);
    }
}
