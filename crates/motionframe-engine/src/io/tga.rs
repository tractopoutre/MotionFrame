//! TGA load/save and alpha premultiplication.

use std::path::Path;

use image::ImageEncoder;

use crate::pipeline::{ImageRgba8, PipelineError};

/// Load a TGA (or other supported format) as RGBA8.
pub fn load_rgba(path: &Path) -> Result<ImageRgba8, PipelineError> {
    let img = image::open(path)
        .map_err(|e| PipelineError::DecodeFailed(path.to_path_buf(), e.to_string()))?;
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    Ok(ImageRgba8 {
        width: w,
        height: h,
        data: rgba.into_raw(),
    })
}

/// Save an `ImageRgba8` as a TGA file.
pub fn save_tga(path: &Path, image: &ImageRgba8) -> Result<(), PipelineError> {
    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    let encoder = image::codecs::tga::TgaEncoder::new(writer);
    encoder
        .write_image(
            &image.data,
            image.width,
            image.height,
            image::ExtendedColorType::Rgba8,
        )
        .map_err(|e| PipelineError::EncodeFailed(path.to_path_buf(), e.to_string()))?;
    Ok(())
}

/// Encode an `ImageRgba8` as TGA bytes.
pub fn encode_tga_bytes(image: &ImageRgba8) -> Result<Vec<u8>, PipelineError> {
    let mut bytes = Vec::new();
    let encoder = image::codecs::tga::TgaEncoder::new(&mut bytes);
    encoder
        .write_image(
            &image.data,
            image.width,
            image.height,
            image::ExtendedColorType::Rgba8,
        )
        .map_err(|e| {
            PipelineError::EncodeFailed(Path::new("<memory>").to_path_buf(), e.to_string())
        })?;
    Ok(bytes)
}

/// Premultiply alpha channel using rounding (not truncation).
///
/// Rounding preserves the top byte value (e.g. `255 * 0.999 + 0.5 → 255`)
/// whereas truncation would lose it (`255 * 0.999 → 254`).
pub fn premultiply_alpha(rgba: &ImageRgba8) -> ImageRgba8 {
    let mut out = ImageRgba8::zeros(rgba.width, rgba.height);
    for (chunk_in, chunk_out) in rgba.data.chunks_exact(4).zip(out.data.chunks_exact_mut(4)) {
        let a = f32::from(chunk_in[3]) / 255.0;
        chunk_out[0] = f32::from(chunk_in[0]).mul_add(a, 0.5).min(255.0) as u8;
        chunk_out[1] = f32::from(chunk_in[1]).mul_add(a, 0.5).min(255.0) as u8;
        chunk_out[2] = f32::from(chunk_in[2]).mul_add(a, 0.5).min(255.0) as u8;
        chunk_out[3] = chunk_in[3];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn save_tga_load_rgba_round_trips_pixels() {
        let img = ImageRgba8 {
            width: 2,
            height: 2,
            data: vec![
                255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 64, 255, 255, 0, 0,
            ],
        };
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before UNIX_EPOCH")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "motionframe-tga-roundtrip-{}-{nonce}.tga",
            std::process::id()
        ));

        save_tga(&path, &img).expect("save TGA");
        let loaded = load_rgba(&path).expect("load TGA");
        std::fs::remove_file(&path).expect("remove temp TGA");

        assert_eq!(loaded.width, img.width);
        assert_eq!(loaded.height, img.height);
        assert_eq!(loaded.data, img.data);
    }

    #[test]
    fn premultiply_opaque_unchanged() {
        let img = ImageRgba8 {
            width: 1,
            height: 1,
            data: vec![200, 100, 50, 255],
        };
        let result = premultiply_alpha(&img);
        // Opaque pixel: 200 * 1.0 + 0.5 = 200.5 → 200
        assert_eq!(result.data[0], 200);
        assert_eq!(result.data[1], 100);
        assert_eq!(result.data[2], 50);
        assert_eq!(result.data[3], 255);
    }

    #[test]
    fn premultiply_half_alpha_rounds() {
        let img = ImageRgba8 {
            width: 1,
            height: 1,
            data: vec![255, 127, 1, 128],
        };
        let result = premultiply_alpha(&img);
        // a = 128/255 ≈ 0.50196
        // R: 255 * 0.50196 + 0.5 = 128.5 → 128
        // G: 127 * 0.50196 + 0.5 = 64.249 → 64
        // B: 1 * 0.50196 + 0.5 = 1.002 → 1
        assert_eq!(result.data[0], 128);
        assert_eq!(result.data[1], 64);
        assert_eq!(result.data[2], 1);
        assert_eq!(result.data[3], 128);
    }

    #[test]
    fn premultiply_transparent_zeroes_rgb() {
        let img = ImageRgba8 {
            width: 1,
            height: 1,
            data: vec![255, 255, 255, 0],
        };
        let result = premultiply_alpha(&img);
        // a = 0/255 = 0.0; R: 0 + 0.5 = 0.5 → 0
        assert_eq!(result.data[0], 0);
        assert_eq!(result.data[1], 0);
        assert_eq!(result.data[2], 0);
        assert_eq!(result.data[3], 0);
    }
}
