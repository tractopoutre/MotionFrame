//! Shared image format detection and decoding.

use crate::pipeline::ImageRgba8;

/// Pick the image format from the filename extension.
///
/// TGA's magic bytes are in the file footer (not the header), so
/// `with_guessed_format()` is unreliable — we detect from extension instead.
fn format_from_name(name: &str) -> Option<image::ImageFormat> {
    let ext = std::path::Path::new(name).extension()?.to_str()?;
    match ext.to_ascii_lowercase().as_str() {
        "tga" => Some(image::ImageFormat::Tga),
        "png" => Some(image::ImageFormat::Png),
        "jpg" | "jpeg" => Some(image::ImageFormat::Jpeg),
        "bmp" => Some(image::ImageFormat::Bmp),
        "tiff" | "tif" => Some(image::ImageFormat::Tiff),
        _ => None,
    }
}

/// Decode raw image bytes into an `ImageRgba8` given a filename for format detection.
///
/// Returns an error string if the format is unsupported or decoding fails.
pub fn decode_image_from_bytes(name: &str, bytes: &[u8]) -> Result<ImageRgba8, String> {
    let format =
        format_from_name(name).ok_or_else(|| format!("unsupported image extension: {name}"))?;
    let cursor = std::io::Cursor::new(bytes);
    let img = image::ImageReader::with_format(cursor, format)
        .decode()
        .map_err(|e| format!("decode failed for {name}: {e}"))?
        .to_rgba8();
    Ok(ImageRgba8 {
        width: img.width(),
        height: img.height(),
        data: img.into_raw(),
    })
}

/// Peek at the dimensions of an encoded image without full decode.
///
/// Uses format detection from filename, same as [`decode_image_from_bytes`].
pub fn peek_dimensions_from_bytes(name: &str, bytes: &[u8]) -> Result<(u32, u32), String> {
    let format =
        format_from_name(name).ok_or_else(|| format!("unsupported image extension: {name}"))?;
    let cursor = std::io::Cursor::new(bytes);
    image::ImageReader::with_format(cursor, format)
        .into_dimensions()
        .map_err(|e| format!("dimension peek failed for {name}: {e}"))
}
