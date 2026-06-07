//! Shared image format detection and decoding.

use crate::pipeline::ImageRgba8;

/// Cap on a decoded image's dimensions. Bounds the work a malformed or hostile
/// image (e.g. a header declaring 60000x60000) can force before any dimension
/// check runs. 16384 px/axis covers any realistic flipbook atlas.
const MAX_IMAGE_DIM: u32 = 16_384;
/// Allocation cap, derived from the dimension cap so the dimension limit is the
/// single binding constraint: any image within `MAX_IMAGE_DIM` decodes, and the
/// cap stays a backstop for compressed bombs that declare small dims but
/// allocate huge intermediates. `16384² * 4 bytes (RGBA) = 1 GiB`.
const MAX_IMAGE_ALLOC: u64 = (MAX_IMAGE_DIM as u64) * (MAX_IMAGE_DIM as u64) * 4;

/// Build the shared decode limits applied to every reader.
fn decode_limits() -> image::Limits {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIM);
    limits.max_image_height = Some(MAX_IMAGE_DIM);
    limits.max_alloc = Some(MAX_IMAGE_ALLOC);
    limits
}

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
    let mut reader = image::ImageReader::with_format(cursor, format);
    reader.limits(decode_limits());
    let img = reader
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
    let mut reader = image::ImageReader::with_format(cursor, format);
    reader.limits(decode_limits());
    reader
        .into_dimensions()
        .map_err(|e| format!("dimension peek failed for {name}: {e}"))
}
