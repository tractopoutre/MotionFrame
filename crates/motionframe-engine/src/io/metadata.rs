//! Atlas metadata (JSON) written alongside the TGA atlases.
//!
//! Six fields form a cross-tool contract: shaders use `strength` to
//! denormalize motion vectors, and `pack_mode`/`loop`/`atlas_width`/`atlas_height`
//! determine tile addressing. See DESIGN.md § "JSON metadata schema".

use std::path::Path;

use crate::pipeline::{PackMode, PipelineError};

/// Atlas metadata written alongside the TGA atlases.
#[derive(serde::Serialize)]
pub struct AtlasMetadata {
    pub strength: f64,
    pub total_frames: u32,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub pack_mode: PackMode,
    #[serde(rename = "loop")]
    pub is_loop: bool,
    /// Whether the color atlas is stored with premultiplied alpha. Consumers
    /// must blend `out = bg*(1-a) + rgb` when true and `out = bg*(1-a) + rgb*a`
    /// when false.
    pub premultiplied_alpha: bool,
}

/// Write atlas metadata as pretty-printed JSON.
pub fn write(path: &Path, meta: &AtlasMetadata) -> Result<(), PipelineError> {
    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(writer, meta)
        .map_err(|e| PipelineError::MetadataWriteFailed(path.to_path_buf(), e.to_string()))?;
    Ok(())
}

/// Build the metadata struct from an `EncodeResult` and serialize as pretty JSON.
///
/// In-memory equivalent of `write` for callers that need the JSON string
/// rather than a file.
pub fn build_metadata_json(
    result: &crate::pipeline::run::EncodeResult,
) -> Result<String, PipelineError> {
    let meta = AtlasMetadata {
        strength: result.strength,
        total_frames: result.total_frames,
        atlas_width: result.atlas_width,
        atlas_height: result.atlas_height,
        pack_mode: result.pack_mode,
        is_loop: result.is_loop,
        premultiplied_alpha: result.premultiplied_alpha,
    };
    serde_json::to_string_pretty(&meta).map_err(|e| {
        PipelineError::MetadataWriteFailed(std::path::PathBuf::from("<memory>"), e.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_metadata_json() {
        let meta = AtlasMetadata {
            strength: 0.123_456_78,
            total_frames: 64,
            atlas_width: 8,
            atlas_height: 8,
            pack_mode: PackMode::Staggered,
            is_loop: false,
            premultiplied_alpha: false,
        };
        let json = serde_json::to_string_pretty(&meta).unwrap();
        assert!(json.contains("\"loop\": false"));
        assert!(json.contains("\"pack_mode\": \"staggered\""));
        assert!(json.contains("\"total_frames\": 64"));
    }

    #[test]
    fn serialize_metadata_pack_mode_normal() {
        let meta = AtlasMetadata {
            strength: 0.0,
            total_frames: 1,
            atlas_width: 1,
            atlas_height: 1,
            pack_mode: PackMode::Normal,
            is_loop: false,
            premultiplied_alpha: false,
        };
        let json = serde_json::to_string_pretty(&meta).unwrap();
        assert!(json.contains("\"pack_mode\": \"normal\""));
    }
}
