pub mod atlas;
pub mod decode;
pub mod metadata;
pub mod sequence;
pub mod source;
pub mod tga;

pub use atlas::{detect_tile_count, slice_atlas, AtlasError};
pub use decode::{decode_image_from_bytes, peek_dimensions_from_bytes};
pub use source::{FrameSource, FrameSourceError, InMemoryFrames};
