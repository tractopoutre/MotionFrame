//! Frame source abstraction.
//!
//! `FrameSource` lets the pipeline iterate frames without owning the storage layout.
//! `InMemoryFrames` keeps all frames decoded in RAM.

use std::sync::Arc;

use crate::pipeline::ImageRgba8;

/// Errors a frame source can return.
#[derive(Debug, thiserror::Error)]
pub enum FrameSourceError {
    #[error("frame {idx} out of range (len = {len})")]
    OutOfRange { idx: usize, len: usize },
    #[error("decode error in frame {name}: {source}")]
    Decode {
        name: String,
        #[source]
        source: image::ImageError,
    },
    #[error("dimension mismatch: frame {idx} is {got_w}x{got_h}, expected {exp_w}x{exp_h}")]
    DimensionMismatch {
        idx: usize,
        got_w: u32,
        got_h: u32,
        exp_w: u32,
        exp_h: u32,
    },
}

/// Random-access source of RGBA frames.
pub trait FrameSource: Send + Sync {
    /// Number of frames.
    fn len(&self) -> usize;

    /// True if no frames.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Common dimensions of all frames (verified at construction time).
    fn dimensions(&self) -> (u32, u32);

    /// Fetch a frame by index. Implementations may decode lazily.
    fn get(&self, idx: usize) -> Result<Arc<ImageRgba8>, FrameSourceError>;
}

/// Frame source backed by a fully-decoded `Vec<Arc<ImageRgba8>>`.
///
/// Use on desktop where all frames fit in RAM.
#[derive(Debug)]
pub struct InMemoryFrames {
    frames: Vec<Arc<ImageRgba8>>,
    dims: (u32, u32),
}

impl InMemoryFrames {
    /// Build from a vector of decoded frames.
    /// Verifies all frames share the same dimensions.
    pub fn new(frames: Vec<ImageRgba8>) -> Result<Self, FrameSourceError> {
        let (exp_w, exp_h) = match frames.first() {
            Some(f) => (f.width, f.height),
            None => {
                return Ok(Self {
                    frames: Vec::new(),
                    dims: (0, 0),
                })
            }
        };
        for (idx, f) in frames.iter().enumerate() {
            if (f.width, f.height) != (exp_w, exp_h) {
                return Err(FrameSourceError::DimensionMismatch {
                    idx,
                    got_w: f.width,
                    got_h: f.height,
                    exp_w,
                    exp_h,
                });
            }
        }
        Ok(Self {
            frames: frames.into_iter().map(Arc::new).collect(),
            dims: (exp_w, exp_h),
        })
    }
}

impl FrameSource for InMemoryFrames {
    fn len(&self) -> usize {
        self.frames.len()
    }

    fn dimensions(&self) -> (u32, u32) {
        self.dims
    }

    fn get(&self, idx: usize) -> Result<Arc<ImageRgba8>, FrameSourceError> {
        self.frames
            .get(idx)
            .cloned()
            .ok_or(FrameSourceError::OutOfRange {
                idx,
                len: self.frames.len(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_frame(w: u32, h: u32) -> ImageRgba8 {
        ImageRgba8 {
            width: w,
            height: h,
            data: vec![0u8; (w * h * 4) as usize],
        }
    }

    #[test]
    fn in_memory_empty() {
        let s = InMemoryFrames::new(Vec::new()).unwrap();
        assert!(s.is_empty());
        assert_eq!(s.dimensions(), (0, 0));
    }

    #[test]
    fn in_memory_uniform() {
        let frames = vec![dummy_frame(8, 4), dummy_frame(8, 4)];
        let s = InMemoryFrames::new(frames).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.dimensions(), (8, 4));
        let f0 = s.get(0).unwrap();
        assert_eq!(f0.width, 8);
    }

    #[test]
    fn in_memory_mismatch_rejected() {
        let frames = vec![dummy_frame(8, 4), dummy_frame(8, 5)];
        let err = InMemoryFrames::new(frames).unwrap_err();
        assert!(matches!(err, FrameSourceError::DimensionMismatch { .. }));
    }

    #[test]
    fn in_memory_out_of_range() {
        let s = InMemoryFrames::new(vec![dummy_frame(2, 2)]).unwrap();
        let err = s.get(5).unwrap_err();
        assert!(matches!(err, FrameSourceError::OutOfRange { .. }));
    }
}
