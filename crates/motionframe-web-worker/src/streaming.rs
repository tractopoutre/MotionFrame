//! Streaming `FrameSource` backed by encoded byte blobs.
//!
//! Holds the encoded PNG/TGA bytes per frame (cheap) and decodes
//! on demand into an LRU cache (capacity 3, sized for Farneback's
//! sequential-pair access pattern).

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, PoisonError};

use lru::LruCache;
use motionframe_engine::io::{
    decode_image_from_bytes, peek_dimensions_from_bytes, FrameSource, FrameSourceError,
};
use motionframe_engine::pipeline::ImageRgba8;

/// One encoded frame: filename + raw bytes (PNG or TGA).
pub struct EncodedFrame {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// `FrameSource` that decodes lazily and caches the most recent few frames.
pub struct StreamingFrames {
    encoded: Vec<EncodedFrame>,
    dims: (u32, u32),
    // Mutex required by FrameSource: Send + Sync. Single-threaded in practice
    // (wasm32 worker), so contention is zero. Recover from poisoning gracefully.
    cache: Mutex<LruCache<usize, Arc<ImageRgba8>>>,
}

/// LRU cache capacity: 3 frames covers Farneback's sequential-pair access.
const FRAME_CACHE_CAPACITY: NonZeroUsize = NonZeroUsize::new(3).unwrap();

impl StreamingFrames {
    /// Build from a vector of encoded frames. Decodes the first frame to
    /// determine reference dimensions, then header-peeks every other frame
    /// to verify consistency.
    ///
    /// # Errors
    /// Returns [`FrameSourceError::Decode`] if the first frame cannot be
    /// decoded or any subsequent frame's header cannot be parsed, and
    /// [`FrameSourceError::DimensionMismatch`] if a later frame's dimensions
    /// disagree with the first.
    pub fn new(encoded: Vec<EncodedFrame>) -> Result<Self, FrameSourceError> {
        if encoded.is_empty() {
            return Ok(Self {
                encoded,
                dims: (0, 0),
                cache: Mutex::new(LruCache::new(FRAME_CACHE_CAPACITY)),
            });
        }
        let first = decode_one(&encoded[0])?;
        let dims = (first.width, first.height);
        for (idx, ef) in encoded.iter().enumerate().skip(1) {
            let (w, h) = peek_dims(ef)?;
            if (w, h) != dims {
                return Err(FrameSourceError::DimensionMismatch {
                    idx,
                    got_w: w,
                    got_h: h,
                    exp_w: dims.0,
                    exp_h: dims.1,
                });
            }
        }
        let mut cache: LruCache<usize, Arc<ImageRgba8>> = LruCache::new(FRAME_CACHE_CAPACITY);
        cache.put(0, Arc::new(first));
        Ok(Self {
            encoded,
            dims,
            cache: Mutex::new(cache),
        })
    }
}

fn peek_dims(ef: &EncodedFrame) -> Result<(u32, u32), FrameSourceError> {
    peek_dimensions_from_bytes(&ef.name, &ef.bytes).map_err(|e| FrameSourceError::Decode {
        name: ef.name.clone(),
        source: image::ImageError::Unsupported(
            image::error::UnsupportedError::from_format_and_kind(
                image::error::ImageFormatHint::Name(ef.name.clone()),
                image::error::UnsupportedErrorKind::Format(image::error::ImageFormatHint::Name(e)),
            ),
        ),
    })
}

fn decode_one(ef: &EncodedFrame) -> Result<ImageRgba8, FrameSourceError> {
    decode_image_from_bytes(&ef.name, &ef.bytes).map_err(|e| FrameSourceError::Decode {
        name: ef.name.clone(),
        source: image::ImageError::Unsupported(
            image::error::UnsupportedError::from_format_and_kind(
                image::error::ImageFormatHint::Name(ef.name.clone()),
                image::error::UnsupportedErrorKind::Format(image::error::ImageFormatHint::Name(e)),
            ),
        ),
    })
}

impl FrameSource for StreamingFrames {
    fn len(&self) -> usize {
        self.encoded.len()
    }

    fn dimensions(&self) -> (u32, u32) {
        self.dims
    }

    fn get(&self, idx: usize) -> Result<Arc<ImageRgba8>, FrameSourceError> {
        if idx >= self.encoded.len() {
            return Err(FrameSourceError::OutOfRange {
                idx,
                len: self.encoded.len(),
            });
        }
        // Recover from poisoning: the cache data is still valid even if a
        // previous holder panicked (only LRU bookkeeping could be stale).
        let mut guard = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        // Fast path: in cache
        if let Some(hit) = guard.get(&idx).cloned() {
            return Ok(hit);
        }
        drop(guard);
        // Decode (potentially slow — don't hold the lock)
        let img = decode_one(&self.encoded[idx])?;
        let arc = Arc::new(img);
        let mut guard = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        guard.put(idx, Arc::clone(&arc));
        Ok(arc)
    }
}
