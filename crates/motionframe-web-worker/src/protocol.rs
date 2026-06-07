//! Worker ↔ main message types.
//!
//! Small messages (`Progress`, `Error`, `Cancelled`) are serialized via
//! `serde-wasm-bindgen`. The `Done` payload is built manually as a JS object
//! with transferable typed arrays — `serde_wasm_bindgen::from_value` on tens
//! of MB on the main thread froze the UI for ~1 s. See `worker.rs` for the
//! sender and `motionframe-web/src/worker_client.rs` for the receiver.

use motionframe_engine::pipeline::{GenerateOptions, Progress};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
/// Encoded source frame sent from the browser main thread to the worker.
pub struct EncodedFrameMsg {
    pub name: String,
    #[serde(with = "serde_bytes")]
    pub bytes: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
/// Commands accepted by the worker.
pub enum CmdToWorker {
    Generate {
        frames: Vec<EncodedFrameMsg>,
        options: GenerateOptions,
    },
    Cancel,
}

/// Pack mode in a serde-friendly form.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PackModeMsg {
    Staggered,
    Normal,
}

/// Scalar fields of the Done payload. Sent as the `meta` JS subobject.
#[derive(Debug, Serialize, Deserialize)]
pub struct DoneMeta {
    pub color_atlas_w: u32,
    pub color_atlas_h: u32,
    pub motion_atlas_w: u32,
    pub motion_atlas_h: u32,
    pub strength: f64,
    pub total_frames: u32,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub columns: u32,
    pub rows: u32,
    pub pack_mode: PackModeMsg,
    pub is_loop: bool,
    pub premultiplied_alpha: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
/// Non-terminal and error messages sent from the worker to the main thread.
pub enum MsgFromWorker {
    Progress(Progress),
    Error(String),
    Cancelled,
}
