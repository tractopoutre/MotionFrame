//! `MotionFrame` engine: optical flow, atlas pack, encoding, optional preview.
//!
//! Disable the `preview` feature to omit wgpu/preview code.

pub mod debug_dump;
pub mod flow;
pub mod io;
pub mod pipeline;
pub mod viz;

#[cfg(feature = "preview")]
pub mod gpu;

#[cfg(feature = "preview")]
pub mod preview;
