//! `MotionFrame` shared egui widget toolkit.
//!
//! Pure widgets plus the shared web app shell.

pub mod app;
pub mod i18n;
pub mod input_panel;
pub mod platform;
pub mod playback;
pub mod tabs;

pub use app::MotionFrameApp;
pub use platform::{EncodedFrame, GenerationEvent, Platform};
