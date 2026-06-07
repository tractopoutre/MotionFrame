//! `MotionFrame` engine wasm worker.
//!
//! Wasm-only crate. Empty body on native targets so the workspace builds.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;

pub mod protocol;
pub mod streaming;
pub mod worker;

#[wasm_bindgen(start)]
/// Initialize logging and panic reporting inside the wasm worker.
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
    log::info!("motionframe-web-worker booted");
}

// Re-export for the JS-side worker.js
pub use worker::handle_message;

// Re-export for the JS-side worker.js to initialize the rayon thread pool.
pub use wasm_bindgen_rayon::init_thread_pool;
