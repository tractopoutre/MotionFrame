//! `MotionFrame` web app entry (main thread).
//!
//! This crate compiles only for `wasm32-unknown-unknown`. On a native target
//! it's an empty library so `cargo build --workspace` succeeds.

#![cfg(target_arch = "wasm32")]

use eframe::wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

mod app;
mod bridge;
mod worker_client;

/// `wasm-bindgen(start)` entry — initializes panic hook + logger and boots eframe.
///
/// # Panics
/// Panics if the host page is missing the expected DOM (no `window`, no
/// `document`, no `#motionframe_canvas`, or that element is not a `<canvas>`).
/// These are programmer/host-page errors, not user input.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Debug);
    log::info!("motionframe-web: start() called");

    wasm_bindgen_futures::spawn_local(async {
        let canvas = web_sys::window()
            .expect("no window")
            .document()
            .expect("no document")
            .get_element_by_id("motionframe_canvas")
            .expect("canvas missing")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("not a canvas");
        log::info!(
            "motionframe-web: canvas obtained ({}x{}), starting eframe…",
            canvas.width(),
            canvas.height()
        );
        // eframe with wgpu (Chrome WebGPU maxInterStageShaderComponents
        // compat was resolved in wgpu 23+ and remains fine in current wgpu).
        match eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|cc| {
                    let jp_font_available = app::install_fonts(&cc.egui_ctx);
                    Ok(Box::new(motionframe_ui::MotionFrameApp::new(
                        app::WebPlatform::new(jp_font_available),
                        cc,
                    )))
                }),
            )
            .await
        {
            Ok(()) => log::info!("motionframe-web: eframe started OK"),
            Err(e) => {
                log::error!("motionframe-web: eframe start FAILED: {e:?}");
                web_sys::console::error_1(&format!("eframe start failed: {e:?}").into());
            }
        }
    });
}
