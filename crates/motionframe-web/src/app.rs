//! Web shell — thin Platform impl.
//!
//! All UI is in `motionframe_ui::MotionFrameApp`. This file is platform glue:
//! folder picker via HTML bridge, downloads via Blob URL, generation via
//! Web Worker postMessage protocol.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::OnceLock;

use motionframe_engine::pipeline::{GenerateOptions, ImageRgba8};
use motionframe_ui::platform::{EncodedFrame, GenerationEvent, Platform};

use crate::bridge;
use crate::worker_client::{self, WorkerClient};

const THIRD_PARTY_LICENSES_DEFLATED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/licenses.deflate"));
const LINE_SEED_JP_DEFLATED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/line_seed_jp.deflate"));
const LINE_SEED_JP_FONT_NAME: &str = "line_seed_jp";

fn third_party_licenses() -> &'static str {
    static CACHE: OnceLock<String> = OnceLock::new();
    CACHE.get_or_init(|| {
        let bytes = miniz_oxide::inflate::decompress_to_vec(THIRD_PARTY_LICENSES_DEFLATED)
            .expect("THIRD-PARTY-LICENSES blob compressed at build time");
        String::from_utf8(bytes).expect("THIRD-PARTY-LICENSES.md is UTF-8")
    })
}

/// Install the bundled UI font into egui's font atlas.
pub fn install_fonts(ctx: &egui::Context) -> bool {
    let font_bytes = match miniz_oxide::inflate::decompress_to_vec(LINE_SEED_JP_DEFLATED) {
        Ok(bytes) => bytes,
        Err(err) => {
            log::error!("decompress LINE Seed JP font failed: {err:?}");
            return false;
        }
    };
    ctx.add_font(egui::epaint::text::FontInsert::new(
        LINE_SEED_JP_FONT_NAME,
        egui::FontData::from_owned(font_bytes),
        vec![
            egui::epaint::text::InsertFontFamily {
                family: egui::FontFamily::Proportional,
                priority: egui::epaint::text::FontPriority::Highest,
            },
            egui::epaint::text::InsertFontFamily {
                family: egui::FontFamily::Monospace,
                priority: egui::epaint::text::FontPriority::Highest,
            },
        ],
    ));
    true
}

/// Platform impl for the browser.
pub struct WebPlatform {
    pending_pick: Rc<RefCell<Option<Vec<EncodedFrame>>>>,
    worker: Option<WorkerClient>,
    ready_signaled: bool,
    start_error: Option<String>,
    jp_font_available: bool,
}

impl WebPlatform {
    pub fn new(jp_font_available: bool) -> Self {
        Self {
            pending_pick: Rc::new(RefCell::new(None)),
            worker: None,
            ready_signaled: false,
            start_error: None,
            jp_font_available,
        }
    }
}

impl Default for WebPlatform {
    fn default() -> Self {
        Self::new(false)
    }
}

impl Platform for WebPlatform {
    fn jp_font_available(&self) -> bool {
        self.jp_font_available
    }

    fn third_party_licenses(&self) -> Option<&'static str> {
        Some(third_party_licenses())
    }

    fn start_folder_pick(&mut self, ctx: &egui::Context) {
        log::info!("WebPlatform::start_folder_pick called");
        // Open the picker SYNCHRONOUSLY here while we're still inside the
        // browser's user-gesture window. spawn_local would defer .click() to
        // a microtask, which some browsers reject as not user-initiated.
        bridge::pick_directory_into(&self.pending_pick, ctx);
    }

    fn take_folder_pick(&mut self) -> Option<Vec<EncodedFrame>> {
        self.pending_pick.borrow_mut().take()
    }

    fn save_outputs(
        &mut self,
        prefix: &str,
        color_atlas: &ImageRgba8,
        motion_atlas: &ImageRgba8,
        metadata_json: &str,
    ) -> Result<(), String> {
        // Encode atlases to TGA in-memory, then download three files.
        let color_tga = motionframe_engine::io::tga::encode_tga_bytes(color_atlas)
            .map_err(|e| format!("encode color tga: {e}"))?;
        let motion_tga = motionframe_engine::io::tga::encode_tga_bytes(motion_atlas)
            .map_err(|e| format!("encode motion tga: {e}"))?;
        let prefix = if prefix.is_empty() { "output" } else { prefix };
        bridge::trigger_download(
            &format!("{prefix}_color_atlas.tga"),
            &color_tga,
            "image/x-tga",
        );
        bridge::trigger_download(
            &format!("{prefix}_motion_atlas.tga"),
            &motion_tga,
            "image/x-tga",
        );
        bridge::trigger_download(
            &format!("{prefix}_meta.json"),
            metadata_json.as_bytes(),
            "application/json",
        );
        Ok(())
    }

    fn start_generation(
        &mut self,
        frames: Vec<EncodedFrame>,
        options: GenerateOptions,
        _ctx: &egui::Context,
    ) {
        // Lazy-spawn the worker on first generation.
        if self.worker.is_none() {
            match WorkerClient::spawn() {
                Ok(w) => self.worker = Some(w),
                Err(e) => {
                    let msg = format!("worker spawn failed: {e:?}");
                    log::error!("{msg}");
                    self.start_error = Some(msg);
                    return;
                }
            }
        }
        let payload: Vec<(String, Vec<u8>)> =
            frames.into_iter().map(|f| (f.name, f.bytes)).collect();
        if let Some(ref w) = self.worker {
            if let Err(e) = w.send_generate(payload, options) {
                let msg = format!("send_generate failed: {e:?}");
                log::error!("{msg}");
                w.push_event(worker_client::WorkerEvent::Error(msg));
            }
        }
    }

    fn take_generation_start_error(&mut self) -> Option<String> {
        self.start_error.take()
    }

    fn take_generation_event(&mut self) -> Option<GenerationEvent> {
        let raw_events = self.worker.as_ref()?.drain_events();
        // We can only return one per call. Take the first; we'll be polled again next frame.
        // But egui repaints continuously while generating, so this is fine.
        let mut iter = raw_events.into_iter();
        let raw = iter.next()?;
        // Stash any extras back in the inbox for next call (in queue order).
        if let Some(ref w) = self.worker {
            let extras: Vec<_> = iter.collect();
            for extra in extras.into_iter().rev() {
                w.push_front(extra);
            }
        }
        Some(translate_event(raw))
    }

    fn cancel_generation(&mut self) {
        if let Some(ref w) = self.worker {
            let _ = w.send_cancel();
        }
    }

    fn handle_dropped_files(&mut self, _dropped: Vec<egui::DroppedFile>) -> Vec<EncodedFrame> {
        // Drops on web are intercepted by `bridge::install_drop_handler` at
        // the document level (capture phase) so that folder drops can be
        // walked via webkitGetAsEntry — eframe's DataTransfer.files-based
        // path can't see directory contents. Frames flow through pending_pick
        // instead of this path, so eframe's `dropped_files` is always empty.
        Vec::new()
    }

    fn signal_ready(&mut self, ctx: &egui::Context) {
        if self.ready_signaled {
            return;
        }
        self.ready_signaled = true;
        // Install the folder-aware drop handler now that the egui Context
        // exists. One-shot guarded by `ready_signaled`.
        bridge::install_drop_handler(&self.pending_pick, ctx);
        if let Some(win) = web_sys::window() {
            if let Ok(ev) = web_sys::Event::new("motionframe:ready") {
                let _ = win.dispatch_event(&ev);
            }
        }
    }
}

fn translate_event(ev: worker_client::WorkerEvent) -> GenerationEvent {
    use worker_client::WorkerEvent as W;
    match ev {
        W::Progress(p) => GenerationEvent::Progress(p),
        W::Done(result) => GenerationEvent::Done(result),
        W::Error(e) => GenerationEvent::Error(e),
        W::Cancelled => GenerationEvent::Cancelled,
    }
}
