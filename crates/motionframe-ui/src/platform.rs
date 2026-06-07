//! Platform abstraction for `MotionFrameApp`.
//!
//! Both desktop and web shells implement `Platform`. Everything in the
//! shared app code goes through this trait for IO and worker management
//! so the same `MotionFrameApp<P>` runs unchanged on both targets.

use motionframe_engine::pipeline::run::EncodeResult;
use motionframe_engine::pipeline::{GenerateOptions, ImageRgba8, Progress};

use crate::i18n::Lang;

/// One frame as delivered by the platform's folder picker.
/// Encoded bytes (PNG or TGA); decoded by the engine on demand.
#[derive(Clone)]
pub struct EncodedFrame {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// Events surfaced by a running generation. Polled each frame by the app.
pub enum GenerationEvent {
    Progress(Progress),
    Done(EncodeResult),
    Error(String),
    Cancelled,
}

/// Platform-specific IO and worker management.
///
/// Desktop impl uses `rfd` + `std::thread`. Web impl uses an HTML file picker
/// bridge + a Web Worker over `postMessage`.
pub trait Platform: 'static {
    /// Initial UI language chosen by the platform.
    fn initial_lang(&self) -> Lang {
        Lang::En
    }

    /// Whether Japanese can be rendered on this platform.
    fn jp_font_available(&self) -> bool {
        false
    }

    /// Persist platform-owned app state.
    fn save_app_state(&mut self, _storage: &mut dyn eframe::Storage, _lang: Lang) {}

    /// Optional license text shown by the shared UI.
    fn third_party_licenses(&self) -> Option<&'static str> {
        None
    }

    /// Trigger a folder picker. May be async; result becomes available
    /// later via [`take_folder_pick`].
    fn start_folder_pick(&mut self, ctx: &egui::Context);

    /// Take the result of a folder pick. Returns `None` when no pick is
    /// pending or it hasn't completed yet. Returns `Some(empty)` if the
    /// user cancelled. Implementations must return at most one result per
    /// completed pick.
    fn take_folder_pick(&mut self) -> Option<Vec<EncodedFrame>>;

    /// Trigger a save flow for the given outputs.
    /// - Desktop: shows a save dialog, then writes 3 files.
    /// - Web: triggers downloads with default names.
    ///
    /// Returns `Err` with a human-readable message if saving failed, so the UI
    /// can surface it instead of silently dropping the output. `Ok(())` also
    /// covers the user cancelling the dialog (nothing to report).
    fn save_outputs(
        &mut self,
        prefix: &str,
        color_atlas: &ImageRgba8,
        motion_atlas: &ImageRgba8,
        metadata_json: &str,
    ) -> Result<(), String>;

    /// Begin a generation run with the given frames + options.
    /// Events are produced via [`take_generation_event`] as the worker runs.
    fn start_generation(
        &mut self,
        frames: Vec<EncodedFrame>,
        options: GenerateOptions,
        ctx: &egui::Context,
    );

    /// Take a synchronous start-generation failure, if one happened.
    fn take_generation_start_error(&mut self) -> Option<String> {
        None
    }

    /// Take the next generation event. `None` if no events pending.
    fn take_generation_event(&mut self) -> Option<GenerationEvent>;

    /// Signal cancel to the running generation. Idempotent.
    fn cancel_generation(&mut self);

    /// Convert a list of `egui::DroppedFile`s into encoded frames.
    /// Desktop reads from `dropped.path`; web reads from `dropped.bytes`
    /// (eframe's web target populates bytes for dropped files).
    /// Returns the frames to load into the app, or empty if nothing matched.
    fn handle_dropped_files(&mut self, dropped: Vec<egui::DroppedFile>) -> Vec<EncodedFrame>;

    /// Optional per-update hook invoked once at app start so the platform can
    /// fire a "ready" signal (e.g., dispatch a DOM event to hide a loading
    /// spinner) and install any DOM-level handlers it owns (e.g., a folder-aware
    /// drop handler on web). The platform is responsible for one-shot guarding.
    /// Default no-op for non-web platforms.
    fn signal_ready(&mut self, _ctx: &egui::Context) {}

    /// Whether the wgpu-based preview pipeline is supported on this platform.
    fn supports_preview(&self) -> bool {
        true
    }
}
