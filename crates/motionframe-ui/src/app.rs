//! Shared `MotionFrameApp` — runs identically on desktop and web via the
//! `Platform` trait. Adding a UI feature here propagates to every platform.

use std::path::Path;

use motionframe_engine::io::sequence;
use motionframe_engine::pipeline::atlas_layout::DEFAULT_PADDING_BOUND;
use motionframe_engine::pipeline::output_detents::{
    build_output_count_detents, snap_to_canonical_skip, DetentEntry,
};
use motionframe_engine::pipeline::run::{
    calculate_required_frames, predict_resize_height, EncodeResult, PackMode,
};
use motionframe_engine::pipeline::{GenerateOptions, ImageRgba8, Progress};
use motionframe_engine::preview::pipeline as preview_pipeline;
use motionframe_engine::viz::arrows;

use crate::i18n::{fmt, t, Key, Lang};
use crate::input_panel;
use crate::platform::{EncodedFrame, GenerationEvent, Platform};
use crate::playback::{self, PlaybackState};
use crate::tabs::{self, TabKind};

/// Read dimensions from `bytes` using the format implied by the filename
/// extension. Returns `None` if the extension is unsupported or the header
/// can't be parsed.
fn peek_dimensions(name: &str, bytes: &[u8]) -> Option<(u32, u32)> {
    motionframe_engine::io::peek_dimensions_from_bytes(name, bytes).ok()
}

/// Application state machine.
#[derive(Default)]
pub enum AppState {
    /// No sequence loaded.
    #[default]
    Empty,
    /// Sequence loaded, ready to generate.
    Ready,
    /// Generation running. Stores the state to return to on cancel/error.
    Generating { return_to: Box<Self> },
    /// Generation complete, result available.
    Done,
}

/// Derive a save-prefix string from the first input filename.
fn derive_save_prefix(first_name: &str) -> String {
    if let Some((prefix, _digits, _ext)) = sequence::detect_pattern(first_name) {
        let trimmed = prefix.trim_end_matches(['_', '-', '.', ' ']);
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    Path::new(first_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("output")
        .to_string()
}

/// The full `MotionFrame` application — same widgets, same state machine,
/// across desktop and web. Platform-specific IO and worker management
/// flow through `P: Platform`.
#[allow(clippy::struct_excessive_bools)] // disparate UI flags; not a state-machine candidate
pub struct MotionFrameApp<P: Platform> {
    platform: P,
    state: AppState,
    frames: Vec<EncodedFrame>,
    frame_dims: Option<(u32, u32)>,
    source_label: Option<String>,
    /// Default filename suggested by the save flow.
    default_save_name: String,
    options: GenerateOptions,
    result: Option<EncodeResult>,
    /// Options that produced `result`. Output tabs (Visualization, Preview)
    /// read from this, not `options`, so live sidebar edits don't warp the
    /// already-generated atlas with the wrong grid/encoding before the user
    /// regenerates.
    result_options: Option<GenerateOptions>,
    /// Snapshot of `options` taken at `start_generation`. Promoted to
    /// `result_options` on `Done`; dropped on `Cancelled`/`Error`. Captured
    /// at start (not Done) so that mid-run sidebar edits don't get attributed
    /// to the just-finished result.
    pending_options: Option<GenerateOptions>,
    progress_fraction: f32,
    progress_label: String,
    error_banner: Option<String>,
    current_tab: TabKind,
    zoom: f32,
    color_tex: Option<egui::TextureHandle>,
    motion_tex: Option<egui::TextureHandle>,
    viz_tex: Option<egui::TextureHandle>,
    viz_atlas: Option<ImageRgba8>,
    playback: PlaybackState,
    preview_initialized: bool,
    last_frame_time: Option<f64>,
    preview_textures_dirty: bool,
    licenses_open: bool,
    /// Suspends auto-layout after direct atlas cols/rows edits.
    atlas_layout_manual: bool,
    lang: Lang,
    jp_font_available: bool,
}

impl<P: Platform> MotionFrameApp<P> {
    /// Construct from a Platform impl + persisted options (if any).
    pub fn new(platform: P, cc: &eframe::CreationContext<'_>) -> Self {
        // Keep egui's user zoom at 100%. Use `set_zoom_factor` (absolute) so
        // persisted zoom settings do not carry between launches.
        cc.egui_ctx.set_zoom_factor(1.0);
        let options = cc
            .storage
            .and_then(|s| eframe::get_value::<GenerateOptions>(s, "motionframe.options"))
            .unwrap_or_default();
        let lang = platform.initial_lang();
        let jp_font_available = platform.jp_font_available();
        Self {
            platform,
            state: AppState::default(),
            frames: Vec::new(),
            frame_dims: None,
            source_label: None,
            default_save_name: String::new(),
            options,
            result: None,
            result_options: None,
            pending_options: None,
            progress_fraction: 0.0,
            progress_label: String::new(),
            error_banner: None,
            current_tab: TabKind::default(),
            zoom: 1.0,
            color_tex: None,
            motion_tex: None,
            viz_tex: None,
            viz_atlas: None,
            playback: PlaybackState::default(),
            preview_initialized: false,
            last_frame_time: None,
            preview_textures_dirty: false,
            licenses_open: false,
            atlas_layout_manual: false,
            lang,
            jp_font_available,
        }
    }

    /// Logical frame count for pipeline-shape math.
    /// Atlas mode: `cols × rows`. Sequence mode: number of files.
    const fn effective_frame_count(&self) -> usize {
        match self.options.input_atlas_dims {
            Some((c, r)) => (c * r) as usize,
            None => self.frames.len(),
        }
    }

    /// Frame count after applying the start/end range slice.
    fn frame_count_after_range(&self) -> u32 {
        let total = self.effective_frame_count() as u32;
        if total == 0 {
            return 0;
        }
        let end = if self.options.end_frame == 0 {
            total
        } else {
            self.options.end_frame
        };
        end.saturating_sub(self.options.start_frame).min(total)
    }

    fn clear_texture_cache(&mut self) {
        self.color_tex = None;
        self.motion_tex = None;
        self.viz_tex = None;
        self.viz_atlas = None;
    }

    fn init_preview(&mut self, frame: &eframe::Frame) {
        if self.preview_initialized {
            return;
        }
        if let Some(render_state) = frame.wgpu_render_state() {
            preview_pipeline::init_preview_resources(render_state);
            self.preview_initialized = true;
        }
    }

    fn upload_preview_textures(&self, frame: &eframe::Frame) {
        let Some(ref result) = self.result else {
            return;
        };
        let Some(render_state) = frame.wgpu_render_state() else {
            return;
        };
        preview_pipeline::upload_preview_textures(
            render_state,
            &result.color_atlas,
            &result.motion_atlas,
        );
    }

    /// Accept a freshly-picked sequence and transition Empty → Ready.
    ///
    /// Single-image drop → atlas mode: runs the heuristic to prefill
    /// `input_atlas_dims` and mirrors it to `atlas_dims`. Multi-image drop →
    /// sequence mode: clears `input_atlas_dims`.
    fn accept_picked_frames(&mut self, frames: Vec<EncodedFrame>, label: Option<String>) {
        if frames.is_empty() {
            return;
        }

        let first = &frames[0];
        let Some((w, h)) = peek_dimensions(&first.name, &first.bytes) else {
            self.error_banner = Some(fmt(t(self.lang, Key::ErrCouldNotDecode), &[&first.name]));
            return;
        };

        if frames.len() == 1 {
            // Atlas mode. Decode the source once to run the heuristic; the
            // engine will re-decode at generation time. Cost is one image
            // decode (~30 ms for typical sizes) — the heuristic itself is
            // sub-millisecond.
            let decoded =
                match motionframe_engine::io::decode_image_from_bytes(&first.name, &first.bytes) {
                    Ok(img) => img,
                    Err(e) => {
                        self.error_banner = Some(fmt(
                            t(self.lang, Key::ErrCouldNotDecodeWith),
                            &[&first.name, &e],
                        ));
                        return;
                    }
                };
            let detected = motionframe_engine::io::detect_tile_count(&decoded);
            let dims = detected.or(self.options.input_atlas_dims).unwrap_or((1, 2));
            self.options.input_atlas_dims = Some(dims);
            // Output atlas dims are auto-derived by `recompute_atlas_layout`.

            let display = label.unwrap_or_else(|| first.name.clone());
            self.source_label = Some(format!(
                "{display} — {}×{} ({} tiles)",
                dims.0,
                dims.1,
                dims.0 * dims.1
            ));
        } else {
            // Sequence mode.
            self.options.input_atlas_dims = None;
            self.source_label = label;
            self.options.output_name_basename = derive_save_prefix(&frames[0].name);
        }

        self.default_save_name = derive_save_prefix(&frames[0].name);
        self.options.start_frame = 0;
        self.options.end_frame = frames.len() as u32;
        self.frames = frames;
        self.frame_dims = Some((w, h));
        self.result = None;
        self.result_options = None;
        self.clear_texture_cache();
        self.state = AppState::Ready;
        self.error_banner = None;
        self.atlas_layout_manual = false;
    }

    fn start_generation(&mut self, ctx: &egui::Context) {
        if self.frames.is_empty() {
            return;
        }
        let opts = self.options.clone();
        let frames = self.frames.clone();
        // Stash the exact opts handed to the worker. On Done we promote this
        // to `result_options` so the output tabs render against what was
        // actually generated, not whatever the sidebar shows now.
        self.pending_options = Some(opts.clone());
        self.platform.start_generation(frames, opts, ctx);
        if let Some(msg) = self.platform.take_generation_start_error() {
            self.pending_options = None;
            self.error_banner = Some(msg);
            return;
        }

        let return_to = match self.state {
            AppState::Done => AppState::Done,
            _ => AppState::Ready,
        };
        self.state = AppState::Generating {
            return_to: Box::new(return_to),
        };
        self.progress_fraction = 0.0;
        self.progress_label = "Starting…".to_string();
    }

    fn poll_worker(&mut self) {
        while let Some(ev) = self.platform.take_generation_event() {
            match ev {
                GenerationEvent::Progress(p) => self.apply_progress(p),
                GenerationEvent::Done(encode_result) => {
                    self.playback.frame_count = encode_result.total_frames;
                    self.playback.set_strength(encode_result.strength as f32);
                    self.playback.time = 0.0;
                    self.playback.playing = true;
                    self.result = Some(encode_result);
                    self.result_options = self.pending_options.take();
                    self.clear_texture_cache();
                    self.preview_textures_dirty = true;
                    self.state = AppState::Done;
                }
                GenerationEvent::Cancelled => {
                    self.pending_options = None;
                    let return_to = match std::mem::take(&mut self.state) {
                        AppState::Generating { return_to } => *return_to,
                        other => other,
                    };
                    self.state = return_to;
                }
                GenerationEvent::Error(msg) => {
                    self.pending_options = None;
                    let return_to = match std::mem::take(&mut self.state) {
                        AppState::Generating { return_to } => *return_to,
                        other => other,
                    };
                    self.error_banner = Some(msg);
                    self.state = return_to;
                }
            }
        }
    }

    fn apply_progress(&mut self, p: Progress) {
        match p {
            Progress::Loading { current, total } => {
                self.progress_fraction = if total > 0 {
                    current as f32 / total as f32
                } else {
                    0.0
                };
                self.progress_label = fmt(t(self.lang, Key::LoadingProgress), &[&current, &total]);
            }
            Progress::Stage { name, fraction } => {
                self.progress_fraction = fraction;
                self.progress_label = name;
            }
            Progress::Done => {
                self.progress_fraction = 1.0;
                self.progress_label = "Done".to_string();
            }
        }
    }

    fn cancel_generation(&mut self) {
        self.platform.cancel_generation();
    }

    fn recompute_atlas_layout(&mut self) {
        if self.atlas_layout_manual {
            return;
        }
        let n_input = self.effective_frame_count();
        if n_input < 2 {
            return;
        }
        let n_output = if self.options.trim_tail_for_exact_output_count {
            self.options.output_frames
        } else {
            calculate_required_frames(n_input, self.options.frame_skip) as u32
        };
        let Some((src_w, src_h)) = self.frame_dims else {
            return;
        };
        let (iw, ih) = match self.options.input_atlas_dims {
            Some((c, r)) if c > 0 && r > 0 => (src_w / c, src_h / r),
            _ => (src_w, src_h),
        };
        if iw == 0 || ih == 0 {
            return;
        }
        let input_aspect = iw as f64 / ih as f64;
        let max_dim = self.options.output_atlas_max_dim;
        if let Some(layout) = motionframe_engine::pipeline::atlas_layout::pick_layout(
            n_output,
            input_aspect,
            self.options.atlas_resolution,
            max_dim,
            DEFAULT_PADDING_BOUND,
        ) {
            self.options.atlas_dims = (layout.cols, layout.rows);
            self.options.tile_pixel_width = layout.tile_width;
        }
    }

    fn sync_output_detent(&mut self, layouts: &[DetentEntry], n_input: u32) {
        if layouts.is_empty() {
            return;
        }

        if self.options.trim_tail_for_exact_output_count {
            let entry = layouts
                .iter()
                .find(|entry| entry.output_count == self.options.output_frames)
                .unwrap_or_else(|| layouts.last().expect("non-empty layouts"));
            self.options.output_frames = entry.output_count;
            self.options.frame_skip = entry.frame_skip;
            return;
        }

        self.options.frame_skip = snap_to_canonical_skip(layouts, n_input, self.options.frame_skip);
        self.options.output_frames =
            calculate_required_frames(n_input as usize, self.options.frame_skip) as u32;
    }

    fn output_dims(&self) -> Option<input_panel::OutputDims> {
        let (sw, sh) = self.frame_dims?;
        let (iw, ih) = match self.options.input_atlas_dims {
            Some((c, r)) if c > 0 && r > 0 => (sw / c, sh / r),
            _ => (sw, sh),
        };
        if iw == 0 || ih == 0 {
            return None;
        }
        let (cols, rows) = self.options.atlas_dims;
        if cols == 0 || rows == 0 {
            return None;
        }

        let frame_width = self.options.tile_pixel_width.max(1);
        let extrude = self.options.extrude.min((frame_width - 1) / 2);
        let valid_width = frame_width - extrude * 2;
        let valid_height = predict_resize_height(ih, iw, valid_width);
        let frame_height = valid_height + extrude * 2;

        let color_w = cols * frame_width;
        let color_h = rows * frame_height;
        let (mv_atlas_w, mv_atlas_h) = if self.options.halve_motion_vector {
            let new_w = ((frame_width / 2).max(1)) * cols;
            let new_h =
                ((f64::from(color_h) * f64::from(new_w) / f64::from(color_w)).ceil() as u32).max(1);
            (new_w, new_h)
        } else {
            (color_w, color_h)
        };
        let (mv_w, mv_h) = if self.options.stagger_pack {
            let tile_h = mv_atlas_h / rows;
            let num_tiles = cols * rows;
            let output_tile_rows = num_tiles.div_ceil(2).div_ceil(cols);
            (mv_atlas_w, output_tile_rows * tile_h)
        } else {
            (mv_atlas_w, mv_atlas_h)
        };

        Some(input_panel::OutputDims {
            color: (color_w, color_h),
            motion: (mv_w, mv_h),
        })
    }

    fn dims_fit_gpu(&self) -> bool {
        let max_dim = self.options.output_atlas_max_dim;
        self.output_dims().is_none_or(|d| !d.exceeds_limit(max_dim))
    }

    fn selected_output_fits_atlas(&self) -> bool {
        let n_input = self.effective_frame_count();
        if n_input == 0 {
            return false;
        }
        let output_count = if self.options.trim_tail_for_exact_output_count {
            (self.options.output_frames as usize)
                .min(calculate_required_frames(n_input, self.options.frame_skip))
        } else {
            calculate_required_frames(n_input, self.options.frame_skip)
        };
        let (cols, rows) = self.options.atlas_dims;
        output_count <= (cols * rows) as usize
    }

    fn trigger_save(&mut self) {
        let Some(ref result) = self.result else {
            return;
        };
        // Build metadata JSON in-memory.
        let json = match motionframe_engine::io::metadata::build_metadata_json(result) {
            Ok(s) => s,
            Err(e) => {
                self.error_banner = Some(fmt(t(self.lang, Key::ErrMetadataSerialize), &[&e]));
                return;
            }
        };
        if let Err(e) = self.platform.save_outputs(
            &self.default_save_name,
            &result.color_atlas,
            &result.motion_atlas,
            &json,
        ) {
            self.error_banner = Some(fmt(t(self.lang, Key::ErrSaveOutputs), &[&e]));
        }
    }
}

// --- eframe::App ---
impl<P: Platform> eframe::App for MotionFrameApp<P> {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, "motionframe.options", &self.options);
        self.platform.save_app_state(storage, self.lang);
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        self.init_preview(frame);
        self.do_update(ui, frame);
    }
}

// --- UI rendering ---
impl<P: Platform> MotionFrameApp<P> {
    // allow(too_many_lines): main event loop orchestrates UI layout, platform events,
    //   worker polling, and state transitions — splitting would fragment the update flow
    #[allow(clippy::too_many_lines)]
    fn do_update(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        // Clone the Context (cheap — Arc-internal) and shadow with a borrow.
        // Keeping `ctx` as `&Context` lets us hand it to `start_generation`
        // etc. while still passing `ui` mutably to `draw_sidebar` and friends
        // without holding a `ui.ctx()` borrow across the call.
        let ctx_owned = ui.ctx().clone();
        let ctx = &ctx_owned;
        // Signal "ready" once so the platform can dismiss a loading overlay
        // and install any DOM handlers it owns (web folder-aware drop handler).
        self.platform.signal_ready(ctx);

        // Drain platform events.
        if let Some(picks) = self.platform.take_folder_pick() {
            self.accept_picked_frames(picks, None);
        }

        // Handle drag-and-drop (window-wide).
        let dropped: Vec<egui::DroppedFile> = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped.is_empty() {
            let frames = self.platform.handle_dropped_files(dropped);
            if !frames.is_empty() {
                self.accept_picked_frames(frames, None);
            }
        }

        if matches!(self.state, AppState::Generating { .. }) {
            self.poll_worker();
            ctx.request_repaint();
        }

        // Upload preview textures if dirty.
        if self.preview_textures_dirty {
            self.upload_preview_textures(frame);
            self.preview_textures_dirty = false;
        }

        // Advance playback time.
        let now_secs = ctx.input(|i| i.time);
        if let Some(last) = self.last_frame_time {
            let dt = now_secs - last;
            self.playback.advance(dt);
        }
        self.last_frame_time = Some(now_secs);
        if self.playback.playing {
            ctx.request_repaint();
        }

        let should_generate = ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::G))
            && matches!(self.state, AppState::Ready | AppState::Done)
            && !self.frames.is_empty()
            && self.selected_output_fits_atlas()
            && self.dims_fit_gpu();
        let should_cancel = ctx.input(|i| i.key_pressed(egui::Key::Escape))
            && matches!(self.state, AppState::Generating { .. });
        let should_save = ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::S))
            && matches!(self.state, AppState::Done);

        if should_cancel {
            self.cancel_generation();
        }

        self.draw_sidebar(ui);
        self.draw_status_bar(ui);
        self.draw_central(ui);
        self.draw_licenses_window(ctx);

        if should_generate && !matches!(self.state, AppState::Generating { .. }) {
            self.start_generation(ctx);
        }
        if should_save {
            self.trigger_save();
        }
    }

    #[allow(clippy::too_many_lines)]
    fn draw_sidebar(&mut self, host_ui: &mut egui::Ui) {
        let ctx = host_ui.ctx().clone();
        let mut do_browse = false;
        let mut do_generate = false;

        egui::Panel::left("options_panel")
            .exact_size(340.0)
            .show_inside(host_ui, |ui| {
                let full_rect = ui.available_rect_before_wrap();
                let footer_height = 72.0_f32.min(full_rect.height());
                let generate_height = 48.0_f32.min((full_rect.height() - footer_height).max(0.0));
                let footer_rect = egui::Rect::from_min_max(
                    egui::pos2(full_rect.min.x, full_rect.max.y - footer_height),
                    full_rect.max,
                );
                let generate_rect = egui::Rect::from_min_max(
                    egui::pos2(
                        full_rect.min.x,
                        (footer_rect.min.y - generate_height).max(full_rect.min.y),
                    ),
                    egui::pos2(full_rect.max.x, footer_rect.min.y),
                );
                let body_rect = egui::Rect::from_min_max(
                    full_rect.min,
                    egui::pos2(full_rect.max.x, generate_rect.min.y),
                );
                let mut body_ui = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(body_rect)
                        .layout(egui::Layout::top_down(egui::Align::LEFT)),
                );
                body_ui.set_clip_rect(body_rect);
                let mut dismiss_banner = false;
                if let Some(ref banner) = self.error_banner {
                    egui::Frame::NONE
                        .fill(egui::Color32::from_rgb(80, 20, 20))
                        .inner_margin(egui::Margin::same(4))
                        .show(&mut body_ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.colored_label(
                                    egui::Color32::from_rgb(255, 200, 200),
                                    banner.as_str(),
                                );
                                if ui.small_button(t(self.lang, Key::ClearSelection)).clicked() {
                                    dismiss_banner = true;
                                }
                            });
                        });
                }
                if dismiss_banner {
                    self.error_banner = None;
                }

                if matches!(self.state, AppState::Empty) {
                    egui::Frame::NONE
                        .stroke(egui::Stroke::new(1.0, egui::Color32::from_gray(128)))
                        .outer_margin(egui::Margin {
                            left: 0,
                            right: 0,
                            top: 8,
                            bottom: 6,
                        })
                        .inner_margin(egui::Margin {
                            left: 8,
                            right: 8,
                            top: 8,
                            bottom: 8,
                        })
                        .show(&mut body_ui, |ui| {
                            ui.label(t(self.lang, Key::DropHereOrBrowse));
                            let browse = egui::Button::new(t(self.lang, Key::BrowseDots)).min_size(
                                egui::vec2(ui.available_width(), ui.spacing().interact_size.y),
                            );
                            if ui.add(browse).clicked() {
                                do_browse = true;
                            }
                        });
                } else {
                    if let Some(ref label) = self.source_label {
                        if self.options.input_atlas_dims.is_some() {
                            // Atlas-mode label already encodes tile count.
                            body_ui.label(label);
                        } else {
                            let count_str = fmt(
                                t(self.lang, Key::FramesCount),
                                &[&self.effective_frame_count()],
                            );
                            body_ui.label(format!("{label} — {count_str}"));
                        }
                    } else {
                        body_ui.label(fmt(
                            t(self.lang, Key::FramesCount),
                            &[&self.effective_frame_count()],
                        ));
                    }
                    let browse = egui::Button::new(t(self.lang, Key::BrowseEllipsis))
                        .min_size(body_ui.spacing().interact_size);
                    if body_ui.add(browse).clicked() {
                        do_browse = true;
                    }
                }

                body_ui.separator();
                self.recompute_atlas_layout();
                let output_dims = self.output_dims();
                let n_input = self.effective_frame_count() as u32;
                let n_after_range = self.frame_count_after_range();
                let canonical_layouts = build_output_count_detents(
                    n_input,
                    self.options.trim_tail_for_exact_output_count,
                );
                self.sync_output_detent(&canonical_layouts, n_input);
                let trim_tail_before = self.options.trim_tail_for_exact_output_count;
                let sequence_loaded =
                    !self.frames.is_empty() && self.options.input_atlas_dims.is_none();
                input_panel::show_options(
                    &mut body_ui,
                    &mut self.options,
                    &mut self.atlas_layout_manual,
                    output_dims,
                    n_input,
                    n_after_range,
                    &canonical_layouts,
                    self.lang,
                    sequence_loaded,
                );
                if trim_tail_before != self.options.trim_tail_for_exact_output_count {
                    self.atlas_layout_manual = false;
                    let canonical_layouts = build_output_count_detents(
                        n_input,
                        self.options.trim_tail_for_exact_output_count,
                    );
                    self.sync_output_detent(&canonical_layouts, n_input);
                    self.recompute_atlas_layout();
                }

                let can_generate = !self.frames.is_empty()
                    && matches!(self.state, AppState::Ready | AppState::Done)
                    && self.selected_output_fits_atlas()
                    && self.dims_fit_gpu();

                let mut generate_ui = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(generate_rect)
                        .layout(egui::Layout::top_down(egui::Align::LEFT)),
                );
                generate_ui.set_clip_rect(generate_rect);
                generate_ui.separator();
                let btn = egui::Button::new(t(self.lang, Key::Generate))
                    .min_size(egui::vec2(generate_ui.available_width(), 32.0));
                if generate_ui.add_enabled(can_generate, btn).clicked() {
                    do_generate = true;
                }

                let mut footer_ui = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(footer_rect)
                        .layout(egui::Layout::bottom_up(egui::Align::LEFT)),
                );
                footer_ui.set_clip_rect(footer_rect);
                footer_ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    ui.add_space(8.0);
                    if self.platform.third_party_licenses().is_some()
                        && ui.button(t(self.lang, Key::Licenses)).clicked()
                    {
                        self.licenses_open = true;
                    }
                    input_panel::show_language_picker(ui, &mut self.lang, self.jp_font_available);
                });
            });

        if do_browse {
            self.platform.start_folder_pick(&ctx);
        }
        if do_generate {
            self.start_generation(&ctx);
        }
    }

    fn draw_licenses_window(&mut self, ctx: &egui::Context) {
        if !self.licenses_open {
            return;
        }
        let Some(licenses) = self.platform.third_party_licenses() else {
            self.licenses_open = false;
            return;
        };
        let mut open = self.licenses_open;
        egui::Window::new(t(self.lang, Key::LicensesWindowTitle))
            .open(&mut open)
            .default_width(720.0)
            .default_height(560.0)
            .vscroll(false)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
                        let mut text = licenses;
                        ui.add(
                            egui::TextEdit::multiline(&mut text)
                                .font(egui::TextStyle::Monospace)
                                .desired_width(f32::INFINITY)
                                .desired_rows(28)
                                .interactive(false),
                        );
                    });
            });
        self.licenses_open = open;
    }

    fn draw_status_bar(&mut self, host_ui: &mut egui::Ui) {
        let mut do_cancel = false;
        let mut do_save = false;
        egui::Panel::bottom("status_bar")
            .show_separator_line(false)
            .show_inside(host_ui, |ui| {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if matches!(self.state, AppState::Generating { .. }) {
                        ui.add(egui::ProgressBar::new(self.progress_fraction).show_percentage());
                        ui.label(&self.progress_label);
                        if ui.button(t(self.lang, Key::Cancel)).clicked() {
                            do_cancel = true;
                        }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let save_enabled = matches!(self.state, AppState::Done);
                        let save_btn = egui::Button::new(
                            egui::RichText::new(t(self.lang, Key::Save))
                                .size(18.0)
                                .strong(),
                        )
                        .min_size(egui::vec2(140.0, 40.0));
                        if ui.add_enabled(save_enabled, save_btn).clicked() {
                            do_save = true;
                        }
                    });
                });
                ui.add_space(8.0);
            });
        if do_cancel {
            self.cancel_generation();
        }
        if do_save {
            self.trigger_save();
        }
    }

    fn draw_central(&mut self, host_ui: &mut egui::Ui) {
        if matches!(self.state, AppState::Done) {
            let ctx = host_ui.ctx().clone();
            self.ensure_textures(&ctx);
        }

        let mut tab_clone = self.current_tab;
        let lang = self.lang;
        egui::CentralPanel::default().show_inside(host_ui, |ui| match self.state {
            AppState::Empty => tabs::draw_empty_state(ui, lang),
            AppState::Ready => {
                tabs::tab_bar(ui, &mut tab_clone, lang);
                tabs::draw_ready_placeholder(ui, lang);
            }
            AppState::Generating { .. } | AppState::Done => {
                tabs::tab_bar(ui, &mut tab_clone, lang);
                self.current_tab = tab_clone;
                self.draw_tab_content(ui);
                tab_clone = self.current_tab;
            }
        });
        self.current_tab = tab_clone;
    }

    fn ensure_textures(&mut self, ctx: &egui::Context) {
        if self.result.is_none() {
            return;
        }
        if self.color_tex.is_none() {
            if let Some(ref result) = self.result {
                self.color_tex = Some(tabs::upload_texture(
                    ctx,
                    "color_atlas",
                    &result.color_atlas,
                ));
            }
        }
        if self.motion_tex.is_none() {
            if let Some(ref result) = self.result {
                self.motion_tex = Some(tabs::upload_texture(
                    ctx,
                    "motion_atlas",
                    &result.motion_atlas,
                ));
            }
        }
        if self.current_tab == TabKind::Visualization && self.viz_tex.is_none() {
            // Render the viz against the options that produced `result`, not
            // the current sidebar values — otherwise a live atlas-grid edit
            // re-tiles the existing flow data with the wrong cell size.
            // `clear_texture_cache` on the next Done refreshes this.
            if let (Some(ref result), Some(ref opts)) = (&self.result, &self.result_options) {
                let (_atlas_cols, atlas_rows) = opts.atlas_dims;
                let frame_width = opts.tile_pixel_width;
                // Guard against a degenerate persisted/zero atlas grid: a 0 row
                // count would divide-by-zero (panic) here.
                let frame_height = result.atlas_height / atlas_rows.max(1);
                let viz_img = arrows::draw_optical_flow_atlas(
                    &result.flows,
                    &result.color_atlas,
                    result.total_frames,
                    frame_width,
                    frame_height,
                    opts,
                );
                let tex = tabs::upload_texture(ctx, "viz_atlas", &viz_img);
                self.viz_tex = Some(tex);
                self.viz_atlas = Some(viz_img);
            }
        }
    }

    fn draw_tab_content(&mut self, ui: &mut egui::Ui) {
        match self.current_tab {
            TabKind::Color => {
                if let Some(ref tex) = self.color_tex {
                    tabs::draw_image_tab(ui, tex, &mut self.zoom, self.lang);
                } else {
                    tabs::draw_ready_placeholder(ui, self.lang);
                }
            }
            TabKind::Motion => {
                if let Some(ref tex) = self.motion_tex {
                    tabs::draw_image_tab(ui, tex, &mut self.zoom, self.lang);
                } else {
                    tabs::draw_ready_placeholder(ui, self.lang);
                }
            }
            TabKind::Visualization => {
                if let Some(ref tex) = self.viz_tex {
                    tabs::draw_image_tab(ui, tex, &mut self.zoom, self.lang);
                } else {
                    tabs::draw_ready_placeholder(ui, self.lang);
                }
            }
            TabKind::Preview => {
                self.draw_preview_tab(ui);
            }
        }
    }

    fn draw_preview_tab(&mut self, ui: &mut egui::Ui) {
        // Reserve vertical space for playback controls below the preview.
        // Without this, on tall layouts the square preview eats all height
        // and the transport ends up clipped behind the status bar.
        const PLAYBACK_RESERVED_HEIGHT: f32 = 110.0;

        // Read the options that produced `result`, not `self.options`. The
        // sidebar values may have drifted since the last generation, and
        // feeding a mismatched atlas grid or encoding to the shader warps
        // the existing atlas with the wrong cell layout.
        let (Some(result), Some(opts)) = (self.result.as_ref(), self.result_options.as_ref())
        else {
            tabs::draw_ready_placeholder(ui, self.lang);
            return;
        };
        let stagger_pack = match result.pack_mode {
            PackMode::Staggered => 1u32,
            PackMode::Normal => 0u32,
        };
        let mv_encoding = opts.motion_vector_encoding.as_u32();

        let available = ui.available_size();
        let max_h = (available.y - PLAYBACK_RESERVED_HEIGHT).max(64.0);
        let side = available.x.min(max_h).min(768.0);
        let (rect, _response) =
            ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::hover());

        let uniforms = preview_pipeline::Uniforms {
            atlas_grid: [opts.atlas_dims.0, opts.atlas_dims.1],
            frame_count: self.playback.frame_count,
            motion_strength: self.playback.strength(),
            time: self.playback.time,
            stagger_pack,
            mv_encoding,
            bg_mode: self.playback.bg.as_u32(),
            premultiplied_alpha: u32::from(result.premultiplied_alpha),
            blend_mode: self.playback.blend_mode.as_u32(),
            _pad2: 0,
            _pad3: 0,
        };

        preview_pipeline::paint_preview(ui, rect, uniforms);

        ui.add_space(4.0);
        let is_playing = playback::draw_playback_controls(ui, &mut self.playback, self.lang);
        if is_playing {
            ui.ctx().request_repaint();
        }
    }
}
