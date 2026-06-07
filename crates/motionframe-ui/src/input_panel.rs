//! Options panel for `MotionFrame` GUI.
//!
//! Widget groups: Input (atlas mode only), Atlas, Motion, Color.
//! Every visible field of `GenerateOptions` maps to exactly one widget.

use motionframe_engine::pipeline::output_detents::DetentEntry;
use motionframe_engine::pipeline::{GenerateOptions, Interpolation, MotionVectorEncoding};

use crate::i18n::{fmt, t, Key, Lang};

/// Maximum supported atlas dimension on either axis. Matches the WebGPU
/// baseline `max_texture_dimension_2d`; exceeding it makes wgpu refuse to
/// allocate the preview texture.
pub const MAX_TEXTURE_DIM: u32 = 8192;

/// Predicted output atlas pixel dimensions, displayed in the Atlas group.
#[derive(Debug, Clone, Copy)]
pub struct OutputDims {
    pub color: (u32, u32),
    pub motion: (u32, u32),
}

impl OutputDims {
    /// `true` when any axis of either atlas exceeds `max_dim`.
    #[must_use]
    pub const fn exceeds_limit(self, max_dim: u32) -> bool {
        let (cw, ch) = self.color;
        let (mw, mh) = self.motion;
        cw > max_dim || ch > max_dim || mw > max_dim || mh > max_dim
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OutputSummary {
    output_count: usize,
    frame_skip: u32,
    ignored_tail_frames: Option<u32>,
    wasted_tiles: Option<u32>,
}

impl OutputSummary {
    #[must_use]
    fn new(
        output_count: usize,
        frame_skip: u32,
        ignored_tail_frames: u32,
        atlas_dims: (u32, u32),
    ) -> Self {
        let tile_count = atlas_dims.0.saturating_mul(atlas_dims.1);
        let wasted_tiles = tile_count.saturating_sub(output_count as u32);
        Self {
            output_count,
            frame_skip,
            ignored_tail_frames: (ignored_tail_frames > 0).then_some(ignored_tail_frames),
            wasted_tiles: (wasted_tiles > 0).then_some(wasted_tiles),
        }
    }

    #[must_use]
    const fn without_wasted_tiles(self) -> Self {
        Self {
            wasted_tiles: None,
            ..self
        }
    }
}

/// Display the full options sidebar.
///
/// `output_dims` is the predicted pixel size of the color and motion atlases
/// for the current options + source. `None` when no source is loaded.
///
/// `atlas_layout_manual` is set to `true` when the user direct-edits the
/// cols/rows `DragValues`. The output-count slider clears it back to `false`.
///
/// `n_input` is the effective input frame count (post atlas-input slicing).
/// `canonical_layouts` is the sorted-by-count detent set for that `n_input`.
#[allow(clippy::too_many_arguments)]
pub fn show_options(
    ui: &mut egui::Ui,
    options: &mut GenerateOptions,
    atlas_layout_manual: &mut bool,
    output_dims: Option<OutputDims>,
    n_input: u32,
    canonical_layouts: &[DetentEntry],
    lang: Lang,
) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        show_input_group(ui, options, lang);
        if options.input_atlas_dims.is_some() {
            ui.separator();
        }
        show_atlas_group(
            ui,
            options,
            atlas_layout_manual,
            output_dims,
            n_input,
            canonical_layouts,
            lang,
        );
        ui.separator();
        show_motion_group(ui, options, lang);
        ui.separator();
        show_color_group(ui, options, lang);
    });
}

/// Input group (visible only in atlas mode — when the user dropped a
/// single image and we sliced it into tiles).
fn show_input_group(ui: &mut egui::Ui, options: &mut GenerateOptions, lang: Lang) {
    let Some((mut cols, mut rows)) = options.input_atlas_dims else {
        return;
    };
    ui.heading(t(lang, Key::InputHeading));

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::InputTilesPerRow));
        ui.add(egui::DragValue::new(&mut cols).range(1..=256))
            .on_hover_text(t(lang, Key::InputTilesHover));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::InputTilesPerColumn));
        ui.add(egui::DragValue::new(&mut rows).range(1..=256))
            .on_hover_text(t(lang, Key::InputTilesHover));
    });

    options.input_atlas_dims = Some((cols, rows));
}

/// Atlas group — output grid, tile size, frame-skip slider, and shared
/// resampling options. Always visible.
fn show_atlas_group(
    ui: &mut egui::Ui,
    options: &mut GenerateOptions,
    atlas_layout_manual: &mut bool,
    output_dims: Option<OutputDims>,
    n_input: u32,
    canonical_layouts: &[DetentEntry],
    lang: Lang,
) {
    ui.heading(t(lang, Key::AtlasHeading));

    show_frame_skip_slider(
        ui,
        options,
        atlas_layout_manual,
        n_input,
        canonical_layouts,
        lang,
    );
    let trim_resp = ui.checkbox(
        &mut options.trim_tail_for_exact_output_count,
        t(lang, Key::TrimTailForExactOutputCount),
    );
    trim_resp.on_hover_text(t(lang, Key::TrimTailForExactOutputCountHover));

    // --- Tiles per row / column (auto-filled, editable, lock badge) ---
    let auto_label = t(lang, Key::LockBadgeAuto);
    let manual_label = t(lang, Key::LockBadgeManual);
    let lock_badge = |is_manual: bool| if is_manual { manual_label } else { auto_label };

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::TilesPerRow));
        let resp = ui.add(egui::DragValue::new(&mut options.atlas_dims.0).range(1..=256));
        if resp.changed() {
            *atlas_layout_manual = true;
        }
        ui.label(lock_badge(*atlas_layout_manual));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::TilesPerColumn));
        let resp = ui.add(egui::DragValue::new(&mut options.atlas_dims.1).range(1..=256));
        if resp.changed() {
            *atlas_layout_manual = true;
        }
        ui.label(lock_badge(*atlas_layout_manual));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::TilePixelWidth));
        ui.add(egui::DragValue::new(&mut options.tile_pixel_width).range(1..=4096))
            .on_hover_text(t(lang, Key::TilePixelWidthHover));
    });

    // --- Max texture dim (POT ComboBox) ---
    ui.horizontal(|ui| {
        ui.label(t(lang, Key::MaxTextureDim));
        let label = format!("{}", options.output_atlas_max_dim);
        egui::ComboBox::from_id_salt("max_atlas_dim")
            .selected_text(label)
            .show_ui(ui, |ui| {
                for v in [1024u32, 2048, 4096, 8192] {
                    ui.selectable_value(&mut options.output_atlas_max_dim, v, v.to_string());
                }
            })
            .response
            .on_hover_text(t(lang, Key::MaxTextureDimHover));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::ResizeAlgorithm));
        let alg_label = match options.resize_algorithm {
            Interpolation::Cubic => t(lang, Key::ResizeCubic),
            Interpolation::Linear => t(lang, Key::ResizeLinear),
            Interpolation::Lanczos => t(lang, Key::ResizeLanczos),
            Interpolation::Nearest => t(lang, Key::ResizeNearest),
        };
        egui::ComboBox::from_id_salt("resize_alg")
            .selected_text(alg_label)
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut options.resize_algorithm,
                    Interpolation::Cubic,
                    t(lang, Key::ResizeCubic),
                );
                ui.selectable_value(
                    &mut options.resize_algorithm,
                    Interpolation::Linear,
                    t(lang, Key::ResizeLinear),
                );
                ui.selectable_value(
                    &mut options.resize_algorithm,
                    Interpolation::Lanczos,
                    t(lang, Key::ResizeLanczos),
                );
                ui.selectable_value(
                    &mut options.resize_algorithm,
                    Interpolation::Nearest,
                    t(lang, Key::ResizeNearest),
                );
            })
            .response
            .on_hover_text(t(lang, Key::ResizeAlgorithmHover));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::Extrude));
        ui.add(egui::DragValue::new(&mut options.extrude).range(0..=8))
            .on_hover_text(t(lang, Key::ExtrudeHover));
    });

    show_predicted_dims(ui, output_dims, options.output_atlas_max_dim, lang);
}

/// Output-count slider. Slider value is a detent index into
/// `canonical_layouts`; the readout shows the corresponding output count
/// and `frame_skip`.
fn show_frame_skip_slider(
    ui: &mut egui::Ui,
    options: &mut GenerateOptions,
    atlas_layout_manual: &mut bool,
    n_input: u32,
    canonical_layouts: &[DetentEntry],
    lang: Lang,
) {
    if canonical_layouts.len() < 2 {
        if n_input > 0 {
            let count = motionframe_engine::pipeline::run::calculate_required_frames(
                n_input as usize,
                options.frame_skip,
            );
            show_output_summary(
                ui,
                OutputSummary::new(count, options.frame_skip, 0, options.atlas_dims),
                lang,
            );
        }
        // N < 3: at most one canonical count; no meaningful choice.
        // Render a disabled placeholder so the layout doesn't shift on
        // source load.
        let mut dummy = 0u32;
        ui.horizontal(|ui| {
            ui.label(t(lang, Key::OutputFrames));
            ui.add_enabled(
                false,
                egui::Slider::new(&mut dummy, 0..=0)
                    .integer()
                    .show_value(false),
            );
        });
        ui.label(t(lang, Key::LoadMoreFramesHint));
        return;
    }

    // Slider value = detent index. Each canonical position gets an equal
    // slice of the slider track, regardless of where it lands in count-space.
    let current_count = if options.trim_tail_for_exact_output_count {
        options.output_frames
    } else {
        motionframe_engine::pipeline::run::calculate_required_frames(
            n_input as usize,
            options.frame_skip,
        ) as u32
    };
    let mut idx = canonical_layouts
        .iter()
        .position(|e| e.output_count == current_count)
        .unwrap_or(0) as u32;
    let max_idx = (canonical_layouts.len() - 1) as u32;

    let available_width = ui.available_width();
    let spacing_y = ui.spacing().item_spacing.y;
    let heading_height = ui.text_style_height(&egui::TextStyle::Heading);
    let body_height = ui.text_style_height(&egui::TextStyle::Body);
    let summary_height = 4.0 + heading_height + spacing_y + body_height;
    let slider_height = ui.spacing().interact_size.y;
    let (_, rect) = ui.allocate_space(egui::vec2(
        available_width,
        summary_height + spacing_y + slider_height,
    ));
    let summary_rect =
        egui::Rect::from_min_size(rect.min, egui::vec2(available_width, summary_height));
    let slider_rect = egui::Rect::from_min_size(
        egui::pos2(rect.min.x, summary_rect.max.y + spacing_y),
        egui::vec2(available_width, slider_height),
    );

    let mut slider_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(slider_rect)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
    let resp = slider_ui
        .horizontal(|ui| {
            ui.label(t(lang, Key::OutputFrames));
            // Stretch the slider to consume the remaining row width.
            ui.spacing_mut().slider_width = (ui.available_width() - 8.0).max(120.0);
            ui.add(
                egui::Slider::new(&mut idx, 0..=max_idx)
                    .integer()
                    .show_value(false),
            )
        })
        .inner;

    if resp.changed() {
        let entry = &canonical_layouts[idx as usize];
        options.output_frames = entry.output_count;
        options.frame_skip = entry.frame_skip;
        *atlas_layout_manual = false;
    }

    let mut summary_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(summary_rect)
            .layout(egui::Layout::top_down(egui::Align::LEFT)),
    );
    show_output_summary(
        &mut summary_ui,
        output_summary_for_detent(
            canonical_layouts,
            idx as usize,
            options.atlas_dims,
            !resp.changed(),
        ),
        lang,
    );
}

fn output_summary_for_detent(
    canonical_layouts: &[DetentEntry],
    idx: usize,
    atlas_dims: (u32, u32),
    show_wasted_tiles: bool,
) -> OutputSummary {
    let entry = &canonical_layouts[idx];
    let summary = OutputSummary::new(
        entry.output_count as usize,
        entry.frame_skip,
        entry.ignored_tail_frames,
        atlas_dims,
    );
    if show_wasted_tiles {
        summary
    } else {
        summary.without_wasted_tiles()
    }
}

// Atlas group surfaces output count prominently before other controls.
fn show_output_summary(ui: &mut egui::Ui, summary: OutputSummary, lang: Lang) {
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(fmt(
            t(lang, Key::OutputFrameSummary),
            &[&summary.output_count],
        ))
        .heading()
        .strong(),
    );
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(fmt(t(lang, Key::FrameSkipSummary), &[&summary.frame_skip])).weak(),
        );
        if let Some(ignored_tail_frames) = summary.ignored_tail_frames {
            ui.colored_label(
                egui::Color32::from_rgb(210, 150, 60),
                fmt(
                    t(lang, Key::IgnoredTailFramesSummary),
                    &[&ignored_tail_frames],
                ),
            );
        }
        if let Some(wasted_tiles) = summary.wasted_tiles {
            ui.colored_label(
                egui::Color32::from_rgb(210, 150, 60),
                fmt(t(lang, Key::WastedTilesSummary), &[&wasted_tiles]),
            );
        }
    });
}

/// Predicted color/motion atlas pixel dims, with red highlight on overflow.
fn show_predicted_dims(
    ui: &mut egui::Ui,
    output_dims: Option<OutputDims>,
    max_dim: u32,
    lang: Lang,
) {
    let Some(dims) = output_dims else { return };
    ui.add_space(4.0);
    let warn = egui::Color32::from_rgb(220, 80, 80);
    let over = |a: u32, b: u32| a > max_dim || b > max_dim;

    let color_text = fmt(
        t(lang, Key::ColorAtlasDims),
        &[&dims.color.0, &dims.color.1],
    );
    if over(dims.color.0, dims.color.1) {
        ui.colored_label(warn, color_text);
    } else {
        ui.label(color_text);
    }

    let motion_text = fmt(
        t(lang, Key::MotionAtlasDims),
        &[&dims.motion.0, &dims.motion.1],
    );
    if over(dims.motion.0, dims.motion.1) {
        ui.colored_label(warn, motion_text);
    } else {
        ui.label(motion_text);
    }

    if dims.exceeds_limit(max_dim) {
        ui.colored_label(warn, fmt(t(lang, Key::ExceedsTextureLimit), &[&max_dim]));
    }
}

/// Motion group — options that shape the motion atlas.
fn show_motion_group(ui: &mut egui::Ui, options: &mut GenerateOptions, lang: Lang) {
    ui.heading(t(lang, Key::MotionHeading));

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::Encoding));
        let enc_label = match options.motion_vector_encoding {
            MotionVectorEncoding::R8G8Remap01 => t(lang, Key::EncodingR8G8Remap01),
            MotionVectorEncoding::SidefxLabsR8G8 => t(lang, Key::EncodingSidefxLabsR8G8),
        };
        egui::ComboBox::from_id_salt("mv_encoding")
            .selected_text(enc_label)
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut options.motion_vector_encoding,
                    MotionVectorEncoding::R8G8Remap01,
                    t(lang, Key::EncodingR8G8Remap01),
                );
                ui.selectable_value(
                    &mut options.motion_vector_encoding,
                    MotionVectorEncoding::SidefxLabsR8G8,
                    t(lang, Key::EncodingSidefxLabsR8G8),
                );
            })
            .response
            .on_hover_text(t(lang, Key::EncodingHover));
    });

    ui.checkbox(
        &mut options.halve_motion_vector,
        t(lang, Key::HalveMotionVector),
    )
    .on_hover_text(t(lang, Key::HalveMotionVectorHover));

    ui.checkbox(&mut options.stagger_pack, t(lang, Key::StaggerPack))
        .on_hover_text(t(lang, Key::StaggerPackHover));

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::TemporalSmoothing));
        ui.add(
            egui::Slider::new(&mut options.temporal_smoothing, 0.0..=1.0)
                .fixed_decimals(2)
                .step_by(0.05),
        )
        .on_hover_text(t(lang, Key::TemporalSmoothingHover));
    });

    ui.checkbox(
        &mut options.analyze_skipped_frames,
        t(lang, Key::AnalyzeSkippedFrames),
    )
    .on_hover_text(t(lang, Key::AnalyzeSkippedFramesHover));

    ui.checkbox(&mut options.is_loop, t(lang, Key::LoopOption))
        .on_hover_text(t(lang, Key::LoopOptionHover));
}

/// Color group — options that shape the color atlas.
fn show_color_group(ui: &mut egui::Ui, options: &mut GenerateOptions, lang: Lang) {
    ui.heading(t(lang, Key::ColorHeading));

    ui.checkbox(
        &mut options.premultiplied_alpha,
        t(lang, Key::PremultipliedAlpha),
    )
    .on_hover_text(t(lang, Key::PremultipliedAlphaHover));
}

/// Sidebar-footer language picker.
///
/// `jp_font_available` controls whether the Japanese option is selectable.
/// When `false`, the option is disabled with a tooltip explaining how to
/// install a Japanese font.
pub fn show_language_picker(ui: &mut egui::Ui, lang: &mut Lang, jp_font_available: bool) {
    ui.separator();
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(t(*lang, Key::LanguageLabel));
        egui::ComboBox::from_id_salt("language_picker")
            .selected_text(lang.display_name())
            .show_ui(ui, |ui| {
                ui.selectable_value(lang, Lang::En, Lang::En.display_name());
                let ja_label = Lang::Ja.display_name();
                if jp_font_available {
                    ui.selectable_value(lang, Lang::Ja, ja_label);
                } else {
                    ui.add_enabled(false, egui::Button::selectable(false, ja_label))
                        .on_disabled_hover_text(t(*lang, Key::LanguageDisabledTooltip));
                }
            });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_summary_hides_zero_wasted_tiles() {
        let summary = OutputSummary::new(48, 1, 0, (8, 6));

        assert_eq!(summary.output_count, 48);
        assert_eq!(summary.frame_skip, 1);
        assert_eq!(summary.ignored_tail_frames, None);
        assert_eq!(summary.wasted_tiles, None);
    }

    #[test]
    fn output_summary_reports_nonzero_wasted_tiles() {
        let summary = OutputSummary::new(48, 1, 0, (7, 7));

        assert_eq!(summary.output_count, 48);
        assert_eq!(summary.frame_skip, 1);
        assert_eq!(summary.wasted_tiles, Some(1));
    }

    #[test]
    fn output_summary_uses_selected_slider_detent() {
        let layouts = [
            DetentEntry {
                output_count: 2,
                frame_skip: 4,
                ignored_tail_frames: 0,
            },
            DetentEntry {
                output_count: 5,
                frame_skip: 1,
                ignored_tail_frames: 0,
            },
        ];

        let summary = output_summary_for_detent(&layouts, 1, (3, 2), true);

        assert_eq!(summary.output_count, 5);
        assert_eq!(summary.frame_skip, 1);
        assert_eq!(summary.wasted_tiles, Some(1));
    }

    #[test]
    fn output_summary_can_suppress_transient_wasted_tiles() {
        let layouts = [
            DetentEntry {
                output_count: 2,
                frame_skip: 4,
                ignored_tail_frames: 0,
            },
            DetentEntry {
                output_count: 5,
                frame_skip: 1,
                ignored_tail_frames: 0,
            },
        ];

        let summary = output_summary_for_detent(&layouts, 1, (3, 2), false);

        assert_eq!(summary.output_count, 5);
        assert_eq!(summary.frame_skip, 1);
        assert_eq!(summary.wasted_tiles, None);
    }

    #[test]
    fn output_summary_includes_ignored_tail_frames() {
        let layouts = [DetentEntry {
            output_count: 16,
            frame_skip: 5,
            ignored_tail_frames: 4,
        }];

        let summary = output_summary_for_detent(&layouts, 0, (4, 4), true);

        assert_eq!(summary.output_count, 16);
        assert_eq!(summary.frame_skip, 5);
        assert_eq!(summary.ignored_tail_frames, Some(4));
    }

    #[test]
    fn output_summary_suppresses_zero_ignored_tail_frames() {
        let layouts = [DetentEntry {
            output_count: 20,
            frame_skip: 4,
            ignored_tail_frames: 0,
        }];

        let summary = output_summary_for_detent(&layouts, 0, (5, 4), true);

        assert_eq!(summary.ignored_tail_frames, None);
    }
}
