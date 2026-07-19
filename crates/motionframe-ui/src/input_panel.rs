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
}

impl OutputSummary {
    #[must_use]
    fn new(output_count: usize, frame_skip: u32, ignored_tail_frames: u32) -> Self {
        Self {
            output_count,
            frame_skip,
            ignored_tail_frames: (ignored_tail_frames > 0).then_some(ignored_tail_frames),
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
/// `n_input` is the total input frame count (post atlas-input slicing).
/// `n_after_range` is the frame count after applying start/end range.
/// `canonical_layouts` is the sorted-by-count detent set for `n_input`.
#[allow(clippy::too_many_arguments)]
pub fn show_options(
    ui: &mut egui::Ui,
    options: &mut GenerateOptions,
    atlas_layout_manual: &mut bool,
    output_dims: Option<OutputDims>,
    n_input: u32,
    n_after_range: u32,
    canonical_layouts: &[DetentEntry],
    lang: Lang,
    sequence_loaded: bool,
) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        show_input_group(ui, options, lang);
        if options.input_atlas_dims.is_some() {
            ui.separator();
        }
        // Frame range appears FIRST (before atlas/output summary)
        show_frame_range_group(ui, options, n_input, lang);
        ui.separator();
        show_atlas_group(
            ui,
            options,
            atlas_layout_manual,
            output_dims,
            n_after_range,
            canonical_layouts,
            lang,
        );
        ui.separator();
        show_output_group(ui, options, sequence_loaded, lang);
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

/// Atlas group — output grid, tile size, atlas resolution, and shared
/// resampling options. Always visible.
#[allow(unused_variables)]
fn show_atlas_group(
    ui: &mut egui::Ui,
    options: &mut GenerateOptions,
    atlas_layout_manual: &mut bool,
    output_dims: Option<OutputDims>,
    n_after_range: u32,
    canonical_layouts: &[DetentEntry],
    lang: Lang,
) {
    ui.heading(t(lang, Key::AtlasHeading));

    show_auto_output_count(
        ui,
        options,
        n_after_range,
        lang,
    );

    // --- Atlas resolution ---
    ui.horizontal(|ui| {
        ui.label(t(lang, Key::AtlasResolution));
        let mut res = options.atlas_resolution;
        let resp = ui.add(egui::DragValue::new(&mut res).range(256..=8192).speed(1));
        if resp.changed() {
            // Snap to power of 2
            let snapped = 256u32.max(2u32.pow(res.ilog2()).min(8192));
            options.atlas_resolution = snapped;
            *atlas_layout_manual = false;
        }
        resp.on_hover_text(t(lang, Key::AtlasResolutionHover));
    });

    // --- Tiles per row / column (auto-filled, editable, lock badge) ---
    let auto_label = t(lang, Key::LockBadgeAuto);
    let manual_label = t(lang, Key::LockBadgeManual);
    let lock_badge = |is_manual: bool| if is_manual { manual_label } else { auto_label };

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::Rows));
        let resp = ui.add(egui::DragValue::new(&mut options.atlas_dims.1).range(1..=256));
        if resp.changed() {
            *atlas_layout_manual = true;
        }
        ui.label(lock_badge(*atlas_layout_manual));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::Columns));
        let resp = ui.add(egui::DragValue::new(&mut options.atlas_dims.0).range(1..=256));
        if resp.changed() {
            *atlas_layout_manual = true;
        }
        ui.label(lock_badge(*atlas_layout_manual));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::TilePixelWidth));
        let tile_size = options.atlas_resolution / options.atlas_dims.0.max(options.atlas_dims.1).max(1);
        ui.label(format!("{} px (auto)", tile_size));
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

/// Output-frames integer DragValue, replacing the old detent-based slider.
fn show_auto_output_count(
    ui: &mut egui::Ui,
    options: &mut GenerateOptions,
    n_after_range: u32,
    lang: Lang,
) {
    if n_after_range == 0 {
        return;
    }
    let max_slots = options.atlas_dims.0.saturating_mul(options.atlas_dims.1);
    let output_count = max_slots.min(n_after_range);
    options.output_frames = output_count;
    options.frame_skip = if output_count > 0 {
        (n_after_range / output_count).saturating_sub(1)
    } else {
        0
    };

    show_output_summary(
        ui,
        OutputSummary::new(output_count as usize, options.frame_skip, 0),
        lang,
    );
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
    });
}

/// Predicted color/motion atlas pixel dims.
fn show_predicted_dims(
    ui: &mut egui::Ui,
    output_dims: Option<OutputDims>,
    _max_dim: u32,
    lang: Lang,
) {
    let Some(dims) = output_dims else { return };
    ui.add_space(4.0);

    let color_text = fmt(
        t(lang, Key::ColorAtlasDims),
        &[&dims.color.0, &dims.color.1],
    );
    ui.label(color_text);

    let motion_text = fmt(
        t(lang, Key::MotionAtlasDims),
        &[&dims.motion.0, &dims.motion.1],
    );
    ui.label(motion_text);
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

/// Output naming section — format template, basename override, type labels,
/// and live filename preview.
fn show_output_group(ui: &mut egui::Ui, options: &mut GenerateOptions, sequence_loaded: bool, lang: Lang) {
    ui.heading(t(lang, Key::OutputHeading));

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::OutputFormat));
        ui.add(
            egui::TextEdit::singleline(&mut options.output_name_format)
                .desired_width(f32::INFINITY)
                .hint_text("[basename]_[cols]x[rows][suffix].[ext]"),
        )
        .on_hover_text(t(lang, Key::OutputFormatHover));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::OutputBaseName));
        ui.add(
            egui::TextEdit::singleline(&mut options.output_name_basename)
                .desired_width(120.0)
                .hint_text(t(lang, Key::OutputBaseNameHover)),
        )
        .on_hover_text(t(lang, Key::OutputBaseNameHover));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::ColorSuffix));
        ui.add(egui::TextEdit::singleline(&mut options.output_suffix_color).desired_width(80.0))
            .on_hover_text(t(lang, Key::ColorSuffixHover));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::MotionSuffix));
        ui.add(egui::TextEdit::singleline(&mut options.output_suffix_motion).desired_width(80.0))
            .on_hover_text(t(lang, Key::MotionSuffixHover));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::MetaSuffix));
        ui.add(egui::TextEdit::singleline(&mut options.output_suffix_meta).desired_width(80.0))
            .on_hover_text(t(lang, Key::MetaSuffixHover));
    });

    // Live preview (only when a sequence is loaded)
    if sequence_loaded {
        let (cols, rows) = options.atlas_dims;
        let basename = if options.output_name_basename.is_empty() {
            "input"
        } else {
            &options.output_name_basename
        };
        ui.add_space(4.0);
        ui.label(t(lang, Key::OutputPreview));

        let format = &options.output_name_format;
        if format.is_empty() {
            ui.colored_label(egui::Color32::RED, t(lang, Key::OutputPreviewEmpty));
        } else {
            let color_name = motionframe_engine::pipeline::output_naming::interpolate_name_format(
                format,
                &motionframe_engine::pipeline::output_naming::NameTokens {
                    basename,
                    cols,
                    rows,
                    suffix: &options.output_suffix_color,
                    ext: "tga",
                },
            );
            let motion_name = motionframe_engine::pipeline::output_naming::interpolate_name_format(
                format,
                &motionframe_engine::pipeline::output_naming::NameTokens {
                    basename,
                    cols,
                    rows,
                    suffix: &options.output_suffix_motion,
                    ext: "tga",
                },
            );
            let meta_name = motionframe_engine::pipeline::output_naming::interpolate_name_format(
                format,
                &motionframe_engine::pipeline::output_naming::NameTokens {
                    basename,
                    cols,
                    rows,
                    suffix: &options.output_suffix_meta,
                    ext: "json",
                },
            );
            ui.label(egui::RichText::new(color_name).size(12.0).weak());
            ui.label(egui::RichText::new(motion_name).size(12.0).weak());
            ui.label(egui::RichText::new(meta_name).size(12.0).weak());
        }
    }
}

/// Frame range section — start/end frame drag values. Only visible in
/// sequence mode (not atlas mode).
fn show_frame_range_group(ui: &mut egui::Ui, options: &mut GenerateOptions, n_input: u32, lang: Lang) {
    if options.input_atlas_dims.is_some() {
        return; // only in sequence mode
    }
    ui.heading(t(lang, Key::FrameRangeHeading));

    let mut start = options.start_frame;
    let mut end = options.end_frame;

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::StartFrame));
        ui.add(egui::DragValue::new(&mut start).range(0..=end.saturating_sub(1).max(0)))
            .on_hover_text(t(lang, Key::StartFrameHover));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::EndFrame));
        ui.add(egui::DragValue::new(&mut end).range(1..=n_input))
            .on_hover_text(t(lang, Key::EndFrameHover));
    });

    if start > end.saturating_sub(1) {
        ui.colored_label(egui::Color32::RED, "Start must be less than end");
    }

    options.start_frame = start;
    options.end_frame = end;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output_summary_for_detent(
        canonical_layouts: &[DetentEntry],
        idx: usize,
    ) -> OutputSummary {
        let entry = &canonical_layouts[idx];
        OutputSummary::new(
            entry.output_count as usize,
            entry.frame_skip,
            entry.ignored_tail_frames,
        )
    }

    #[test]
    fn output_summary_includes_ignored_tail_frames() {
        let layouts = [DetentEntry {
            output_count: 16,
            frame_skip: 5,
            ignored_tail_frames: 4,
        }];

        let summary = output_summary_for_detent(&layouts, 0);

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

        let summary = output_summary_for_detent(&layouts, 0);

        assert_eq!(summary.ignored_tail_frames, None);
    }

    #[test]
    fn preview_uses_interpolate_name_format() {
        let mut opts = GenerateOptions::default();
        opts.output_name_format = "[basename]_custom.[ext]".into();
        let (cols, rows) = opts.atlas_dims;
        let tokens = motionframe_engine::pipeline::output_naming::NameTokens {
            basename: "test",
            cols,
            rows,
            suffix: "",
            ext: "tga",
        };
        let name = motionframe_engine::pipeline::output_naming::interpolate_name_format(
            &opts.output_name_format, &tokens,
        );
        assert_eq!(name, "test_custom.tga");
    }

    #[test]
    fn empty_format_falls_back_to_default_in_preview() {
        let mut opts = GenerateOptions::default();
        opts.output_name_format = String::new();
        let (cols, rows) = opts.atlas_dims;
        let tokens = motionframe_engine::pipeline::output_naming::NameTokens {
            basename: "test",
            cols,
            rows,
            suffix: "_MV",
            ext: "tga",
        };
        let name = motionframe_engine::pipeline::output_naming::interpolate_name_format(
            &opts.output_name_format, &tokens,
        );
        assert_eq!(name, "test_8x8_MV.tga");
    }
}
