//! Tab views for the central panel area (Color / Motion / Visualization / Preview).

use egui::{ColorImage, TextureHandle, TextureOptions};

use motionframe_engine::pipeline::ImageRgba8;

use crate::i18n::{t, Key, Lang};

/// Which tab is currently selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TabKind {
    /// Color atlas view.
    #[default]
    Color,
    /// Motion atlas view.
    Motion,
    /// Arrow-based optical flow visualization.
    Visualization,
    /// Animated GPU preview that warps the color atlas by the encoded motion vectors.
    Preview,
}

/// Render the tab bar and return the newly selected tab (if changed).
pub fn tab_bar(ui: &mut egui::Ui, current: &mut TabKind, lang: Lang) {
    ui.horizontal(|ui| {
        if ui
            .selectable_label(*current == TabKind::Color, t(lang, Key::TabColor))
            .clicked()
        {
            *current = TabKind::Color;
        }
        if ui
            .selectable_label(*current == TabKind::Motion, t(lang, Key::TabMotion))
            .clicked()
        {
            *current = TabKind::Motion;
        }
        if ui
            .selectable_label(
                *current == TabKind::Visualization,
                t(lang, Key::TabVisualization),
            )
            .clicked()
        {
            *current = TabKind::Visualization;
        }
        if ui
            .selectable_label(*current == TabKind::Preview, t(lang, Key::TabPreview))
            .clicked()
        {
            *current = TabKind::Preview;
        }
    });
    ui.separator();
}

/// Render empty-state placeholder (for Empty / Loading states).
pub fn draw_empty_state(ui: &mut egui::Ui, lang: Lang) {
    ui.centered_and_justified(|ui| {
        ui.vertical_centered(|ui| {
            ui.label(
                egui::RichText::new(t(lang, Key::DropFrameOrFolder))
                    .color(egui::Color32::from_gray(128))
                    .size(20.0),
            );
            ui.label(
                egui::RichText::new(t(lang, Key::AcceptedFormats))
                    .color(egui::Color32::from_gray(100))
                    .size(14.0),
            );
        });
    });
}

/// Render a "Click Generate" placeholder for the Ready state.
pub fn draw_ready_placeholder(ui: &mut egui::Ui, lang: Lang) {
    ui.centered_and_justified(|ui| {
        ui.label(
            egui::RichText::new(t(lang, Key::ClickGenerate))
                .color(egui::Color32::from_gray(128))
                .size(16.0),
        );
    });
}

/// Render an image tab with zoom slider and scroll area.
///
/// Texture sizes from `size_vec2()` are in points. Divide by `pixels_per_point`
/// so zoom=1 means "one texture pixel per physical pixel" — what users expect
/// "100%" to look like.
pub fn draw_image_tab(ui: &mut egui::Ui, texture: &TextureHandle, zoom: &mut f32, lang: Lang) {
    ui.horizontal(|ui| {
        ui.label(t(lang, Key::Zoom));
        ui.add(egui::Slider::new(zoom, 0.25..=4.0).show_value(true));
    });
    let ppp = ui.ctx().pixels_per_point();
    let tex_size = texture.size_vec2() * (*zoom / ppp);
    egui::ScrollArea::both().show(ui, |ui| {
        ui.image(egui::load::SizedTexture::new(texture.id(), tex_size));
    });
}

/// Upload an `ImageRgba8` to the GPU as an egui texture.
pub fn upload_texture(
    ctx: &egui::Context,
    name: impl Into<String>,
    img: &ImageRgba8,
) -> TextureHandle {
    let color_image =
        ColorImage::from_rgba_unmultiplied([img.width as usize, img.height as usize], &img.data);
    ctx.load_texture(name, color_image, TextureOptions::default())
}
