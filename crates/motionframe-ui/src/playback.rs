//! Playback transport controls for the preview tab.

use crate::i18n::{fmt, t, Key, Lang};

/// Background fill behind the preview, applied where the color atlas alpha < 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreviewBg {
    Black,
    Gray,
    White,
    #[default]
    Checker,
}

impl PreviewBg {
    /// Wire format consumed by the preview shader's `bg_mode` uniform.
    pub const fn as_u32(self) -> u32 {
        match self {
            Self::Black => 0,
            Self::Gray => 1,
            Self::White => 2,
            Self::Checker => 3,
        }
    }
}

/// How adjacent frames are blended in the preview. Lets the artist A/B the
/// motion-vector warp against a naive cross-fade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Sample each frame at UV warped by ±mv*t, then mix.
    #[default]
    MotionVector,
    /// Plain `mix(c0, c1, t_frac)` — no UV warp.
    CrossFade,
}

impl BlendMode {
    /// Wire format consumed by the preview shader's `blend_mode` uniform.
    pub const fn as_u32(self) -> u32 {
        match self {
            Self::MotionVector => 0,
            Self::CrossFade => 1,
        }
    }
}

/// Playback state for the preview animation.
pub struct PlaybackState {
    /// Whether the animation is currently playing.
    pub playing: bool,
    /// Fractional frame index (time).
    pub time: f32,
    /// Frames per second (1–60, default 8).
    pub fps: f32,
    /// Motion vector strength (0–2). Driven by generation metadata; the UI
    /// only displays it. Mutate via [`PlaybackState::set_strength`].
    strength: f32,
    /// Total number of frames in the atlas.
    pub frame_count: u32,
    /// Background fill mode for the preview.
    pub bg: PreviewBg,
    /// Blend mode: motion-vector warped vs. plain cross-fade.
    pub blend_mode: BlendMode,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self {
            playing: true,
            time: 0.0,
            fps: 8.0,
            strength: 1.0,
            frame_count: 1,
            bg: PreviewBg::default(),
            blend_mode: BlendMode::default(),
        }
    }
}

impl PlaybackState {
    /// Read the current motion-vector strength.
    pub const fn strength(&self) -> f32 {
        self.strength
    }

    /// Update the motion-vector strength from generation metadata.
    pub const fn set_strength(&mut self, value: f32) {
        self.strength = value.clamp(0.0, 2.0);
    }

    /// Advance the playback time by the given delta (seconds).
    /// Wraps around at `frame_count` for seamless looping.
    pub fn advance(&mut self, dt_secs: f64) {
        if !self.playing || self.frame_count == 0 {
            return;
        }
        self.time += (dt_secs * f64::from(self.fps)) as f32;
        let total = self.frame_count as f32;
        self.time %= total;
        if self.time < 0.0 {
            self.time += total;
        }
    }
}

/// Draw playback transport controls. Returns true if animation is playing (caller should repaint).
pub fn draw_playback_controls(ui: &mut egui::Ui, state: &mut PlaybackState, lang: Lang) -> bool {
    ui.horizontal(|ui| {
        let label = if state.playing {
            t(lang, Key::Pause)
        } else {
            t(lang, Key::Play)
        };
        if ui.button(label).clicked() {
            state.playing = !state.playing;
        }

        ui.separator();

        // Frame scrub slider — always interactive; dragging while playing
        // pulls the time forward/back, then animation continues from there.
        // Hide the slider's auto value text and show a monospace fixed-digit
        // label so the row width stays constant regardless of frame number.
        let max = state.frame_count.saturating_sub(1) as f32;
        let mut frame_val = state.time.clamp(0.0, max);
        ui.label(t(lang, Key::Frame));
        if ui
            .add(
                // smart_aim biases toward "nicer" round numbers (0.0, 0.1, …)
                // and would skip 0.05 increments even with step_by, so we
                // disable it and let step_by be the only quantizer.
                //
                // clamping(Edits) is load-bearing: the default Always mode
                // re-snaps the value every frame and marks the response as
                // changed, which would write the snapped value back into
                // state.time and pin playback at low FPS.
                egui::Slider::new(&mut frame_val, 0.0..=max)
                    .step_by(0.05)
                    .smart_aim(false)
                    .clamping(egui::SliderClamping::Edits)
                    .show_value(false),
            )
            .changed()
        {
            state.time = frame_val;
        }
        let max_int = state.frame_count.saturating_sub(1);
        let frame_text = format!("{frame_val:.2}/{max_int}");
        let widest_frame_text = format!("{max:.2}/{max_int}");
        let frame_width = ui
            .painter()
            .layout_no_wrap(
                widest_frame_text,
                egui::TextStyle::Monospace.resolve(ui.style()),
                ui.visuals().text_color(),
            )
            .rect
            .width();
        ui.allocate_ui_with_layout(
            egui::vec2(frame_width, ui.spacing().interact_size.y),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                ui.monospace(frame_text);
            },
        );

        ui.separator();

        // FPS slider
        ui.label(t(lang, Key::Fps));
        ui.add(egui::Slider::new(&mut state.fps, 1.0..=60.0).integer());

        ui.separator();

        // Strength is sourced from generation metadata; show as a label so it
        // can't drift from the encoded atlas it was rendered against.
        ui.label(fmt(
            t(lang, Key::Strength),
            &[&format!("{:.2}", state.strength)],
        ));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::Background));
        ui.radio_value(&mut state.bg, PreviewBg::Black, t(lang, Key::BgBlack));
        ui.radio_value(&mut state.bg, PreviewBg::Gray, t(lang, Key::BgGray));
        ui.radio_value(&mut state.bg, PreviewBg::White, t(lang, Key::BgWhite));
        ui.radio_value(&mut state.bg, PreviewBg::Checker, t(lang, Key::BgChecker));
    });

    ui.horizontal(|ui| {
        ui.label(t(lang, Key::Blend));
        ui.radio_value(
            &mut state.blend_mode,
            BlendMode::MotionVector,
            t(lang, Key::BlendMotionVector),
        )
        .on_hover_text(t(lang, Key::BlendMotionVectorHover));
        ui.radio_value(
            &mut state.blend_mode,
            BlendMode::CrossFade,
            t(lang, Key::BlendCrossFade),
        )
        .on_hover_text(t(lang, Key::BlendCrossFadeHover));
    });

    state.playing
}
