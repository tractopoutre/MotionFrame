//! Locale lifecycle for the desktop shell.
//!
//! Maps a BCP-47 tag (or the user's OS locale) to a `Lang`, persists the
//! user's choice under a dedicated eframe storage key, probes the OS for a
//! Japanese-capable font, and installs fonts on the egui context.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use motionframe_ui::i18n::Lang;

/// eframe storage key for the persisted user-language preference. Kept
/// separate from `motionframe.options` so render-pipeline math and chrome
/// preferences have independent lifetimes.
pub const STORAGE_KEY: &str = "motionframe.locale";

/// Map a BCP-47 language tag (e.g. `"ja-JP"`, `"en"`, `"fr-CA"`) to a `Lang`.
///
/// The primary subtag (everything before the first `-` or `_`) is matched
/// case-insensitively. Unrecognized or empty tags resolve to English.
pub fn lang_from_tag(tag: &str) -> Lang {
    let primary = tag
        .split(['-', '_'])
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    match primary.as_str() {
        "ja" => Lang::Ja,
        _ => Lang::En,
    }
}

/// Read the OS user locale and map it to a `Lang`. Falls back to English
/// when `sys-locale` returns nothing.
pub fn detect_system_lang() -> Lang {
    sys_locale::get_locale()
        .map(|tag| lang_from_tag(&tag))
        .unwrap_or_default()
}

/// Load the persisted language preference; if absent, detect from the OS.
pub fn load_or_detect(storage: Option<&dyn eframe::Storage>) -> Lang {
    storage
        .and_then(|s| eframe::get_value::<Lang>(s, STORAGE_KEY))
        .unwrap_or_else(detect_system_lang)
}

/// Persist the chosen language under [`STORAGE_KEY`].
pub fn save(storage: &mut dyn eframe::Storage, lang: Lang) {
    eframe::set_value(storage, STORAGE_KEY, &lang);
}

/// Per-OS candidate paths for a Japanese-capable system font, in priority
/// order. The first existing file wins.
#[cfg(target_os = "macos")]
const JP_FONT_CANDIDATES: &[&str] = &[
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
];

#[cfg(target_os = "windows")]
const JP_FONT_CANDIDATES: &[&str] = &[
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
];

#[cfg(all(unix, not(target_os = "macos")))]
const JP_FONT_CANDIDATES: &[&str] = &[
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansJP-Regular.otf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf",
    "/usr/share/fonts/truetype/ipafont-gothic/ipag.ttf",
];

/// Find a Japanese-capable font on the local filesystem. `None` if no
/// candidate exists (e.g. minimal Linux install with no JP font package).
pub fn probe_jp_font() -> Option<PathBuf> {
    for candidate in JP_FONT_CANDIDATES {
        let p = Path::new(candidate);
        if p.exists() {
            return Some(p.to_path_buf());
        }
    }
    None
}

/// Internal name under which the probed JP font is registered with egui.
const JP_FONT_NAME: &str = "system_jp";

/// Compute the `FontTweak` that aligns `fallback_font_bytes`'s baseline with
/// the baseline of egui's primary proportional font.
///
/// Reads metrics through `skrifa` — the same crate epaint uses internally,
/// so the values are exactly what the layout code consumes. The formula
/// mirrors `epaint::text::text_layout` (per-font ascent + 0.5 * row-height
/// centering for fallbacks). No empirical constants — purely derived from
/// the metrics of the fonts in play.
///
/// Returns `FontTweak::default()` if either font fails to parse; the
/// fallback degrades to "small visible misalignment" rather than missing
/// glyphs.
fn fallback_baseline_tweak(fallback_font_bytes: &[u8]) -> egui::FontTweak {
    let Some(fallback) = font_metrics(fallback_font_bytes, 0) else {
        log::warn!("could not parse fallback font metrics; using default tweak");
        return egui::FontTweak::default();
    };

    let defs = egui::FontDefinitions::default();
    let primary_name = defs
        .families
        .get(&egui::FontFamily::Proportional)
        .and_then(|v| v.first());
    let Some(primary_data) = primary_name.and_then(|n| defs.font_data.get(n)) else {
        return egui::FontTweak::default();
    };
    let Some(primary) = font_metrics(&primary_data.font, primary_data.index) else {
        return egui::FontTweak::default();
    };

    // epaint (text_layout.rs:684) places each glyph at:
    //     y = font_face_ascent + 0.5 * (primary_row_height - this_row_height)
    // For the primary font the centering term is zero. So:
    //     primary_baseline_em  = primary.ascent
    //     fallback_baseline_em = fallback.ascent
    //                          + 0.5 * (primary.row_height - fallback.row_height)
    // FontTweak.y_offset_factor (multiplied by font size) is added to the
    // glyph's `y` position via metrics.y_offset_in_points. To make
    // fallback_baseline_em + tweak == primary_baseline_em:
    let primary_tweak = &primary_data.tweak;
    let primary_baseline = primary.ascent * primary_tweak.scale + primary_tweak.y_offset_factor;
    let fallback_baseline_uncorrected =
        fallback.ascent + 0.5 * (primary.row_height - fallback.row_height);

    egui::FontTweak {
        y_offset_factor: primary_baseline - fallback_baseline_uncorrected,
        ..Default::default()
    }
}

/// Normalized line metrics (per `units_per_em`) for the face at `index`.
///
/// Reads via `skrifa::FontRef::metrics(Size::unscaled(), …)`, which is the
/// same call epaint makes internally; metric-source picking
/// (`USE_TYPO_METRICS` / `hhea` / win-fallback) lives inside skrifa, so we
/// don't replicate it.
struct FontLineMetrics {
    ascent: f32,
    /// `(ascent - descent + leading) / upem`. Descender is negative.
    row_height: f32,
}

fn font_metrics(bytes: &[u8], index: u32) -> Option<FontLineMetrics> {
    use skrifa::MetadataProvider;

    let font = skrifa::FontRef::from_index(bytes, index).ok()?;
    let m = font.metrics(
        skrifa::instance::Size::unscaled(),
        skrifa::instance::LocationRef::default(),
    );
    let upem = f32::from(m.units_per_em);
    if upem == 0.0 {
        return None;
    }
    let ascent = m.ascent / upem;
    let descent = m.descent / upem;
    let leading = m.leading / upem;
    Some(FontLineMetrics {
        ascent,
        row_height: ascent - descent + leading,
    })
}

/// Install fonts on the egui context. Default fonts are always installed;
/// when a JP font path is provided, it is registered as the *last* fallback
/// for both proportional and monospace families.
///
/// Latin glyphs continue to resolve through egui's default fonts. Japanese
/// characters fall through to the registered system JP font.
pub fn install_fonts(ctx: &egui::Context, jp_font_path: Option<&Path>) {
    let mut defs = egui::FontDefinitions::default();

    let Some(path) = jp_font_path else {
        ctx.set_fonts(defs);
        return;
    };

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            log::warn!(
                "failed to read JP font at {}: {e}; CJK glyphs will render as tofu",
                path.display()
            );
            ctx.set_fonts(defs);
            return;
        }
    };

    let tweak = fallback_baseline_tweak(&bytes);
    let font_data = egui::FontData::from_owned(bytes).tweak(tweak);
    defs.font_data
        .insert(JP_FONT_NAME.to_owned(), Arc::new(font_data));
    defs.families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .push(JP_FONT_NAME.to_owned());
    defs.families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .push(JP_FONT_NAME.to_owned());

    ctx.set_fonts(defs);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lang_from_tag_recognizes_japanese() {
        assert_eq!(lang_from_tag("ja"), Lang::Ja);
        assert_eq!(lang_from_tag("ja-JP"), Lang::Ja);
        assert_eq!(lang_from_tag("ja_JP"), Lang::Ja);
        assert_eq!(lang_from_tag("JA"), Lang::Ja);
        assert_eq!(lang_from_tag("Ja-jp"), Lang::Ja);
    }

    #[test]
    fn lang_from_tag_defaults_to_english() {
        assert_eq!(lang_from_tag("en"), Lang::En);
        assert_eq!(lang_from_tag("en-US"), Lang::En);
        assert_eq!(lang_from_tag("fr-CA"), Lang::En);
        assert_eq!(lang_from_tag("zh-CN"), Lang::En);
        assert_eq!(lang_from_tag(""), Lang::En);
        assert_eq!(lang_from_tag("???"), Lang::En);
    }

    struct FakeStorage {
        map: std::collections::HashMap<String, String>,
    }

    impl FakeStorage {
        fn new() -> Self {
            Self {
                map: std::collections::HashMap::new(),
            }
        }
    }

    impl eframe::Storage for FakeStorage {
        fn get_string(&self, key: &str) -> Option<String> {
            self.map.get(key).cloned()
        }
        fn set_string(&mut self, key: &str, value: String) {
            self.map.insert(key.to_owned(), value);
        }
        fn flush(&mut self) {}
    }

    #[test]
    fn lang_round_trips_through_storage() {
        let mut s = FakeStorage::new();
        save(&mut s, Lang::Ja);
        let read_back = load_or_detect(Some(&s));
        assert_eq!(read_back, Lang::Ja);
    }

    #[test]
    fn missing_lang_falls_back_to_detection() {
        let s = FakeStorage::new();
        // detect_system_lang() reads the real OS locale here; we only
        // assert load_or_detect doesn't panic and returns *some* Lang.
        let _ = load_or_detect(Some(&s));
    }
}
