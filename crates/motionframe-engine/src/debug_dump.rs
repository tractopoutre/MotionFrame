//! Debug dump helpers: save intermediate pipeline stages as TGA images.
//!
//! Enable by setting `MFRAME_DUMP=1` in the environment before running.
//! Images land in `debug_dump/`.

use std::path::Path;

use crate::io::tga::save_tga;
use crate::pipeline::{Flow, ImageF32, ImageRgba8};

/// Returns `true` if debug dump is enabled (env var `MFRAME_DUMP` set).
pub fn is_enabled() -> bool {
    std::env::var("MFRAME_DUMP").is_ok()
}

fn save(label: &str, img: &ImageRgba8) {
    let path = format!("debug_dump/{label}.tga");
    std::fs::create_dir_all("debug_dump").ok();
    if let Err(e) = save_tga(Path::new(&path), img) {
        eprintln!("[debug_dump] failed to write {path}: {e}");
    }
}

/// Save an `ImageF32` as a grayscale TGA.
pub fn save_f32(label: &str, img: &ImageF32) {
    if !is_enabled() {
        return;
    }
    let mut data = Vec::with_capacity((img.width * img.height * 4) as usize);
    for &v in &img.data {
        let u8v = (v.clamp(0.0, 1.0) * 255.0) as u8;
        data.extend_from_slice(&[u8v, u8v, u8v, 255]);
    }
    let rgba = ImageRgba8 { width: img.width, height: img.height, data };
    save(label, &rgba);
}

/// Save a `Flow` as a red-cyan motion visualization TGA.
pub fn save_flow(label: &str, flow: &Flow) {
    if !is_enabled() {
        return;
    }
    let max_val = flow
        .data
        .iter()
        .map(|[dx, dy]| dx.abs().max(dy.abs()))
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1e-8);
    let inv = 127.0 / max_val;
    let mut data = Vec::with_capacity((flow.width * flow.height * 4) as usize);
    for &[dx, dy] in &flow.data {
        let r = (dx * inv + 127.0).clamp(0.0, 255.0) as u8;
        let g = (dy * inv + 127.0).clamp(0.0, 255.0) as u8;
        data.extend_from_slice(&[r, g, 0, 255]);
    }
    let rgba = ImageRgba8 { width: flow.width, height: flow.height, data };
    save(label, &rgba);
}

/// Save raw RGBA8 pixel data as TGA.
pub fn save_rgba(label: &str, w: u32, h: u32, data: &[u8]) {
    if !is_enabled() {
        return;
    }
    let rgba = ImageRgba8 { width: w, height: h, data: data.to_vec() };
    save(label, &rgba);
}
