use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let license_src = manifest
        .join("..")
        .join("..")
        .join("THIRD-PARTY-LICENSES.md");
    let font_src = manifest
        .join("assets")
        .join("fonts")
        .join("LINESeedJP_A_TTF_Rg.ttf");
    println!("cargo:rerun-if-changed={}", license_src.display());
    println!("cargo:rerun-if-changed={}", font_src.display());

    let text = std::fs::read(&license_src)
        .unwrap_or_else(|_| b"Third-party license text was not bundled with this build.".to_vec());

    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR set by cargo"));
    let compressed = miniz_oxide::deflate::compress_to_vec(&text, 10);
    std::fs::write(out_dir.join("licenses.deflate"), &compressed)
        .expect("write compressed licenses blob");

    let font = std::fs::read(&font_src).expect("read LINE Seed JP font");
    let compressed_font = miniz_oxide::deflate::compress_to_vec(&font, 10);
    std::fs::write(out_dir.join("line_seed_jp.deflate"), &compressed_font)
        .expect("write compressed LINE Seed JP font blob");
}
