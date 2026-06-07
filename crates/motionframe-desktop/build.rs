use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = manifest
        .join("..")
        .join("..")
        .join("THIRD-PARTY-LICENSES.md");
    println!("cargo:rerun-if-changed={}", src.display());

    // The file lives at the workspace root, two directories up. In workspace
    // builds it's always present. In packaged/`cargo install` builds (where
    // only crate-local files are copied into the tarball) it's missing — fall
    // back to a placeholder rather than panicking, so the binary still builds.
    let text = std::fs::read(&src)
        .unwrap_or_else(|_| b"Third-party license text was not bundled with this build.".to_vec());
    let compressed = miniz_oxide::deflate::compress_to_vec(&text, 10);

    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR set by cargo"));
    std::fs::write(out_dir.join("licenses.deflate"), &compressed)
        .expect("write compressed licenses blob");
}
