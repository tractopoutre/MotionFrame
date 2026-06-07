#[test]
fn loads_explosion_frame() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("explosion00")
        .join("explosion00-frame001.tga");
    let img = motionframe_engine::io::tga::load_rgba(&path).unwrap();
    assert_eq!(img.width, 400);
    assert_eq!(img.height, 400);
    assert_eq!(img.data.len(), 640_000);
}

#[test]
fn premultiply_preserves_dimensions() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("explosion00")
        .join("explosion00-frame001.tga");
    let img = motionframe_engine::io::tga::load_rgba(&path).unwrap();
    let premul = motionframe_engine::io::tga::premultiply_alpha(&img);
    assert_eq!(premul.width, img.width);
    assert_eq!(premul.height, img.height);
    assert_eq!(premul.data.len(), img.data.len());
}
