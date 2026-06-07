//! Validates that the preview WGSL shader parses and validates correctly.

#[test]
fn preview_shader_compiles() {
    let src = include_str!("../src/preview/shader.wgsl");
    let mut frontend = naga::front::wgsl::Frontend::new();
    let module = frontend.parse(src).expect("WGSL parse failed");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator.validate(&module).expect("WGSL validation failed");
}
