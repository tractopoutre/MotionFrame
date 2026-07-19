@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(in_tex);
    let w = i32(dims.x);
    let h = i32(dims.y);
    let x = min(i32(id.x), w - 1);
    let y = min(i32(id.y), h - 1);
    let pixel = textureLoad(in_tex, vec2(x, y), 0);
    // Match CPU rgba_to_gray_f32: black floor = 64, alpha-aware premultiplied luma
    let lum = dot(pixel.rgb, vec3(0.299, 0.587, 0.114));
    let gray = pixel.a * 63.0 + lum * 191.0;
    textureStore(out_tex, vec2(x, y), vec4(gray, 0.0, 0.0, 0.0));
}
