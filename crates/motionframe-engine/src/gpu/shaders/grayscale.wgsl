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
    let gray = dot(pixel.rgb, vec3(0.299, 0.587, 0.114)) * 255.0;
    textureStore(out_tex, vec2(x, y), vec4(gray, 0.0, 0.0, 0.0));
}
