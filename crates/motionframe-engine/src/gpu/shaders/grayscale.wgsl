@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(in_tex);
    let w = dims.x;
    let h = dims.y;
    let x = min(id.x, w - 1);
    let y = min(id.y, h - 1);
    let pixel = textureLoad(in_tex, vec2i(x, y), 0);
    // Rgba8Unorm loads as [0,1]; BT.601 luma * 255 to match CPU [0,255] range
    let gray = dot(pixel.rgb, vec3(0.299, 0.587, 0.114)) * 255.0;
    textureStore(out_tex, vec2i(x, y), vec4(gray, 0.0, 0.0, 0.0));
}
