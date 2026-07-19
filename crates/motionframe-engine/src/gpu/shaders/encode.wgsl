@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> max_strength: vec4<f32>;
@group(0) @binding(2) var out_tex: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(in_tex);
    let w = i32(dims.x);
    let h = i32(dims.y);
    let x = min(i32(id.x), w - 1);
    let y = min(i32(id.y), h - 1);

    let flow = textureLoad(in_tex, vec2(x, y), 0);
    let strength = max_strength.x;

    var r: f32;
    var g: f32;
    if (strength < 1e-8) {
        r = 0.5;
        g = 0.5;
    } else {
        r = clamp(flow.x / strength * 0.5 + 0.5, 0.0, 1.0);
        g = clamp(flow.y / strength * 0.5 + 0.5, 0.0, 1.0);
    }

    textureStore(out_tex, vec2(x, y), vec4(r, g, 0.0, 0.0));
}
