@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let in_dims = textureDimensions(in_tex);
    let out_dims = textureDimensions(out_tex);
    let ow = i32(out_dims.x);
    let oh = i32(out_dims.y);
    let sw = i32(in_dims.x);
    let sh = i32(in_dims.y);
    let x = min(i32(id.x), ow - 1);
    let y = min(i32(id.y), oh - 1);

    let sx = min(x / 2, sw - 1);
    let sy = min(y / 2, sh - 1);

    let flow = textureLoad(in_tex, vec2(sx, sy), 0);

    textureStore(out_tex, vec2(x, y), vec4(flow.x * 2.0, flow.y * 2.0, 0.0, 0.0));
}
