@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let in_dims = textureDimensions(in_tex);
    let out_dims = textureDimensions(out_tex);
    let ow = out_dims.x;
    let oh = out_dims.y;
    let x = min(id.x, ow - 1);
    let y = min(id.y, oh - 1);

    let sx = i32(x / 2u);
    let sy = i32(y / 2u);
    let sw = in_dims.x;
    let sh = in_dims.y;
    let sx_clamped = min(sx, sw - 1);
    let sy_clamped = min(sy, sh - 1);

    let flow = textureLoad(in_tex, vec2i(sx_clamped, sy_clamped), 0);

    // Scale flow by 2.0 for the larger resolution
    textureStore(out_tex, vec2i(x, y), vec4(flow.x * 2.0, flow.y * 2.0, 0.0, 0.0));
}
