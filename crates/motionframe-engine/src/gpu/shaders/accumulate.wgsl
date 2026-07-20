@group(0) @binding(0) var accum_in: texture_2d<f32>;
@group(0) @binding(1) var pair_flow: texture_2d<f32>;
@group(0) @binding(2) var accum_out: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let out_dims = textureDimensions(accum_out);
    let ow = i32(out_dims.x);
    let oh = i32(out_dims.y);
    let x = min(i32(id.x), ow - 1);
    let y = min(i32(id.y), oh - 1);

    let prev = textureLoad(accum_in, vec2(x, y), 0);
    let flow = textureLoad(pair_flow, vec2(x, y), 0);
    textureStore(accum_out, vec2(x, y), vec4(prev.x + flow.x, prev.y + flow.y, 0.0, 0.0));
}
