@group(0) @binding(0) var fwd_flow: texture_2d<f32>;
@group(0) @binding(1) var bwd_flow: texture_2d<f32>;
@group(0) @binding(2) var out_flow: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let out_dims = textureDimensions(out_flow);
    let ow = i32(out_dims.x);
    let oh = i32(out_dims.y);
    let x = min(i32(id.x), ow - 1);
    let y = min(i32(id.y), oh - 1);

    let f = textureLoad(fwd_flow, vec2(x, y), 0);
    let b = textureLoad(bwd_flow, vec2(x, y), 0);
    let combined = 0.5 * (f.xy - b.xy);
    textureStore(out_flow, vec2(x, y), vec4(combined.x, combined.y, 0.0, 0.0));
}
