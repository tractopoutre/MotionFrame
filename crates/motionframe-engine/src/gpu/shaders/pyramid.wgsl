@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let in_dims = textureDimensions(in_tex);
    let out_dims = textureDimensions(out_tex);
    let ow = out_dims.x;
    let oh = out_dims.y;
    let x = min(id.x, ow - 1);
    let y = min(id.y, oh - 1);

    let sx = x * 2u;
    let sy = y * 2u;
    let iw = in_dims.x;
    let ih = in_dims.y;

    let c00 = textureLoad(in_tex, vec2i(min(sx, iw - 1), min(sy, ih - 1)), 0).r;
    let c10 = textureLoad(in_tex, vec2i(min(sx + 1, iw - 1), min(sy, ih - 1)), 0).r;
    let c01 = textureLoad(in_tex, vec2i(min(sx, iw - 1), min(sy + 1, ih - 1)), 0).r;
    let c11 = textureLoad(in_tex, vec2i(min(sx + 1, iw - 1), min(sy + 1, ih - 1)), 0).r;

    let avg = (c00 + c10 + c01 + c11) * 0.25;
    textureStore(out_tex, vec2i(x, y), vec4(avg, 0.0, 0.0, 0.0));
}
