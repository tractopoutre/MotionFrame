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

    let sx = (f32(x) + 0.5) * f32(sw) / f32(ow) - 0.5;
    let sy = (f32(y) + 0.5) * f32(sh) / f32(oh) - 0.5;

    let csx = clamp(sx, 0.0, f32(sw - 1));
    let csy = clamp(sy, 0.0, f32(sh - 1));
    let x0 = i32(floor(csx));
    let y0 = i32(floor(csy));
    let x1 = min(x0 + 1, sw - 1);
    let y1 = min(y0 + 1, sh - 1);
    let fx = csx - f32(x0);
    let fy = csy - f32(y0);

    let v00 = textureLoad(in_tex, vec2(x0, y0), 0);
    let v10 = textureLoad(in_tex, vec2(x1, y0), 0);
    let v01 = textureLoad(in_tex, vec2(x0, y1), 0);
    let v11 = textureLoad(in_tex, vec2(x1, y1), 0);

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let flow_x = v00.x * w00 + v10.x * w10 + v01.x * w01 + v11.x * w11;
    let flow_y = v00.y * w00 + v10.y * w10 + v01.y * w01 + v11.y * w11;
    textureStore(out_tex, vec2(x, y), vec4(flow_x * 2.0, flow_y * 2.0, 0.0, 0.0));
}
