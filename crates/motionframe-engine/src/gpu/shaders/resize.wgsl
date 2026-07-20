@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: vec4<f32>;

// params: (src_w, src_h, dst_w, dst_h)

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dst_dims = textureDimensions(dst_tex);
    let dw = i32(dst_dims.x);
    let dh = i32(dst_dims.y);
    let x = min(i32(id.x), dw - 1);
    let y = min(i32(id.y), dh - 1);

    let src_w = i32(params.x);
    let src_h = i32(params.y);
    let dst_w = i32(params.z);
    let dst_h = i32(params.w);

    let sx = (f32(x) + 0.5) * f32(src_w) / f32(dst_w) - 0.5;
    let sy = (f32(y) + 0.5) * f32(src_h) / f32(dst_h) - 0.5;

    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));
    let x1 = min(x0 + 1, src_w - 1);
    let y1 = min(y0 + 1, src_h - 1);
    let fx = sx - f32(x0);
    let fy = sy - f32(y0);

    let c00 = textureLoad(src_tex, vec2(max(x0, 0), max(y0, 0)), 0);
    let c10 = textureLoad(src_tex, vec2(x1, max(y0, 0)), 0);
    let c01 = textureLoad(src_tex, vec2(max(x0, 0), y1), 0);
    let c11 = textureLoad(src_tex, vec2(x1, y1), 0);

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let r = c00.x * w00 + c10.x * w10 + c01.x * w01 + c11.x * w11;
    let g = c00.y * w00 + c10.y * w10 + c01.y * w01 + c11.y * w11;
    let b = c00.z * w00 + c10.z * w10 + c01.z * w01 + c11.z * w11;
    let a = c00.w * w00 + c10.w * w10 + c01.w * w01 + c11.w * w11;

    textureStore(dst_tex, vec2(x, y), vec4(r, g, b, a));
}
