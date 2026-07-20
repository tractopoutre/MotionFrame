@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> mode: vec4<f32>;

// mode.x: 0=Nearest, 1=Bilinear, 2=Bicubic(Catmull-Rom)

fn catmull_rom(t: f32) -> array<f32, 4> {
    let t2 = t * t;
    let t3 = t2 * t;
    return array(
        (-0.5) * t3 + (0.5 * t2 - 0.5 * t),
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    );
}

fn sample_bicubic(tex: texture_2d<f32>, x: f32, y: f32, w: i32, h: i32) -> vec4<f32> {
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let wx = catmull_rom(fx);
    let wy = catmull_rom(fy);
    var result = vec4(0.0);
    for (var j = 0i; j < 4; j++) {
        let iy = clamp(y0 - 1 + j, 0, h - 1);
        for (var i = 0i; i < 4; i++) {
            let ix = clamp(x0 - 1 + i, 0, w - 1);
            let v = textureLoad(tex, vec2(ix, iy), 0);
            result += v * (wx[i] * wy[j]);
        }
    }
    return result;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let src_dims = textureDimensions(src_tex);
    let dst_dims = textureDimensions(dst_tex);
    let sw = i32(src_dims.x);
    let sh = i32(src_dims.y);
    let dw = i32(dst_dims.x);
    let dh = i32(dst_dims.y);
    let x = min(i32(id.x), dw - 1);
    let y = min(i32(id.y), dh - 1);
    let interp = i32(mode.x);

    if (interp == 0) {
        // Nearest neighbor
        let sx = i32(round(f32(x) * f32(sw) / f32(dw)));
        let sy = i32(round(f32(y) * f32(sh) / f32(dh)));
        let px = clamp(sx, 0, sw - 1);
        let py = clamp(sy, 0, sh - 1);
        textureStore(dst_tex, vec2(x, y), textureLoad(src_tex, vec2(px, py), 0));
        return;
    }

    let sx = (f32(x) + 0.5) * f32(sw) / f32(dw) - 0.5;
    let sy = (f32(y) + 0.5) * f32(sh) / f32(dh) - 0.5;

    if (interp == 1) {
        // Bilinear
        let x0 = max(i32(floor(sx)), 0);
        let y0 = max(i32(floor(sy)), 0);
        let x1 = min(x0 + 1, sw - 1);
        let y1 = min(y0 + 1, sh - 1);
        let fx = sx - f32(x0);
        let fy = sy - f32(y0);
        let c00 = textureLoad(src_tex, vec2(x0, y0), 0);
        let c10 = textureLoad(src_tex, vec2(x1, y0), 0);
        let c01 = textureLoad(src_tex, vec2(x0, y1), 0);
        let c11 = textureLoad(src_tex, vec2(x1, y1), 0);
        let w00 = (1.0 - fx) * (1.0 - fy);
        let w10 = fx * (1.0 - fy);
        let w01 = (1.0 - fx) * fy;
        let w11 = fx * fy;
        textureStore(dst_tex, vec2(x, y), c00 * w00 + c10 * w10 + c01 * w01 + c11 * w11);
    } else {
        // Bicubic (Catmull-Rom)
        let c = sample_bicubic(src_tex, sx, sy, sw, sh);
        textureStore(dst_tex, vec2(x, y), c);
    }
}
