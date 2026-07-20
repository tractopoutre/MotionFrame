@group(0) @binding(0) var accum_in: texture_2d<f32>;
@group(0) @binding(1) var pair_flow: texture_2d<f32>;
@group(0) @binding(2) var accum_out: texture_storage_2d<rg32float, write>;

// Catmull-Rom basis (a = -0.5)
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

// Sample flow field at fractional position with Catmull-Rom interpolation.
// Zero-fill border (BORDER_CONSTANT(0)): out-of-frame particles get zero flow.
fn sample_flow(tex: texture_2d<f32>, x: f32, y: f32, w: i32, h: i32) -> vec2<f32> {
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let fx = x - f32(x0);
    let fy = y - f32(y0);

    let wx = catmull_rom(fx);
    let wy = catmull_rom(fy);

    var result = vec2(0.0, 0.0);
    for (var j = 0i; j < 4; j++) {
        let iy = y0 - 1 + j;
        for (var i = 0i; i < 4; i++) {
            let ix = x0 - 1 + i;
            if (ix >= 0 && ix < w && iy >= 0 && iy < h) {
                let val = textureLoad(tex, vec2(ix, iy), 0).xy;
                let wt = wx[i] * wy[j];
                result += val * wt;
            }
        }
    }
    return result;
}

// Heun's method (improved Euler) integration:
//   k1 = flow(x + acc.x, y + acc.y)
//   k2 = flow(x + acc.x + k1.x, y + acc.y + k1.y)
//   acc += 0.5 * (k1 + k2)
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let out_dims = textureDimensions(accum_out);
    let ow = i32(out_dims.x);
    let oh = i32(out_dims.y);
    let x = min(i32(id.x), ow - 1);
    let y = min(i32(id.y), oh - 1);

    let prev = textureLoad(accum_in, vec2(x, y), 0);
    let adx = prev.x;
    let ady = prev.y;

    let sx1 = f32(x) + adx;
    let sy1 = f32(y) + ady;
    let k1 = sample_flow(pair_flow, sx1, sy1, ow, oh);

    let sx2 = sx1 + k1.x;
    let sy2 = sy1 + k1.y;
    let k2 = sample_flow(pair_flow, sx2, sy2, ow, oh);

    let fx = 0.5 * (k1.x + k2.x);
    let fy = 0.5 * (k1.y + k2.y);

    textureStore(accum_out, vec2(x, y), vec4(prev.x + fx, prev.y + fy, 0.0, 0.0));
}
