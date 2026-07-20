@group(0) @binding(0) var base_tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: vec4<f32>;
@group(0) @binding(2) var out_tex: texture_storage_2d<r32float, write>;

// params: (scale_x, scale_y, sigma, half_ksize)

fn gauss(x: f32, sigma: f32) -> f32 {
    return exp(-0.5 * x * x / (sigma * sigma));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let base_dims = textureDimensions(base_tex);
    let out_dims = textureDimensions(out_tex);
    let bw = i32(base_dims.x);
    let bh = i32(base_dims.y);
    let ow = i32(out_dims.x);
    let oh = i32(out_dims.y);
    let x = min(i32(id.x), ow - 1);
    let y = min(i32(id.y), oh - 1);

    let scale_x = params.x;
    let scale_y = params.y;
    let sigma = params.z;
    let half = i32(params.w);

    // Center in the original base image
    let cx = (f32(x) + 0.5) * scale_x - 0.5;
    let cy = (f32(y) + 0.5) * scale_y - 0.5;

    let inv_sig2 = 1.0 / (2.0 * sigma * sigma);
    var sum = 0.0;
    var wsum = 0.0;
    for (var dy = -half; dy <= half; dy++) {
        let py = clamp(i32(round(cy)) + dy, 0, bh - 1);
        for (var dx = -half; dx <= half; dx++) {
            let px = clamp(i32(round(cx)) + dx, 0, bw - 1);
            let v = textureLoad(base_tex, vec2(px, py), 0).r;
            let w = gauss(f32(dx), sigma) * gauss(f32(dy), sigma);
            sum += v * w;
            wsum += w;
        }
    }
    textureStore(out_tex, vec2(x, y), vec4(sum / wsum, 0.0, 0.0, 0.0));
}
