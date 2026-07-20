@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: vec4<f32>;
@group(0) @binding(3) var<uniform> params2: vec4<f32>;
@group(0) @binding(4) var<storage, read> kernel_data: array<f32>;

// params: (kernel_size, ig11, ig03, ig33)
// params2: (ig55, _, _, _)
// kernel_data: g[0..ksize], xg[0..ksize], xxg[0..ksize]

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(in_tex);
    let iw = i32(dims.x);
    let ih = i32(dims.y);
    let ox = i32(id.x);
    let oy = i32(id.y);
    if (ox >= iw || oy >= ih) {
        return;
    }

    let ksize = i32(params.x);
    let half = ksize / 2;
    let ig11 = params.y;
    let ig03 = params.z;
    let ig33 = params.w;
    let ig55 = params2.x;

    var b1 = 0.0; var b2 = 0.0; var b3 = 0.0; var b4 = 0.0; var b5 = 0.0; var b6 = 0.0;

    for (var ky = 0i; ky < ksize; ky++) {
        let ry = clamp(oy + ky - half, 0, ih - 1);

        var sg = 0.0; var sxg = 0.0; var sxxg = 0.0;
        for (var kx = 0i; kx < ksize; kx++) {
            let rx = clamp(ox + kx - half, 0, iw - 1);
            let v = textureLoad(in_tex, vec2(rx, ry), 0).r;
            sg += v * kernel_data[kx];
            sxg += v * kernel_data[ksize + kx];
            sxxg += v * kernel_data[2 * ksize + kx];
        }

        b1 += sg * kernel_data[ky];
        b2 += sg * kernel_data[ksize + ky];
        b4 += sg * kernel_data[2 * ksize + ky];
        b3 += sxg * kernel_data[ky];
        b5 += sxxg * kernel_data[ky];
        b6 += sxg * kernel_data[ksize + ky];
    }

    let r4 = b5 * ig33 + b1 * ig03;
    let r5 = b4 * ig33 + b1 * ig03;
    let r6 = b6 * ig55;
    let r2 = b3 * ig11;
    let r3 = b2 * ig11;

    textureStore(out_tex, vec2(ox * 2, oy), vec4(r4, r6, r5, r2));
    textureStore(out_tex, vec2(ox * 2 + 1, oy), vec4(r3, 0.0, 0.0, 0.0));
}
