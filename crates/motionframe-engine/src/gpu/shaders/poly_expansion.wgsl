// Polynomial expansion: computes 5 polynomial coefficients per pixel
// from the grayscale image using 5×5 separable convolution with binomial
// kernel [1,4,6,4,1]/16 and derived difference kernels.
//
// Horizontal pass: convolve with g, xg, xxg → 3 intermediate values
// Vertical pass:   convolve intermediates with g, xg, xxg → 6 b-values
// Combine: Gram matrix inverse → 5 coefficients [r4, r6, r5, r2, r3]
//
// Output: 2 texels per pixel
//   texel0 = (r4, r6, r5, r2)
//   texel1 = (r3, 0, 0, 0)

@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;

const G: array<f32, 5> = array(1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0);
const XG: array<f32, 5> = array(-1.0 / 8.0, -2.0 / 8.0, 0.0 / 8.0, 2.0 / 8.0, 1.0 / 8.0);
const XXG: array<f32, 5> = array(1.0 / 4.0, 1.0 / 4.0, 0.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0);

const IG11: f32 = 1.0;
const IG03: f32 = -2.0 / 3.0;
const IG33: f32 = 2.0 / 3.0;
const IG55: f32 = 1.0;

fn hconv_row(tex: texture_2d<f32>, x: i32, y: i32, iw: i32) -> vec3f {
    var sg = 0.0; var sxg = 0.0; var sxxg = 0.0;
    for (var k = 0i; k < 5; k++) {
        let sx = clamp(x + k - 2, 0, iw - 1);
        let v = textureLoad(tex, vec2(sx, y), 0).r;
        sg += v * G[k];
        sxg += v * XG[k];
        sxxg += v * XXG[k];
    }
    return vec3f(sg, sxg, sxxg);
}

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

    // Compute 3 horizontal convs at 5 rows for the vertical kernel
    var b1 = 0.0; var b2 = 0.0; var b3 = 0.0; var b4 = 0.0; var b5 = 0.0; var b6 = 0.0;
    for (var ky = 0i; ky < 5; ky++) {
        let ry = clamp(oy + ky - 2, 0, ih - 1);
        let hc = hconv_row(in_tex, ox, ry, iw);
        let sg = hc.x; let sxg = hc.y; let sxxg = hc.z;
        b1 += sg * G[ky];
        b2 += sg * XG[ky];
        b4 += sg * XXG[ky];
        b3 += sxg * G[ky];
        b5 += sxxg * G[ky];
        b6 += sxg * XG[ky];
    }

    let r4 = b5 * IG33 + b1 * IG03;
    let r5 = b4 * IG33 + b1 * IG03;
    let r6 = b6 * IG55;
    let r2 = b3 * IG11;
    let r3 = b2 * IG11;

    textureStore(out_tex, vec2(ox * 2, oy), vec4(r4, r6, r5, r2));
    textureStore(out_tex, vec2(ox * 2 + 1, oy), vec4(r3, 0.0, 0.0, 0.0));
}
