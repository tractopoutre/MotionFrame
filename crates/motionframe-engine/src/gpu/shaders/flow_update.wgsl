// Flow update: builds 5 tensor elements from two polynomial expansions,
// applies 5×5 binomial blur, solves the 2×2 linear system.
//
// Each thread reads the 5×5 neighborhood of poly coefficients, computes
// per-pixel tensor elements, accumulates with binomial weights, and solves.
// Prior flow warps frame B's polynomial expansion.
//
// binding 0: poly_a (texture_2d<f32>, RGBA32F, 2 texels per pixel)
// binding 1: poly_b (texture_2d<f32>, RGBA32F, 2 texels per pixel)
// binding 2: prior_flow (texture_2d<f32>, RG32F)
// binding 3: params (uniform vec4) — x = winsize, y = scale_factor
// binding 4: out_tex (texture_storage_2d<rg32float, write>)

@group(0) @binding(0) var poly_a_tex: texture_2d<f32>;
@group(0) @binding(1) var poly_b_tex: texture_2d<f32>;
@group(0) @binding(2) var prior_flow_tex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: vec4<f32>;
@group(0) @binding(4) var out_tex: texture_storage_2d<rg32float, write>;

const BINOM: array<f32, 5> = array(1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0);

struct Poly5 {
    r4: f32, r6: f32, r5: f32, r2: f32, r3: f32,
}

fn load_poly(tex: texture_2d<f32>, x: i32, y: i32) -> Poly5 {
    let t0 = textureLoad(tex, vec2(x * 2, y), 0);
    let t1 = textureLoad(tex, vec2(x * 2 + 1, y), 0);
    return Poly5(t0.x, t0.y, t0.z, t0.w, t1.x);
}

fn sample_poly_bilinear(tex: texture_2d<f32>, u: f32, v: f32, w: i32, h: i32) -> Poly5 {
    let fu = clamp(u, 0.0, f32(w - 1));
    let fv = clamp(v, 0.0, f32(h - 1));
    let x0 = i32(floor(fu));
    let y0 = i32(floor(fv));
    let x1 = min(x0 + 1, w - 1);
    let y1 = min(y0 + 1, h - 1);
    let ax = fu - f32(x0);
    let ay = fv - f32(y0);

    let w00 = (1.0 - ax) * (1.0 - ay);
    let w10 = ax * (1.0 - ay);
    let w01 = (1.0 - ax) * ay;
    let w11 = ax * ay;

    let p00 = load_poly(tex, x0, y0);
    let p10 = load_poly(tex, x1, y0);
    let p01 = load_poly(tex, x0, y1);
    let p11 = load_poly(tex, x1, y1);

    return Poly5(
        p00.r4 * w00 + p10.r4 * w10 + p01.r4 * w01 + p11.r4 * w11,
        p00.r6 * w00 + p10.r6 * w10 + p01.r6 * w01 + p11.r6 * w11,
        p00.r5 * w00 + p10.r5 * w10 + p01.r5 * w01 + p11.r5 * w11,
        p00.r2 * w00 + p10.r2 * w10 + p01.r2 * w01 + p11.r2 * w11,
        p00.r3 * w00 + p10.r3 * w10 + p01.r3 * w01 + p11.r3 * w11,
    );
}

struct Tensors {
    t0: f32, t1: f32, t2: f32, t3: f32, t4: f32,
}

fn build_tensors(pa: Poly5, pb: Poly5, dx: f32, dy: f32) -> Tensors {
    let a_val = (pa.r4 + pb.r4) * 0.5;
    let b_val = (pa.r6 + pb.r6) * 0.25;
    let c_val = (pa.r5 + pb.r5) * 0.5;
    let d_val = (pb.r2 - pa.r2) * 0.5 + a_val * dx + b_val * dy;
    let e_val = (pb.r3 - pa.r3) * 0.5 + b_val * dx + c_val * dy;
    return Tensors(
        a_val * a_val + b_val * b_val,
        b_val * a_val + b_val * c_val,
        b_val * b_val + c_val * c_val,
        a_val * d_val + b_val * e_val,
        b_val * d_val + c_val * e_val,
    );
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(poly_a_tex);
    let pw = i32(dims.x) / 2;
    let ph = i32(dims.y);
    let x = min(i32(id.x), pw - 1);
    let y = min(i32(id.y), ph - 1);
    if (x < 0 || y < 0) {
        return;
    }

    // Read prior flow at this pixel
    let prior_dims = textureDimensions(prior_flow_tex);
    let prior = textureLoad(prior_flow_tex, vec2(min(x, i32(prior_dims.x) - 1), min(y, i32(prior_dims.y) - 1)), 0);
    let dx = prior.x;
    let dy = prior.y;

    // 5×5 binomial blur on tensor elements built from the neighborhood
    var t0 = 0.0; var t1 = 0.0; var t2 = 0.0; var t3 = 0.0; var t4 = 0.0;

    for (var ky = 0i; ky < 5; ky++) {
        let wy = BINOM[ky];
        for (var kx = 0i; kx < 5; kx++) {
            let wx = BINOM[kx];
            let w = wx * wy;

            let nx = clamp(x + kx - 2, 0, pw - 1);
            let ny = clamp(y + ky - 2, 0, ph - 1);

            let pa = load_poly(poly_a_tex, nx, ny);
            let pb = sample_poly_bilinear(poly_b_tex, f32(nx) + dx, f32(ny) + dy, pw, ph);
            let tens = build_tensors(pa, pb, dx, dy);

            t0 += tens.t0 * w;
            t1 += tens.t1 * w;
            t2 += tens.t2 * w;
            t3 += tens.t3 * w;
            t4 += tens.t4 * w;
        }
    }

    // Solve 2×2 system
    let scale_factor = params.y;
    // Binomial kernel is already normalized (sum = 1.0), no additional scaling needed

    let g11_s = t0;
    let g12_s = t1;
    let g22_s = t2;
    let h1_s = t3;
    let h2_s = t4;

    let det = g11_s * g22_s - g12_s * g12_s + 1e-3;
    let flow_x = (g22_s * h1_s - g12_s * h2_s) / det;
    let flow_y = (g11_s * h2_s - g12_s * h1_s) / det;

    textureStore(out_tex, vec2(x, y), vec4(flow_x * scale_factor, flow_y * scale_factor, 0.0, 0.0));
}
