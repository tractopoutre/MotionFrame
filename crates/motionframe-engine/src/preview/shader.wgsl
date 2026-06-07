struct Uniforms {
    atlas_grid: vec2<u32>,    // (cols, rows) of the COLOR atlas tile grid
    frame_count: u32,
    motion_strength: f32,
    time: f32,                // fractional frame index
    stagger_pack: u32,        // 0 = normal, 1 = stagger-packed
    mv_encoding: u32,         // 0 = R8G8 remap, 1 = SideFX Labs polar
    bg_mode: u32,             // 0 = black, 1 = gray, 2 = white, 3 = checker
    premultiplied_alpha: u32, // 0 = straight alpha, 1 = premultiplied
    blend_mode: u32,          // 0 = motion-vector blend, 1 = cross-fade only
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var mv_tex:    texture_2d<f32>;
@group(0) @binding(3) var samp:      sampler;     // bilinear
@group(0) @binding(4) var samp_n:    sampler;     // nearest — required for SideFX polar

// Texture UV for cell `frame_idx` of a (grid.x × grid.y) tile grid, given an
// in-cell uv ∈ [0,1].
fn cell_uv(uv: vec2<f32>, frame_idx: u32, grid: vec2<u32>) -> vec2<f32> {
    let cell = vec2<f32>(1.0) / vec2<f32>(grid);
    let col = f32(frame_idx % grid.x);
    let row = f32(frame_idx / grid.x);
    return (vec2<f32>(col, row) + uv) * cell;
}

// SideFX Labs R8G8 polar decode. Mirrors `SideFx_DecodeMotionVector_float`
// from the Unity reference: high bit of G is a polar flip, low 7 bits are
// magnitude, R is the low 8 bits of the [0,511] polar angle.
fn decode_sidefx(encoded: vec2<f32>) -> vec2<f32> {
    let mapped = round(encoded * 255.0);
    let polar_flip_bit = floor(mapped.y / 128.0);
    let magnitude_bits = mapped.y - polar_flip_bit * 128.0;
    let encoded_polar = mapped.x + polar_flip_bit * 256.0;
    let polar = (encoded_polar / 511.0) * 6.28318530718;
    let magnitude = magnitude_bits / 127.0;
    return vec2<f32>(cos(polar) * magnitude, sin(polar) * magnitude);
}

// Fetch the encoded motion vector for cell `i`, decode it (R8G8 remap or
// SideFX polar), apply motion_strength, and Y-flip into shader UV convention
// (V-down). Returns the displacement to apply to UVs at this fragment.
fn fetch_decoded_mv(uv: vec2<f32>, i: u32, mv_grid: vec2<u32>) -> vec2<f32> {
    var raw: vec2<f32>;
    if (u.stagger_pack == 1u) {
        let stagger_idx = i / 2u;
        let stagger_uv  = cell_uv(uv, stagger_idx, mv_grid);
        var sample4: vec4<f32>;
        if (u.mv_encoding == 1u) {
            sample4 = textureSample(mv_tex, samp_n, stagger_uv);
        } else {
            sample4 = textureSample(mv_tex, samp, stagger_uv);
        }
        if (i % 2u == 0u) {
            raw = sample4.bg;
        } else {
            raw = sample4.ra;
        }
    } else {
        let mv_uv = cell_uv(uv, i, mv_grid);
        if (u.mv_encoding == 1u) {
            raw = textureSample(mv_tex, samp_n, mv_uv).rg;
        } else {
            raw = textureSample(mv_tex, samp, mv_uv).rg;
        }
    }
    var mv: vec2<f32>;
    if (u.mv_encoding == 1u) {
        mv = decode_sidefx(raw) * u.motion_strength;
    } else {
        mv = (raw - vec2<f32>(0.5)) * 2.0 * u.motion_strength;
    }
    // Encoder Y-flips for the on-disk format (game-engine convention, +V = up).
    // WGSL UV is V-down, so undo for the preview.
    mv.y = -mv.y;
    return mv;
}

// Procedural background sampled at the preview's UV. Returns opaque RGBA.
// uv is in [0,1] across the preview rect, so the checker is roughly square.
fn bg_color(uv: vec2<f32>) -> vec3<f32> {
    if (u.bg_mode == 0u) { return vec3<f32>(0.0); }
    if (u.bg_mode == 1u) { return vec3<f32>(0.5); }
    if (u.bg_mode == 2u) { return vec3<f32>(1.0); }
    // Checker: 16 squares across; alternate two grays so transparency is
    // obvious without overpowering the foreground.
    let cell = floor(uv * 16.0);
    let parity = (i32(cell.x) + i32(cell.y)) & 1;
    if (parity == 0) { return vec3<f32>(0.8); }
    return vec3<f32>(1.0);
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) % 2) * 4.0 - 1.0;
    let uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let t_floor = floor(u.time);
    let t_frac  = fract(u.time);
    let i0 = u32(t_floor) % u.frame_count;
    let i1 = (i0 + 1u) % u.frame_count;

    // Stagger packing fits two source tiles into one packed pixel, so the
    // motion-atlas grid is (cols, ceil(rows/2)). The color atlas is unaffected.
    var mv_grid = u.atlas_grid;
    if (u.stagger_pack == 1u) {
        mv_grid.y = (u.atlas_grid.y + 1u) / 2u;
    }

    // blend_mode == 1 forces the warp to zero so the preview is a pure
    // cross-fade between the two source frames — useful for A/B-ing what the
    // motion vectors are buying versus the naive baseline.
    var mv: vec2<f32>;
    if (u.blend_mode == 1u) {
        mv = vec2<f32>(0.0);
    } else {
        mv = fetch_decoded_mv(uv, i0, mv_grid);
    }

    // Saturate so warped UVs stay inside their atlas cell — `cell_uv` only
    // offsets by (col, row) and won't clamp at cell boundaries.
    let uv_curr = saturate(uv - mv * t_frac);
    let uv_next = saturate(uv + mv * (1.0 - t_frac));
    let c0 = textureSample(color_tex, samp, cell_uv(uv_curr, i0, u.atlas_grid));
    let c1 = textureSample(color_tex, samp, cell_uv(uv_next, i1, u.atlas_grid));
    let comp = mix(c0, c1, t_frac);
    // Composite over the chosen background. Formula depends on how the
    // encoder stored the atlas: premultiplied rgb is already scaled by alpha,
    // straight rgb is not. Mismatched math darkens or brightens edge pixels.
    let bg = bg_color(uv);
    var out_rgb: vec3<f32>;
    if (u.premultiplied_alpha == 1u) {
        out_rgb = bg * (1.0 - comp.a) + comp.rgb;
    } else {
        out_rgb = mix(bg, comp.rgb, comp.a);
    }
    return vec4<f32>(out_rgb, 1.0);
}
