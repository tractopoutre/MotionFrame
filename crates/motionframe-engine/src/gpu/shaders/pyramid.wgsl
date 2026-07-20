@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<r32float, write>;

// 3-tap Gaussian approximation (sigma=0.5, matches CPU's blur-then-resize at scale=0.5)
// Weights: exp(-0²/(2*0.5²))=1.0, exp(-1²/(2*0.5²))=exp(-2)≈0.135
// Normalized: [0.135/(1+2*0.135), 1.0/(1+2*0.135), ...] = [0.106, 0.788, 0.106]
const W: array<f32, 3> = array(0.106, 0.788, 0.106);

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let in_dims = textureDimensions(in_tex);
    let out_dims = textureDimensions(out_tex);
    let ow = i32(out_dims.x);
    let oh = i32(out_dims.y);
    let iw = i32(in_dims.x);
    let ih = i32(in_dims.y);
    let x = min(i32(id.x), ow - 1);
    let y = min(i32(id.y), oh - 1);

    let sx = 2 * x;
    let sy = 2 * y;

    var sum = 0.0;
    for (var dy = 0i; dy < 3; dy++) {
        let py = clamp(sy + dy - 1, 0, ih - 1);
        let wy = W[dy];
        for (var dx = 0i; dx < 3; dx++) {
            let px = clamp(sx + dx - 1, 0, iw - 1);
            let v = textureLoad(in_tex, vec2(px, py), 0).r;
            sum += v * W[dx] * wy;
        }
    }
    textureStore(out_tex, vec2(x, y), vec4(sum, 0.0, 0.0, 0.0));
}
