# GPU Compute Pipeline — Implementation Plan

> **For agentic workers:** Use subagent-driven-development or executing-plans.

**Goal:** Move Farneback optical flow + atlas encoding to wgpu compute shaders, keeping CPU path as fallback.

**Architecture:** A `GpuPipeline` struct (in a new `engine::gpu` module) that takes frames, dispatches compute shaders, returns encoded atlas. CPU path untouched.

**Tech Stack:** wgpu (already present), WGSL compute shaders, `bytemuck` for buffer casting (already in deps).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `crates/motionframe-engine/src/gpu/mod.rs` | **Create** | `GpuPipeline` struct, initialization from existing wgpu device, dispatch orchestration |
| `crates/motionframe-engine/src/gpu/shaders/grayscale.wgsl` | Create | R8G8B8A8 → R8 luminance |
| `crates/motionframe-engine/src/gpu/shaders/pyramid.wgsl` | Create | 2×2 box downsample |
| `crates/motionframe-engine/src/gpu/shaders/poly_expansion.wgsl` | Create | 5×5 separable binomial blur + 6-element polynomial fit per pixel |
| `crates/motionframe-engine/src/gpu/shaders/flow_update.wgsl` | Create | Tensor blur + 2×2 linear solve |
| `crates/motionframe-engine/src/gpu/shaders/upsample.wgsl` | Create | Bilinear upscale + factor scale |
| `crates/motionframe-engine/src/gpu/shaders/encode.wgsl` | Create | f16x2 → R8G8 normalize + quantize |
| `crates/motionframe-engine/src/pipeline/run.rs` | Modify | `run_pipeline` checks for `GpuPipeline`, delegates if available |

---

## What Each Shader Does (one page total)

**grayscale**: `out[x] = dot(in[x].rgb, vec3(0.299, 0.587, 0.114))`

**pyramid**: `out[x,y] = (in[2x,2y] + in[2x+1,2y] + in[2x,2y+1] + in[2x+1,2y+1]) / 4`

**poly_expansion** (two dispatches per level):
1. Horizontal separable blur (5-tap binomial: 1,4,6,4,1 on 6 coeff values)
2. Vertical separable blur
3. Output 2 texels per pixel (6 f16 coeffs: A1[0..2], A2[0..2], b[0..1])

**flow_update** (two dispatches per level):
1. Separable 5×5 blur on tensor elements (A1·A1, A1·A2, A2·A2, A1·b, A2·b)
2. Per-pixel solve: `det = A11*A22 - A12*A12; flow_x = (A22*b1 - A12*b2)/det; flow_y = (A11*b2 - A12*b1)/det`

**upsample**: `out[x,y] = bilinear(in, x/2, y/2) * 2.0`

**encode**: `out.r = (flow.x / max_strength) * 0.5 + 0.5; out.g = (flow.y / max_strength) * 0.5 + 0.5` → quantize to u8

---

## Tasks

### Task 1: Bootstrapping + utility types

- Create `crates/motionframe-engine/src/gpu/mod.rs` with `GpuPipeline` struct containing:
  - `device: Arc<Device>`, `queue: Queue` — accepted from caller (reuse preview device)
  - `bind_group_layouts`, `pipeline_layouts`, `compute_pipelines` for each shader
  - `texture_format: TextureFormat` (R16G16B16A16_FLOAT)
  - `max_workgroup_size: u32` (queried from device limits)
- Add `pub mod gpu;` to `crates/motionframe-engine/src/lib.rs`
- `init()`: creates pipeline layouts, bind group layouts, compute pipelines (one per shader file, loaded with `include_str!`)
- `create_texture(w, h, format, usage)` helper
- `create_buffer(size, usage, contents)` helper

### Task 2: Write all 6 WGSL shaders

Create each shader file in `crates/motionframe-engine/src/gpu/shaders/`. Each shader:
- Takes `@group(0) @binding(0)` input textures (2D), `@group(0) @binding(N)` output storage textures
- Uses `@compute @workgroup_size(16, 16, 1)`
- One `fn main(@builtin(global_invocation_id) id: vec3<u32>)` entry point

### Task 3: Per-frame compute dispatch

Add to `GpuPipeline`:
```rust
pub fn compute_flow(
    &self,
    encoder: &mut CommandEncoder,
    frames: &[&TextureView],    // 2 frames (current, next)
    accum: &TextureView,         // flow accumulator
    out_atlas: &TextureView,     // output R8G8 atlas
) -> Result<(), wgpu::Error>
```

Per frame-pair:
- Create pyramid textures for both frames (reuse pool, allocate max dims once)
- Loop levels coarse → fine:
  1. Dispatch `grayscale` if level 0
  2. Dispatch `pyramid` to build next level
  3. Dispatch `poly_expansion`
  4. If not coarsest: dispatch `flow_update`, `upsample`
  5. If coarsest: dispatch `flow_update` (no upsample needed)
- After loop: accumulate flow into `accum` texture (simple add shader)
- After all pairs: dispatch `encode` on `accum` → `out_atlas`

### Task 4: Wire into pipeline

In `run_pipeline`, add a `gpu: Option<&GpuPipeline>` parameter. If `Some` and frames fit in VRAM budget, delegate to GPU path. Otherwise fall through to CPU.

### Task 5: Wire into desktop app

In `run_convert` in desktop crate, create or receive `GpuPipeline` from the existing wgpu device (created for preview), pass it to `run_pipeline`.

### Task 6: Verify against parity tests

Run `cargo test --test flow_parity` — GPU output must match CPU output within f32 tolerance.

---

## Key Design Decisions

- **No CPU sync until final readback**: All dispatches chained on same encoder
- **Texture pool**: Pre-allocate pyramid textures at max resolution once, reuse per frame pair
- **Shader loading**: `include_str!("shaders/foo.wgsl")` — no build step needed
- **Workgroup size**: 16×16 = 256 invocations, works on all GPUs
- **Fallback**: CPU path untouched; `GpuPipeline::new()` returns `None` if device creation fails
