use std::sync::Arc;

use rayon::prelude::*;

use crate::pipeline::{Flow, GenerateOptions, ImageRgba8};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType, BufferDescriptor,
    BufferUsages, CommandEncoder, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, DeviceDescriptor, Extent3d, Instance, InstanceDescriptor,
    MapMode, Origin3d, PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor,
    PowerPreference, Queue, RequestAdapterOptions, ShaderModule, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, StorageTextureAccess, TexelCopyBufferInfo, TexelCopyBufferLayout,
    TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
    TextureViewDimension,
};

/// GPU-accelerated Farneback optical flow pipeline.
///
/// Owns a wgpu device, queue, and all compute pipeline state. Call
/// `compute()` to run the full coarse-to-fine flow estimation on a
/// slice of RGBA8 frames and produce encoded color + motion atlases.
///
/// Batch optimizations (all in a single encoder submission):
/// - All frame pairs processed without GPU readback between them
/// - Forward + backward + bidirectional combine per pair on GPU
/// - Accumulation stays on GPU; single readback at the very end
/// - Pre-allocated buffers avoid per-call allocation overhead
// allow(dead_code): encode/bgl_encode kept for API completeness
#[allow(dead_code)]
pub struct GpuPipeline {
    device: Arc<Device>,
    queue: Queue,
    // Compute pipelines
    grayscale: ComputePipeline,
    pyramid_blur: ComputePipeline,
    poly_expansion: ComputePipeline,
    flow_update: ComputePipeline,
    upsample: ComputePipeline,
    encode: ComputePipeline,
    resize: ComputePipeline,
    accumulate: ComputePipeline,
    combine_bidir: ComputePipeline,
    // Bind group layouts
    bgl_tex_in_out: BindGroupLayout,
    bgl_pyramid: BindGroupLayout,
    bgl_poly: BindGroupLayout,
    bgl_tex_in_out_rg: BindGroupLayout,
    bgl_flow_update: BindGroupLayout,
    bgl_encode: BindGroupLayout,
    bgl_accum_combine: BindGroupLayout,
    bgl_resize: BindGroupLayout,
    zero_prior: Texture,
}

impl GpuPipeline {
    /// Attempt to initialize a GPU pipeline with a new wgpu device.
    /// Returns `None` if no suitable GPU adapter is available.
    pub fn try_init() -> Option<Self> {
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            ..RequestAdapterOptions::default()
        }))
        .ok()?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&DeviceDescriptor::default())).ok()?;
        Some(Self::new(Arc::new(device), queue))
    }

    /// Create a new GPU pipeline from an existing wgpu device + queue.
    // allow(too_many_lines): initialization is verbose but sequential
    #[allow(clippy::too_many_lines)]
    pub fn new(device: Arc<Device>, queue: Queue) -> Self {
        // --- Bind group layouts ---

        // 1 sampled texture + 1 R32Float storage texture (grayscale, pyramid)
        let bgl_tex_in_out = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bgl_tex_in_out"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // 1 sampled texture + 1 uniform + 1 R32Float storage (pyramid blur from original)
        let bgl_pyramid = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bgl_pyramid"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // 1 sampled texture + 1 RGBA32F storage + 2 uniforms + 1 storage buffer (poly_expansion)
        let bgl_poly = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bgl_poly"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // 1 sampled texture + 1 RG32Float storage texture (upsample)
        let bgl_tex_in_out_rg = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bgl_tex_in_out_rg"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rg32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // 3 sampled textures + 1 uniform + 1 RG32Float storage (flow_update)
        let bgl_flow_update = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bgl_flow_update"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rg32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // 1 sampled texture + 1 uniform + 1 RG32Float storage (encode)
        let bgl_encode = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bgl_encode"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rg32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // 1 sampled RGBA8 texture + 1 RGBA8 storage (resize)
        let bgl_resize = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bgl_resize"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // 2 sampled RG32Float textures + 1 RG32Float storage (accumulate, combine)
        let bgl_accum_combine = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bgl_accum_combine"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rg32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // --- Pipeline layouts ---
        let pl_1ch = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pl_1ch"),
            bind_group_layouts: &[Some(&bgl_tex_in_out)],
            immediate_size: 0,
        });
        let pl_pyramid = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pl_pyramid"),
            bind_group_layouts: &[Some(&bgl_pyramid)],
            immediate_size: 0,
        });
        let pl_poly = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pl_poly"),
            bind_group_layouts: &[Some(&bgl_poly)],
            immediate_size: 0,
        });
        let pl_rg = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pl_rg"),
            bind_group_layouts: &[Some(&bgl_tex_in_out_rg)],
            immediate_size: 0,
        });
        let pl_flow = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pl_flow"),
            bind_group_layouts: &[Some(&bgl_flow_update)],
            immediate_size: 0,
        });
        let pl_encode = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pl_encode"),
            bind_group_layouts: &[Some(&bgl_encode)],
            immediate_size: 0,
        });
        let pl_accum_combine = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pl_accum_combine"),
            bind_group_layouts: &[Some(&bgl_accum_combine)],
            immediate_size: 0,
        });
        let pl_resize = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pl_resize"),
            bind_group_layouts: &[Some(&bgl_resize)],
            immediate_size: 0,
        });

        // --- Shader modules ---
        let mod_grayscale = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("grayscale"),
            source: ShaderSource::Wgsl(include_str!("shaders/grayscale.wgsl").into()),
        });
        let mod_pyramid = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("pyramid"),
            source: ShaderSource::Wgsl(include_str!("shaders/pyramid.wgsl").into()),
        });
        let mod_poly = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("poly_expansion"),
            source: ShaderSource::Wgsl(include_str!("shaders/poly_expansion.wgsl").into()),
        });
        let mod_flow = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("flow_update"),
            source: ShaderSource::Wgsl(include_str!("shaders/flow_update.wgsl").into()),
        });
        let mod_upsample = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("upsample"),
            source: ShaderSource::Wgsl(include_str!("shaders/upsample.wgsl").into()),
        });
        let mod_encode = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("encode"),
            source: ShaderSource::Wgsl(include_str!("shaders/encode.wgsl").into()),
        });
        let mod_resize = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("resize"),
            source: ShaderSource::Wgsl(include_str!("shaders/resize.wgsl").into()),
        });
        let mod_accum = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("accumulate"),
            source: ShaderSource::Wgsl(include_str!("shaders/accumulate.wgsl").into()),
        });
        let mod_combine = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("combine_bidir"),
            source: ShaderSource::Wgsl(include_str!("shaders/combine_bidir.wgsl").into()),
        });

        // --- Compute pipelines ---
        let grayscale = Self::make_pipeline(&device, &pl_1ch, &mod_grayscale, "grayscale");
        let pyramid_blur = Self::make_pipeline(&device, &pl_pyramid, &mod_pyramid, "pyramid_blur");
        let poly_expansion = Self::make_pipeline(&device, &pl_poly, &mod_poly, "poly_expansion");
        let flow_update = Self::make_pipeline(&device, &pl_flow, &mod_flow, "flow_update");
        let upsample = Self::make_pipeline(&device, &pl_rg, &mod_upsample, "upsample");
        let encode = Self::make_pipeline(&device, &pl_encode, &mod_encode, "encode");
        let resize = Self::make_pipeline(&device, &pl_resize, &mod_resize, "resize");
        let accumulate = Self::make_pipeline(&device, &pl_accum_combine, &mod_accum, "accumulate");
        let combine_bidir =
            Self::make_pipeline(&device, &pl_accum_combine, &mod_combine, "combine_bidir");

        // --- Pre-allocated zero-prior texture (1x1 Rg32Float, zero-filled) ---
        let zero_prior = device.create_texture(&TextureDescriptor {
            label: Some("zero_prior"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rg32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &zero_prior,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &[0u8; 8],
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        Self {
            device,
            queue,
            grayscale,
            pyramid_blur,
            poly_expansion,
            flow_update,
            upsample,
            encode,
            resize,
            accumulate,
            combine_bidir,
            bgl_tex_in_out,
            bgl_pyramid,
            bgl_poly,
            bgl_tex_in_out_rg,
            bgl_flow_update,
            bgl_encode,
            bgl_accum_combine,
            bgl_resize,
            zero_prior,
        }
    }

    fn make_pipeline(
        device: &Device,
        layout: &PipelineLayout,
        module: &ShaderModule,
        label: &str,
    ) -> ComputePipeline {
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    fn tex_view(tex: &Texture) -> TextureView {
        tex.create_view(&TextureViewDescriptor::default())
    }

    // allow(unused_self): called with method syntax for consistency
    #[allow(clippy::unused_self)]
    fn dispatch_16(
        &self,
        encoder: &mut CommandEncoder,
        pipeline: &ComputePipeline,
        bind_group: &BindGroup,
        w: u32,
        h: u32,
    ) {
        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
    }

    /// Compute optical flow for one frame pair (forward direction only).
    ///
    /// Returns the finest-level flow texture and its dimensions.
    /// All work is recorded into `encoder`; no queue submission or readback.
    // allow(unused_self): used via method calls on self
    #[allow(clippy::unused_self, clippy::too_many_arguments)]
    fn compute_pair_flow(
        &self,
        encoder: &mut CommandEncoder,
        gray_a: &Texture,
        gray_b: &Texture,
        poly_n: u32,
        poly_sigma: f32,
        winsize: u32,
        use_gaussian: bool,
        iterations: u32,
    ) -> (Texture, u32, u32) {
        let pyr_a = self.pyramid_all(encoder, gray_a);
        let pyr_b = self.pyramid_all(encoder, gray_b);
        let num_levels = pyr_a.len();
        if num_levels == 0 {
            let (gw, gh) = (gray_a.width(), gray_a.height());
            let tex = self.make_tex(
                gw,
                gh,
                TextureFormat::Rg32Float,
                TextureUsages::STORAGE_BINDING
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC,
                "zero_flow",
            );
            return (tex, gw, gh);
        }

        let mut prior_flow: Option<(Texture, u32, u32)> = None;

        for level in 0..num_levels {
            let (_, lw, lh) = pyr_a[level];

            let pa = self.poly_expand(encoder, &pyr_a[level], poly_n, poly_sigma);
            let pb = self.poly_expand(encoder, &pyr_b[level], poly_n, poly_sigma);

            let init_flow: Option<Texture> = if level == 0 {
                None
            } else if let Some((ref prev_tex, ..)) = prior_flow.as_ref() {
                let up = self.make_tex(
                    lw,
                    lh,
                    TextureFormat::Rg32Float,
                    TextureUsages::STORAGE_BINDING
                        | TextureUsages::TEXTURE_BINDING
                        | TextureUsages::COPY_SRC,
                    "up_flow",
                );
                let in_view = Self::tex_view(prev_tex);
                let out_view = Self::tex_view(&up);
                let up_bg = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("bg_upsample"),
                    layout: &self.bgl_tex_in_out_rg,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::TextureView(&in_view),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::TextureView(&out_view),
                        },
                    ],
                });
                self.dispatch_16(encoder, &self.upsample, &up_bg, lw, lh);
                Some(up)
            } else {
                None
            };

            let mut cur_flow = init_flow;
            let num_iters = iterations.max(1);
            for _ in 0..num_iters {
                cur_flow = Some(self.flow_update(
                    encoder,
                    &pa,
                    &pb,
                    cur_flow.as_ref(),
                    lw,
                    lh,
                    winsize,
                    use_gaussian,
                ));
            }
            prior_flow = cur_flow.map(|f| (f, lw, lh));
        }

        prior_flow.unwrap_or_else(|| {
            let (_, lw, lh) = pyr_a[0];
            let tex = self.make_tex(
                lw,
                lh,
                TextureFormat::Rg32Float,
                TextureUsages::STORAGE_BINDING
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC,
                "zero_flow",
            );
            (tex, lw, lh)
        })
    }

    /// Run flow computation for all batches in a SINGLE GPU submission.
    ///
    /// Uploads frames once, processes every batch (forward+backward+accumulate)
    /// in one encoder, records copy-to-staging for all outputs, submits once,
    /// polls once, and returns all flows.
    ///
    /// `frames`: original-resolution premultiplied RGBA frames.
    /// `batches`: frame index groups, one per output flow.
    /// `tail_batch`: optional additional batch for loop wrap (first+last frames).
    #[allow(clippy::too_many_lines, clippy::cast_lossless)]
    pub fn compute_batch_flows(
        &self,
        frames: &[ImageRgba8],
        batches: &[Vec<usize>],
        tail_batch: Option<&[usize]>,
        options: &GenerateOptions,
    ) -> Vec<Flow> {
        if frames.is_empty() || batches.is_empty() {
            return Vec::new();
        }

        let (atlas_cols, atlas_rows) = options.atlas_dims;
        let src_aspect = frames[0].width as f64 / frames[0].height as f64;
        let (tile_w, tile_h) = crate::pipeline::atlas_layout::compute_tile_dims(
            options.atlas_resolution,
            atlas_cols,
            atlas_rows,
            src_aspect,
        );

        // Collect all unique frame indices needed across all batches + tail
        let mut needed: Vec<usize> = batches.iter().flat_map(|b| b.iter().copied()).collect();
        if let Some(tail) = tail_batch {
            needed.extend_from_slice(tail);
        }
        needed.sort_unstable();
        needed.dedup();

        // Upload only the needed frames
        let orig_texs: Vec<(usize, Texture)> = needed
            .iter()
            .map(|&i| (i, self.upload_frame(&frames[i])))
            .collect();

        let flow_w = tile_w;
        let flow_h = tile_h;

        // Single encoder for ALL batch work
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        // GPU resize to tile size + grayscale for each needed frame
        let mut gray_map: Vec<(usize, Texture)> = Vec::with_capacity(orig_texs.len());
        for &(idx, ref tex) in &orig_texs {
            let interp = match options.resize_algorithm {
                crate::pipeline::Interpolation::Nearest => 0u32,
                crate::pipeline::Interpolation::Linear => 1,
                _ => 2,
            };
            let resized = self.resize_tex(&mut encoder, tex, tile_w.max(1), tile_h.max(1), interp);
            let gray = self.grayscale(&mut encoder, &resized);
            gray_map.push((idx, gray));
        }

        let gray_of = |idx: usize| -> &Texture {
            let entry = gray_map.iter().find(|(i, _)| *i == idx).unwrap();
            &entry.1
        };

        // Process each batch: forward+backward+combine per pair, accumulate per batch
        let mut all_batches: Vec<&[usize]> = batches.iter().map(Vec::as_slice).collect();
        if let Some(tail) = tail_batch {
            all_batches.push(tail);
        }
        let mut batch_accum = Vec::with_capacity(all_batches.len());

        for batch_indices in all_batches {
            if batch_indices.len() < 2 {
                let (fw, fh) = (flow_w, flow_h);
                let tex = self.make_tex(
                    fw, fh,
                    TextureFormat::Rg32Float,
                    TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
                    "zero_accum",
                );
                batch_accum.push((tex, fw, fh));
                continue;
            }

            // Determine pair indices for this batch
            let last_pair = batch_indices.len().saturating_sub(1);
            let pair_indices: Vec<usize> = if options.analyze_skipped_frames {
                (0..last_pair).collect()
            } else {
                vec![0, last_pair - 1]
            };

            // Init per-batch accumulation texture to zeros
            let init_accum = self.make_tex(
                flow_w, flow_h,
                TextureFormat::Rg32Float,
                TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
                "batch_accum",
            );
            let zeros = vec![0u8; (flow_w * flow_h * 8) as usize];
            self.queue.write_texture(
                TexelCopyTextureInfo { texture: &init_accum, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                &zeros,
                TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(flow_w * 8), rows_per_image: Some(flow_h) },
                Extent3d { width: flow_w, height: flow_h, depth_or_array_layers: 1 },
            );

            let mut accum = init_accum;
            let mut actual_w = flow_w;
            let mut actual_h = flow_h;

            for &pair_idx in &pair_indices {
                let i0 = batch_indices[pair_idx];
                let i1 = batch_indices[pair_idx + 1];
                let g0 = gray_of(i0);
                let g1 = gray_of(i1);

                let (fwd_tex, fw, fh) = self.compute_pair_flow(
                    &mut encoder, g0, g1,
                    options.farneback.poly_n, options.farneback.poly_sigma,
                    options.farneback.winsize, options.farneback.use_gaussian,
                    options.farneback.iterations,
                );
                let (bwd_tex, _, _) = self.compute_pair_flow(
                    &mut encoder, g1, g0,
                    options.farneback.poly_n, options.farneback.poly_sigma,
                    options.farneback.winsize, options.farneback.use_gaussian,
                    options.farneback.iterations,
                );

                let combined = self.make_tex(
                    fw, fh,
                    TextureFormat::Rg32Float,
                    TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                    "combined",
                );
                self.combine_flows(&mut encoder, &fwd_tex, &bwd_tex, &combined, fw, fh);

                let next_accum = self.make_tex(
                    fw, fh,
                    TextureFormat::Rg32Float,
                    TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
                    "batch_accum_n",
                );
                self.accumulate_flow(&mut encoder, &accum, &combined, &next_accum, fw, fh);
                accum = next_accum;
                actual_w = fw;
                actual_h = fh;
            }

            batch_accum.push((accum, actual_w, actual_h));
        }

        // --- Record copy-to-staging for all batch accum textures (in the SAME encoder) ---
        #[allow(clippy::items_after_statements)]
        struct StagingSlot {
            buf: Buffer,
            bpr: u32,
            w: u32,
            h: u32,
        }
        let mut staging_slots: Vec<StagingSlot> = Vec::with_capacity(batch_accum.len());
        for &(ref tex, w, h) in &batch_accum {
            let bpr = ((w * 8) + 255) & !255;
            let staging = self.device.create_buffer(&BufferDescriptor {
                label: Some("batch_readback"),
                size: (bpr * h) as u64,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
                },
                Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            staging_slots.push(StagingSlot { buf: staging, bpr, w, h });
        }

        // Single submit for ALL work
        self.queue.submit(Some(encoder.finish()));

        // Poll once, then read all staging buffers
        let (tx, rx) = std::sync::mpsc::channel();
        for slot in &staging_slots {
            let tx = tx.clone();
            slot.buf.slice(..).map_async(MapMode::Read, move |r| {
                let _ = tx.send(r.is_ok());
            });
        }
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .ok();
        // Drain all map completions
        for _ in &staging_slots {
            let _ = rx.recv();
        }

        // Read and normalize each flow
        let mut flows: Vec<Flow> = Vec::with_capacity(staging_slots.len());
        for slot in &staging_slots {
            let data = slot.buf.slice(..).get_mapped_range();
            let mut flow = Flow::zeros(slot.w, slot.h);
            for y in 0..slot.h as usize {
                for x in 0..slot.w as usize {
                    let off = y * slot.bpr as usize + x * 8;
                    let fx = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                    let fy = f32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap());
                    flow.data[y * slot.w as usize + x] = [fx, fy];
                }
            }
            drop(data);

            // Normalize
            let nw = slot.w as f32;
            let nh = slot.h as f32;
            for px in &mut flow.data {
                px[0] /= nw;
                px[1] /= nh;
            }

            // Resize to expected output if needed
            let result = if slot.w != flow_w || slot.h != flow_h {
                resize_flow_bilinear(&flow, flow_w, flow_h)
            } else {
                flow
            };
            flows.push(result);

            slot.buf.unmap();
        }

        flows
    }

    /// Run GPU flow computation on all provided frames and return the accumulated flow.
    ///
    /// All frame pairs are processed in a single GPU encoder submission:
    /// - Forward + backward Farneback per pair, combined bidirectionally on GPU
    /// - Per-pair flow accumulated on GPU (ping-pong accumulation textures)
    /// - Single readback at the end
    ///
    /// The accumulated flow is normalized (dx/=width, dy/=height) before returning.
    // allow(too_many_lines, cast_lossless): pipeline orchestration is sequential,
    //   f64 casts from u32 are safe for image dimensions
    #[allow(clippy::too_many_lines, clippy::cast_lossless)]
    pub fn compute_flow(
        &self,
        frames: &[ImageRgba8],
        options: &GenerateOptions,
    ) -> Result<Flow, String> {
        if frames.len() < 2 {
            return Err("Need at least 2 frames".into());
        }

        let (atlas_cols, atlas_rows) = options.atlas_dims;
        let src_aspect = frames[0].width as f64 / frames[0].height as f64;
        let (tile_w, tile_h) = crate::pipeline::atlas_layout::compute_tile_dims(
            options.atlas_resolution,
            atlas_cols,
            atlas_rows,
            src_aspect,
        );

        let flow_w = tile_w;
        let flow_h = tile_h;

        // Upload original frames as GPU textures, then resize on GPU
        let orig_texs: Vec<Texture> = frames.iter().map(|f| self.upload_frame(f)).collect();

        // Respect analyze_skipped_frames: when false, only process first→last pair
        let last_pair = orig_texs.len().saturating_sub(1);
        let pair_indices: Vec<usize> = if options.analyze_skipped_frames {
            (0..last_pair).collect()
        } else {
            vec![0, last_pair - 1]
        };

        // Single encoder for ALL the work
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        // GPU resize: dispatch resize shader for each frame
        let interp = match options.resize_algorithm {
            crate::pipeline::Interpolation::Nearest => 0u32,
            crate::pipeline::Interpolation::Linear => 1,
            _ => 2,
        };
        let frame_texs: Vec<Texture> = orig_texs
            .iter()
            .map(|tex| self.resize_tex(&mut encoder, tex, tile_w.max(1), tile_h.max(1), interp))
            .collect();

        // Pre-compute grayscale for every frame (shared across forward/backward passes)
        let gray_texs: Vec<Texture> = frame_texs
            .iter()
            .map(|tex| self.grayscale(&mut encoder, tex))
            .collect();

        // Initialize accumulation texture as zeros via write_texture
        // (ordered before encoder submission by the queue)
        let init_accum = self.make_tex(
            flow_w,
            flow_h,
            TextureFormat::Rg32Float,
            TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            "accum_0",
        );
        let accum_bytes = vec![0u8; (flow_w * flow_h * 8) as usize];
        self.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &init_accum,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &accum_bytes,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(flow_w * 8),
                rows_per_image: Some(flow_h),
            },
            Extent3d {
                width: flow_w,
                height: flow_h,
                depth_or_array_layers: 1,
            },
        );

        let mut accum = init_accum;
        let mut flow_w_actual = flow_w;
        let mut flow_h_actual = flow_h;

        for &pair_idx in &pair_indices {
            let g0 = &gray_texs[pair_idx];
            let g1 = &gray_texs[pair_idx + 1];

            // === FORWARD PASS ===
            let (fwd_tex, fw, fh) = self.compute_pair_flow(
                &mut encoder,
                g0,
                g1,
                options.farneback.poly_n,
                options.farneback.poly_sigma,
                options.farneback.winsize,
                options.farneback.use_gaussian,
                options.farneback.iterations,
            );

            // === BACKWARD PASS (reversed frame order) ===
            let (bwd_tex, _, _) = self.compute_pair_flow(
                &mut encoder,
                g1,
                g0,
                options.farneback.poly_n,
                options.farneback.poly_sigma,
                options.farneback.winsize,
                options.farneback.use_gaussian,
                options.farneback.iterations,
            );

            // === BIDIRECTIONAL COMBINE on GPU ===
            let combined = self.make_tex(
                fw,
                fh,
                TextureFormat::Rg32Float,
                TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                "combined",
            );
            self.combine_flows(&mut encoder, &fwd_tex, &bwd_tex, &combined, fw, fh);

            // === ACCUMULATE on GPU (ping-pong) ===
            let next_accum = self.make_tex(
                fw,
                fh,
                TextureFormat::Rg32Float,
                TextureUsages::STORAGE_BINDING
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC,
                "accum_n",
            );
            self.accumulate_flow(&mut encoder, &accum, &combined, &next_accum, fw, fh);
            accum = next_accum;
            flow_w_actual = fw;
            flow_h_actual = fh;
        }

        // Single submit for ALL pairs
        self.queue.submit(Some(encoder.finish()));

        // Single readback at the end, using actual flow dimensions
        let final_flow = self.readback_flow(&accum, flow_w_actual, flow_h_actual);

        // Normalize (dx /= width, dy /= height)
        let norm_w = flow_w_actual as f32;
        let norm_h = flow_h_actual as f32;
        let mut accum_flow = final_flow;
        for pixel in &mut accum_flow.data {
            pixel[0] /= norm_w;
            pixel[1] /= norm_h;
        }

        // Resize to the expected output resolution if needed
        if flow_w_actual != flow_w || flow_h_actual != flow_h {
            accum_flow = resize_flow_bilinear(&accum_flow, flow_w, flow_h);
        }

        Ok(accum_flow)
    }

    /// Run the full GPU-accelerated flow pipeline on a set of frames.
    ///
    /// Returns (`color_atlas`, `motion_atlas`, `max_strength`).
    // allow(cast_lossless): f64 casts from u32/f32 are safe for image dims
    #[allow(clippy::cast_lossless)]
    pub fn compute(
        &self,
        frames: &[ImageRgba8],
        options: &GenerateOptions,
    ) -> Result<(ImageRgba8, ImageRgba8, f64), String> {
        let (atlas_cols, atlas_rows) = options.atlas_dims;
        let src_aspect = frames[0].width as f64 / frames[0].height as f64;
        let (tile_w, tile_h) = crate::pipeline::atlas_layout::compute_tile_dims(
            options.atlas_resolution,
            atlas_cols,
            atlas_rows,
            src_aspect,
        );
        let frame_skip = options.frame_skip.max(1) as usize;
        let output_frames = options.output_frames.max(1) as usize;

        let selected: Vec<&ImageRgba8> = frames
            .iter()
            .step_by(frame_skip)
            .take(output_frames)
            .collect();
        let selected_owned: Vec<ImageRgba8> = selected.iter().map(|f| (*f).clone()).collect();
        let accum_flow = self.compute_flow(&selected_owned, options)?;

        let interp = options.resize_algorithm;
        let tile_frames: Vec<ImageRgba8> = selected
            .par_iter()
            .map(|f| crate::pipeline::atlas::resize_nyquist(f, tile_w.max(1), interp))
            .collect();

        let flow_w = tile_w;

        let max_strength = accum_flow
            .data
            .iter()
            .map(|[dx, dy]| dx.hypot(*dy))
            .fold(f32::NEG_INFINITY, f32::max)
            .max(1e-8);

        let atlas_w = tile_w * atlas_cols;
        let atlas_h = tile_h * atlas_rows;
        let mut motion_atlas = ImageRgba8::zeros(atlas_w, atlas_h);

        let scale = 0.5;
        for ty in 0..atlas_rows {
            for tx in 0..atlas_cols {
                let tile_idx = (ty * atlas_cols + tx) as usize;
                if tile_idx >= selected.len() {
                    break;
                }
                for row in 0..tile_h {
                    for col in 0..tile_w {
                        let si = (row as usize) * (flow_w as usize) + (col as usize);
                        let dy = ty * tile_h + row;
                        let dx = tx * tile_w + col;
                        let di = (dy as usize * atlas_w as usize + dx as usize) * 4;
                        let [fx, fy] = accum_flow.data[si];
                        motion_atlas.data[di] =
                            (fx / max_strength * scale + scale).clamp(0.0, 1.0) as u8;
                        motion_atlas.data[di + 1] =
                            (fy / max_strength * scale + scale).clamp(0.0, 1.0) as u8;
                        motion_atlas.data[di + 2] = 0;
                        motion_atlas.data[di + 3] = 255;
                    }
                }
            }
        }

        // Build color atlas from selected frames
        let color_atlas =
            build_color_atlas_simple(&tile_frames, atlas_cols, atlas_rows, tile_w, tile_h);

        Ok((color_atlas, motion_atlas, f64::from(max_strength)))
    }

    // ---- Per-pass helpers ----

    fn upload_frame(&self, frame: &ImageRgba8) -> Texture {
        let tex = self.device.create_texture(&TextureDescriptor {
            label: Some("frame"),
            size: Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &frame.data,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(frame.width * 4),
                rows_per_image: Some(frame.height),
            },
            Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            },
        );
        tex
    }

    fn make_tex(
        &self,
        w: u32,
        h: u32,
        fmt: TextureFormat,
        usage: TextureUsages,
        label: &str,
    ) -> Texture {
        self.device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: fmt,
            usage,
            view_formats: &[],
        })
    }

    fn resize_tex(
        &self,
        encoder: &mut CommandEncoder,
        src: &Texture,
        dst_w: u32,
        dst_h: u32,
        interp: u32,
    ) -> Texture {
        let dst = self.make_tex(
            dst_w,
            dst_h,
            TextureFormat::Rgba8Unorm,
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            "resized",
        );
        let ubuf = self.device.create_buffer(&BufferDescriptor {
            label: Some("resize_ubuf"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(
            &ubuf,
            0,
            bytemuck::bytes_of(&[interp as f32, 0.0, 0.0, 0.0]),
        );
        let bg = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bg_resize"),
            layout: &self.bgl_resize,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&Self::tex_view(src)),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&Self::tex_view(&dst)),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: ubuf.as_entire_binding(),
                },
            ],
        });
        self.dispatch_16(encoder, &self.resize, &bg, dst_w, dst_h);
        dst
    }

    fn grayscale(&self, encoder: &mut CommandEncoder, frame: &Texture) -> Texture {
        let w = frame.width();
        let h = frame.height();
        let out = self.make_tex(
            w,
            h,
            TextureFormat::R32Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            "gray",
        );
        let bg = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bg_gray"),
            layout: &self.bgl_tex_in_out,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&Self::tex_view(frame)),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&Self::tex_view(&out)),
                },
            ],
        });
        self.dispatch_16(encoder, &self.grayscale, &bg, w, h);
        out
    }

    // allow(cast_lossless, cast_possible_wrap): level counts and sigma math fit f32
    #[allow(clippy::cast_lossless, clippy::cast_possible_wrap)]
    fn pyramid_all(
        &self,
        encoder: &mut CommandEncoder,
        base: &Texture,
    ) -> Vec<(Texture, u32, u32)> {
        let orig_w = base.width();
        let orig_h = base.height();
        let pyr_scale = 0.5f64;

        let mut levels: Vec<(Texture, u32, u32)> = Vec::new();

        // Build each level independently from the ORIGINAL base texture
        // with the correct Gaussian anti-aliasing (matches CPU build_level_image).
        for level_k in 1u32.. {
            let scale = pyr_scale.powi(level_k as i32);
            let next_w = (f64::from(orig_w) * scale).round() as u32;
            let next_h = (f64::from(orig_h) * scale).round() as u32;

            if next_w < 32 || next_h < 32 {
                break;
            }

            // Compute Gaussian sigma matching CPU: sigma = (1/scale - 1) * 0.5
            let sigma = ((1.0 / scale) - 1.0) * 0.5;
            let ksize = ((sigma * 5.0).round() as u32).max(3) | 1;
            let half = (ksize / 2) as i32;

            let scale_x = f64::from(orig_w) / f64::from(next_w);
            let scale_y = f64::from(orig_h) / f64::from(next_h);

            let out = self.make_tex(
                next_w, next_h,
                TextureFormat::R32Float,
                TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                "pyr",
            );

            let ubuf = self.device.create_buffer(&BufferDescriptor {
                label: Some("pyr_ubuf"),
                size: 16,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(
                &ubuf, 0,
                bytemuck::bytes_of(&[scale_x as f32, scale_y as f32, sigma as f32, half as f32]),
            );

            let bg = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("bg_pyr"),
                layout: &self.bgl_pyramid,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&Self::tex_view(base)),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: ubuf.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(&Self::tex_view(&out)),
                    },
                ],
            });
            self.dispatch_16(encoder, &self.pyramid_blur, &bg, next_w, next_h);

            levels.push((out, next_w, next_h));
        }

        levels.reverse();
        levels
    }

    // allow(many_single_char_names): math notation (n, s, g, x, w, h) matches poly.rs
    // allow(cast_possible_wrap, cast_lossless): poly_n/u32→i32 and poly_sigma/f32→f64 controlled inputs
    #[allow(
        clippy::many_single_char_names,
        clippy::cast_possible_wrap,
        clippy::cast_lossless
    )]
    fn poly_expand(
        &self,
        encoder: &mut CommandEncoder,
        level: &(Texture, u32, u32),
        poly_n: u32,
        poly_sigma: f32,
    ) -> Texture {
        let (ref tex, w, h) = *level;

        // --- Compute Gaussian-based poly expansion kernels (matches poly.rs) ---
        let n = poly_n as i32;
        let sigma = poly_sigma as f64;
        let ksize = (2 * n + 1) as u32;

        let g_raw: Vec<f64> = (-n..=n)
            .map(|i| (-f64::from(i).powi(2) / (2.0 * sigma * sigma)).exp())
            .collect();
        let sum_g: f64 = g_raw.iter().sum();
        let mut g = vec![0.0f32; ksize as usize];
        let mut xg = vec![0.0f32; ksize as usize];
        let mut xxg = vec![0.0f32; ksize as usize];
        let mut m2 = 0.0f64;
        let mut m4 = 0.0f64;
        for i in -n..=n {
            let idx = (i + n) as usize;
            let x = f64::from(i);
            let gv = g_raw[idx] / sum_g;
            g[idx] = gv as f32;
            xg[idx] = (-x * gv) as f32;
            xxg[idx] = (x * x * gv) as f32;
            m2 += x * x * gv;
            m4 += x * x * x * x * gv;
        }
        let ig11 = 1.0 / m2;
        let ig03 = -m2 / (m4 - m2 * m2);
        let ig33 = 1.0 / (m4 - m2 * m2);
        let ig55 = 1.0 / (m2 * m2);

        // Create per-call buffers (thread-safe: each Rayon task gets its own)
        let params_buf = self.device.create_buffer(&BufferDescriptor {
            label: Some("poly_params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(
            &params_buf,
            0,
            bytemuck::bytes_of(&[ksize as f32, ig11 as f32, ig03 as f32, ig33 as f32]),
        );
        let params2_buf = self.device.create_buffer(&BufferDescriptor {
            label: Some("poly_params2"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(
            &params2_buf,
            0,
            bytemuck::bytes_of(&[ig55 as f32, 0.0, 0.0, 0.0]),
        );

        // Pack kernel data
        let mut kernel_bytes = Vec::with_capacity(3 * ksize as usize * 4);
        kernel_bytes.extend_from_slice(bytemuck::cast_slice(&g));
        kernel_bytes.extend_from_slice(bytemuck::cast_slice(&xg));
        kernel_bytes.extend_from_slice(bytemuck::cast_slice(&xxg));
        let kernel_buf = self.device.create_buffer(&BufferDescriptor {
            label: Some("poly_kernel"),
            size: (kernel_bytes.len()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&kernel_buf, 0, &kernel_bytes);

        let out = self.make_tex(
            w * 2,
            h,
            TextureFormat::Rgba32Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            "poly",
        );

        let bg = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bg_poly"),
            layout: &self.bgl_poly,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&Self::tex_view(tex)),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&Self::tex_view(&out)),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: params2_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: kernel_buf.as_entire_binding(),
                },
            ],
        });
        self.dispatch_16(encoder, &self.poly_expansion, &bg, w, h);
        out
    }

    // allow(many_single_char_names): math notation (n, k, i) matches update.rs
    // allow(needless_range_loop): kernel construction matches the CPU reference
    // allow(too_many_arguments): matches CPU update_flow_with_workspace signature
    #[allow(
        clippy::many_single_char_names,
        clippy::needless_range_loop,
        clippy::too_many_arguments
    )]
    fn flow_update(
        &self,
        encoder: &mut CommandEncoder,
        poly_a: &Texture,
        poly_b: &Texture,
        prior_flow: Option<&Texture>,
        w: u32,
        h: u32,
        winsize: u32,
        use_gaussian: bool,
    ) -> Texture {
        // Build smoothing kernel (matches update.rs build_gaussian_1d / box)
        let kernel: Vec<f32> = if use_gaussian {
            let n = winsize as usize;
            let sigma = (f64::from(winsize / 2) * 0.3).max(0.3);
            let half = (n as f64 - 1.0) / 2.0;
            let mut k = vec![0.0f64; n];
            let mut sum = 0.0f64;
            for i in 0..n {
                let x = i as f64 - half;
                let v = (-x * x / (2.0 * sigma * sigma)).exp();
                k[i] = v;
                sum += v;
            }
            k.iter().map(|&v| (v / sum) as f32).collect()
        } else {
            vec![1.0f32; winsize as usize]
        };

        // Create per-call kernel buffer (thread-safe: each Rayon task gets its own)
        let kernel_bytes: Vec<u8> = bytemuck::cast_slice(&kernel).to_vec();
        let kernel_buf = self.device.create_buffer(&BufferDescriptor {
            label: Some("flow_kernel"),
            size: (kernel_bytes.len()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&kernel_buf, 0, &kernel_bytes);

        let out = self.make_tex(
            w,
            h,
            TextureFormat::Rg32Float,
            TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            "flow",
        );

        let prior = prior_flow.unwrap_or(&self.zero_prior);

        // Create per-call params buffer (thread-safe)
        let params = self.device.create_buffer(&BufferDescriptor {
            label: Some("flow_params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(
            &params,
            0,
            bytemuck::bytes_of(&[
                winsize as f32,
                0.0,
                if use_gaussian { 1.0 } else { 0.0 },
                0.0,
            ]),
        );

        let bg = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bg_flow"),
            layout: &self.bgl_flow_update,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&Self::tex_view(poly_a)),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&Self::tex_view(poly_b)),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&Self::tex_view(prior)),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&Self::tex_view(&out)),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: kernel_buf.as_entire_binding(),
                },
            ],
        });
        self.dispatch_16(encoder, &self.flow_update, &bg, w, h);
        out
    }

    /// Accumulate `pair_flow` into `accum_in`, writing result to `accum_out`.
    ///
    /// All textures are `Rg32Float` with the given dimensions.
    fn accumulate_flow(
        &self,
        encoder: &mut CommandEncoder,
        accum_in: &Texture,
        pair_flow: &Texture,
        accum_out: &Texture,
        w: u32,
        h: u32,
    ) {
        let bg = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bg_accum"),
            layout: &self.bgl_accum_combine,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&Self::tex_view(accum_in)),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&Self::tex_view(pair_flow)),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&Self::tex_view(accum_out)),
                },
            ],
        });
        self.dispatch_16(encoder, &self.accumulate, &bg, w, h);
    }

    /// Combine forward and backward flows bidirectionally: `0.5 * (fwd - bwd)`.
    ///
    /// All textures are `Rg32Float` with the given dimensions.
    fn combine_flows(
        &self,
        encoder: &mut CommandEncoder,
        fwd: &Texture,
        bwd: &Texture,
        out: &Texture,
        w: u32,
        h: u32,
    ) {
        let bg = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bg_combine"),
            layout: &self.bgl_accum_combine,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&Self::tex_view(fwd)),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&Self::tex_view(bwd)),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&Self::tex_view(out)),
                },
            ],
        });
        self.dispatch_16(encoder, &self.combine_bidir, &bg, w, h);
    }

    // allow(dead_code): kept for API completeness
    #[allow(dead_code)]
    fn encode_flow(
        &self,
        encoder: &mut CommandEncoder,
        accum: &Flow,
        max_strength: f32,
    ) -> Texture {
        let w = accum.width;
        let h = accum.height;

        let out = self.make_tex(
            w,
            h,
            TextureFormat::Rg32Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
            "encoded",
        );

        // Upload accumulator to GPU
        let accum_tex = self.make_tex(
            w,
            h,
            TextureFormat::Rg32Float,
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            "accum_upload",
        );
        let accum_bytes: Vec<u8> = accum
            .data
            .iter()
            .flat_map(|&v| bytemuck::bytes_of(&v).to_vec())
            .collect();
        self.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &accum_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &accum_bytes,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 8),
                rows_per_image: Some(h),
            },
            Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        let ubuf = self.device.create_buffer(&BufferDescriptor {
            label: Some("encode_ubuf"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&ubuf, 0, bytemuck::bytes_of(&[max_strength, 0.0, 0.0, 0.0]));

        let bg = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bg_encode"),
            layout: &self.bgl_encode,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&Self::tex_view(&accum_tex)),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: ubuf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&Self::tex_view(&out)),
                },
            ],
        });
        self.dispatch_16(encoder, &self.encode, &bg, w, h);
        out
    }

    // allow(cast_lossless): bpr*h fits u64 for any sensible image size
    #[allow(clippy::cast_lossless)]
    fn readback_flow(&self, tex: &Texture, w: u32, h: u32) -> Flow {
        let bpr = ((w * 8) + 255) & !255;
        let staging = self.device.create_buffer(&BufferDescriptor {
            label: Some("readback"),
            size: u64::from(bpr * h),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &staging,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(h),
                },
            },
            Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
        rx.recv().unwrap().expect("map failed");

        let data = slice.get_mapped_range();
        let mut flow = Flow::zeros(w, h);
        for y in 0..h as usize {
            for x in 0..w as usize {
                let off = y * bpr as usize + x * 8;
                let fx = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                let fy = f32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap());
                flow.data[y * w as usize + x] = [fx, fy];
            }
        }
        drop(data);
        staging.unmap();
        flow
    }
}

fn resize_flow_bilinear(flow: &Flow, new_w: u32, new_h: u32) -> Flow {
    if flow.width == new_w && flow.height == new_h {
        return flow.clone();
    }
    let mut out = Flow::zeros(new_w, new_h);
    let sw = flow.width as f32;
    let sh = flow.height as f32;
    let dw = new_w as f32;
    let dh = new_h as f32;
    for dy in 0..new_h {
        for dx in 0..new_w {
            let sx = (dx as f32 + 0.5) * sw / dw - 0.5;
            let sy = (dy as f32 + 0.5) * sh / dh - 0.5;
            let x0 = (sx.floor().max(0.0).min((flow.width - 1) as f32)) as usize;
            let y0 = (sy.floor().max(0.0).min((flow.height - 1) as f32)) as usize;
            let x1 = (x0 + 1).min(flow.width as usize - 1);
            let y1 = (y0 + 1).min(flow.height as usize - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            let f00 = flow.data[y0 * flow.width as usize + x0];
            let f10 = flow.data[y0 * flow.width as usize + x1];
            let f01 = flow.data[y1 * flow.width as usize + x0];
            let f11 = flow.data[y1 * flow.width as usize + x1];
            let w00 = (1.0 - fx) * (1.0 - fy);
            let w10 = fx * (1.0 - fy);
            let w01 = (1.0 - fx) * fy;
            let w11 = fx * fy;
            let idx = dy as usize * new_w as usize + dx as usize;
            out.data[idx][0] =
                f00[0].mul_add(w00, f10[0].mul_add(w10, f01[0].mul_add(w01, f11[0] * w11)));
            out.data[idx][1] =
                f00[1].mul_add(w00, f10[1].mul_add(w10, f01[1].mul_add(w01, f11[1] * w11)));
        }
    }
    out
}

fn build_color_atlas_simple(
    frames: &[ImageRgba8],
    cols: u32,
    rows: u32,
    tw: u32,
    th: u32,
) -> ImageRgba8 {
    let aw = cols * tw;
    let ah = rows * th;
    let mut atlas = ImageRgba8::zeros(aw, ah);
    for idx in 0..(cols * rows).min(frames.len() as u32) {
        let tx = (idx % cols) * tw;
        let ty = (idx / cols) * th;
        let f = &frames[idx as usize];
        for row in 0..th.min(f.height) {
            for col in 0..tw.min(f.width) {
                let sy = (row * f.height / th) as usize;
                let sx = (col * f.width / tw) as usize;
                let si = (sy * f.width as usize + sx) * 4;
                let di = ((ty + row) as usize * aw as usize + (tx + col) as usize) * 4;
                atlas.data[di..di + 4].copy_from_slice(&f.data[si..si + 4]);
            }
        }
    }
    atlas
}
