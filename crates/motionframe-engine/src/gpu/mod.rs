use std::sync::Arc;

use rayon::prelude::*;

use crate::pipeline::{Flow, GenerateOptions, ImageRgba8};
use wgpu::*;

/// GPU-accelerated Farneback optical flow pipeline.
///
/// Owns a wgpu device, queue, and all compute pipeline state. Call
/// `compute()` to run the full coarse-to-fine flow estimation on a
/// slice of RGBA8 frames and produce encoded color + motion atlases.
pub struct GpuPipeline {
    device: Arc<Device>,
    queue: Queue,
    // Compute pipelines
    grayscale: ComputePipeline,
    pyramid: ComputePipeline,
    poly_expansion: ComputePipeline,
    flow_update: ComputePipeline,
    upsample: ComputePipeline,
    encode: ComputePipeline,
    // Bind group layouts
    bgl_tex_in_out: BindGroupLayout,
    bgl_tex_in_out_rgba: BindGroupLayout,
    bgl_tex_in_out_rg: BindGroupLayout,
    bgl_flow_update: BindGroupLayout,
    bgl_encode: BindGroupLayout,
}

impl GpuPipeline {
    /// Attempt to initialize a GPU pipeline with a new wgpu device.
    /// Returns `None` if no suitable GPU adapter is available.
    pub fn try_init() -> Option<Self> {
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(
            &RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                ..RequestAdapterOptions::default()
            },
        ))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor::default(),
        ))
        .ok()?;
        Some(Self::new(Arc::new(device), queue))
    }

    /// Create a new GPU pipeline from an existing wgpu device + queue.
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

        // 1 sampled texture + 1 RGBA32Float storage texture (poly_expansion)
        let bgl_tex_in_out_rgba = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bgl_tex_in_out_rgba"),
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

        // --- Pipeline layouts ---
        let pl_1ch = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pl_1ch"),
            bind_group_layouts: &[Some(&bgl_tex_in_out)],
            immediate_size: 0,
        });
        let pl_rgba = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pl_rgba"),
            bind_group_layouts: &[Some(&bgl_tex_in_out_rgba)],
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

        // --- Shader modules ---
        let mod_grayscale =
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("grayscale"),
                source: ShaderSource::Wgsl(include_str!("shaders/grayscale.wgsl").into()),
            });
        let mod_pyramid =
            device.create_shader_module(ShaderModuleDescriptor {
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
        let mod_upsample =
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("upsample"),
                source: ShaderSource::Wgsl(include_str!("shaders/upsample.wgsl").into()),
            });
        let mod_encode =
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("encode"),
                source: ShaderSource::Wgsl(include_str!("shaders/encode.wgsl").into()),
            });

        // --- Compute pipelines ---
        let grayscale = Self::make_pipeline(&device, &pl_1ch, &mod_grayscale, "grayscale");
        let pyramid = Self::make_pipeline(&device, &pl_1ch, &mod_pyramid, "pyramid");
        let poly_expansion = Self::make_pipeline(&device, &pl_rgba, &mod_poly, "poly_expansion");
        let flow_update = Self::make_pipeline(&device, &pl_flow, &mod_flow, "flow_update");
        let upsample = Self::make_pipeline(&device, &pl_rg, &mod_upsample, "upsample");
        let encode = Self::make_pipeline(&device, &pl_encode, &mod_encode, "encode");

        Self {
            device,
            queue,
            grayscale,
            pyramid,
            poly_expansion,
            flow_update,
            upsample,
            encode,
            bgl_tex_in_out,
            bgl_tex_in_out_rgba,
            bgl_tex_in_out_rg,
            bgl_flow_update,
            bgl_encode,
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

    /// Run the full GPU-accelerated flow pipeline on a set of frames.
    ///
    /// Returns (color_atlas, motion_atlas, max_strength).
    pub fn compute(
        &self,
        frames: &[ImageRgba8],
        options: &GenerateOptions,
    ) -> Result<(ImageRgba8, ImageRgba8, f64), String> {
        if frames.len() < 2 {
            return Err("Need at least 2 frames".into());
        }

        let (atlas_cols, atlas_rows) = options.atlas_dims;
        let src_aspect = frames[0].width as f64 / frames[0].height as f64;
        let (tile_w, tile_h) = crate::pipeline::atlas_layout::compute_tile_dims(
            options.atlas_resolution, atlas_cols, atlas_rows, src_aspect,
        );
        let frame_skip = options.frame_skip.max(1) as usize;
        let output_frames = options.output_frames.max(1) as usize;

        // Select frames respecting frame_skip and output_frames
        let selected: Vec<&ImageRgba8> = frames.iter().step_by(frame_skip).take(output_frames).collect();
        if selected.len() < 2 {
            return Err("Need at least 2 frames after skip/limit".into());
        }

        // Resize all selected frames to tile dimensions (CPU, parallel)
        let interp = options.resize_algorithm;
        let tile_frames: Vec<ImageRgba8> = selected
            .par_iter()
            .map(|f| {
                crate::pipeline::atlas::resize_nyquist(f, tile_w.max(1), interp)
            })
            .collect();

        let flow_w = tile_w;
        let flow_h = tile_h;
        let mut accum_flow = Flow::zeros(flow_w, flow_h);

        // Upload tile frames to GPU
        let frame_texs: Vec<Texture> = tile_frames
            .iter()
            .map(|f| self.upload_frame(f))
            .collect::<Result<Vec<_>, _>>()?;

        // Process each consecutive frame pair
        for pair_idx in 0..frame_texs.len().saturating_sub(1) {
            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());

            let gray0 = self.grayscale(&mut encoder, &frame_texs[pair_idx]);
            let gray1 = self.grayscale(&mut encoder, &frame_texs[pair_idx + 1]);
            let pyr0 = self.pyramid_all(&mut encoder, &gray0);
            let pyr1 = self.pyramid_all(&mut encoder, &gray1);
            let num_levels = pyr0.len();

            // Coarse-to-fine: level 0 = coarsest (smallest), level N-1 = finest (largest).
            let mut prior_flow: Option<(Texture, u32, u32)> = None;

            for level in 0..num_levels {
                let (_, lw, lh) = pyr0[level];

                let pa = self.poly_expand(&mut encoder, &pyr0[level]);
                let pb = self.poly_expand(&mut encoder, &pyr1[level]);

                // Upsample prior flow to current level size (or zero-init for coarsest)
                // Build initial flow for this level (upsampled from coarser level, or zero)
                let init_flow: Option<Texture> = if level == 0 {
                    None
                } else if prior_flow.is_none() {
                    None
                } else {
                    let up = {
                        let (ref prev_tex, ..) = prior_flow.as_ref().unwrap();
                        let up = self.make_tex(lw, lh, TextureFormat::Rg32Float,
                            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
                            "up_flow");
                        let in_view = Self::tex_view(prev_tex);
                        let out_view = Self::tex_view(&up);
                        let up_bg = self.device.create_bind_group(&BindGroupDescriptor {
                            label: Some("bg_upsample"),
                            layout: &self.bgl_tex_in_out_rg,
                            entries: &[
                                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&in_view) },
                                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&out_view) },
                            ],
                        });
                        self.dispatch_16(&mut encoder, &self.upsample, &up_bg, lw, lh);
                        up
                    };
                    Some(up)
                };

                // Run multiple flow update iterations at this level
                let mut cur_flow = init_flow;
                let num_iters = options.farneback.iterations.max(1);
                for _iter in 0..num_iters {
                    cur_flow = Some(self.flow_update(
                        &mut encoder,
                        &pa,
                        &pb,
                        cur_flow.as_ref(),
                        lw,
                        lh,
                        options.farneback.winsize,
                        2.0,
                    ));
                }
                prior_flow = cur_flow.map(|f| (f, lw, lh));
            }

            if let Some((ref flow_tex, fw, fh)) = prior_flow {
                let pair_flow = self.readback_flow(flow_tex, fw, fh);
                let resized = resize_flow_bilinear(&pair_flow, flow_w, flow_h);
                for (i, v) in resized.data.iter().enumerate() {
                    accum_flow.data[i][0] += v[0];
                    accum_flow.data[i][1] += v[1];
                }
            }

            self.queue.submit(Some(encoder.finish()));
        }

        // Encode accumulated flow using the encoding specified in options
        let max_strength = accum_flow.data.iter()
            .map(|[dx, dy]| (dx * dx + dy * dy).sqrt())
            .fold(f32::NEG_INFINITY, f32::max)
            .max(1e-8);

        let atlas_w = tile_w * atlas_cols;
        let atlas_h = tile_h * atlas_rows;
        let mut motion_atlas = ImageRgba8::zeros(atlas_w, atlas_h);

        let scale = 0.5;
        for ty in 0..atlas_rows {
            for tx in 0..atlas_cols {
                let tile_idx = (ty * atlas_cols + tx) as usize;
                if tile_idx >= selected.len() { break; }
                for row in 0..tile_h {
                    for col in 0..tile_w {
                        let si = (row as usize) * (flow_w as usize) + (col as usize);
                        let dy = ty * tile_h + row;
                        let dx = tx * tile_w + col;
                        let di = (dy as usize * atlas_w as usize + dx as usize) * 4;
                        let [fx, fy] = accum_flow.data[si];
                        motion_atlas.data[di] = (fx / max_strength * scale + scale).clamp(0.0, 1.0) as u8;
                        motion_atlas.data[di + 1] = (fy / max_strength * scale + scale).clamp(0.0, 1.0) as u8;
                        motion_atlas.data[di + 2] = 0;
                        motion_atlas.data[di + 3] = 255;
                    }
                }
            }
        }

        // Build color atlas from selected frames
        let color_atlas = build_color_atlas_simple(&tile_frames, atlas_cols, atlas_rows, tile_w, tile_h);

        Ok((color_atlas, motion_atlas, max_strength as f64))
    }

    // ---- Per-pass helpers ----

    fn upload_frame(&self, frame: &ImageRgba8) -> Result<Texture, String> {
        let tex = self.device.create_texture(&TextureDescriptor {
            label: Some("frame"),
            size: Extent3d { width: frame.width, height: frame.height, depth_or_array_layers: 1 },
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
            Extent3d { width: frame.width, height: frame.height, depth_or_array_layers: 1 },
        );
        Ok(tex)
    }

    fn make_tex(&self, w: u32, h: u32, fmt: TextureFormat, usage: TextureUsages, label: &str) -> Texture {
        self.device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: fmt,
            usage,
            view_formats: &[],
        })
    }

    fn grayscale(&self, encoder: &mut CommandEncoder, frame: &Texture) -> Texture {
        let w = frame.width();
        let h = frame.height();
        let out = self.make_tex(
            w, h,
            TextureFormat::R32Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            "gray",
        );
        let bg = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bg_gray"),
            layout: &self.bgl_tex_in_out,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&Self::tex_view(frame)) },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&Self::tex_view(&out)) },
            ],
        });
        self.dispatch_16(encoder, &self.grayscale, &bg, w, h);
        out
    }

    fn pyramid_all(&self, encoder: &mut CommandEncoder, base: &Texture) -> Vec<(Texture, u32, u32)> {
        let mut levels: Vec<(Texture, u32, u32)> = Vec::new();

        let mut cur_w = base.width();
        let mut cur_h = base.height();
        let mut cur_src: Option<Texture> = None;

        loop {
            let next_w = (cur_w / 2).max(1);
            let next_h = (cur_h / 2).max(1);

            let down = self.make_tex(
                next_w, next_h,
                TextureFormat::R32Float,
                TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                "pyr",
            );

            if next_w < 32 || next_h < 32 {
                levels.push((down, cur_w, cur_h));
                break;
            }

            let src = cur_src.as_ref().unwrap_or(base);
            let bg = {
                let down_view = Self::tex_view(&down);
                self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("bg_pyr"),
                    layout: &self.bgl_tex_in_out,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&Self::tex_view(src)) },
                        BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&down_view) },
                    ],
                })
            };
            self.dispatch_16(encoder, &self.pyramid, &bg, next_w, next_h);

            levels.push((down.clone(), next_w, next_h));
            cur_src = Some(down);
            cur_w = next_w;
            cur_h = next_h;
        }

        levels.reverse();
        levels
    }

    fn poly_expand(&self, encoder: &mut CommandEncoder, level: &(Texture, u32, u32)) -> Texture {
        let (ref tex, w, h) = *level;
        let out = self.make_tex(
            w * 2, h,
            TextureFormat::Rgba32Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            "poly",
        );
        let bg = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bg_poly"),
            layout: &self.bgl_tex_in_out_rgba,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&Self::tex_view(tex)) },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&Self::tex_view(&out)) },
            ],
        });
        self.dispatch_16(encoder, &self.poly_expansion, &bg, w, h);
        out
    }

    fn flow_update(
        &self,
        encoder: &mut CommandEncoder,
        poly_a: &Texture,
        poly_b: &Texture,
        prior_flow: Option<&Texture>,
        w: u32,
        h: u32,
        winsize: u32,
        scale_factor: f32,
    ) -> Texture {
        let out = self.make_tex(
            w, h,
            TextureFormat::Rg32Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
            "flow",
        );

        // Zero prior if none
        let zero_prior = self.make_tex(
            1, 1,
            TextureFormat::Rg32Float,
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            "zero_prior",
        );
        self.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &zero_prior, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All,
            },
            &[0u8; 8],
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8), rows_per_image: Some(1) },
            Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let prior = prior_flow.unwrap_or(&zero_prior);

        let params = self.device.create_buffer(&BufferDescriptor {
            label: Some("flow_params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(
            &params, 0,
            bytemuck::bytes_of(&[winsize as f32, scale_factor, 0.0, 0.0]),
        );

        let bg = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bg_flow"),
            layout: &self.bgl_flow_update,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&Self::tex_view(poly_a)) },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&Self::tex_view(poly_b)) },
                BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&Self::tex_view(prior)) },
                BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: BindingResource::TextureView(&Self::tex_view(&out)) },
            ],
        });
        self.dispatch_16(encoder, &self.flow_update, &bg, w, h);
        out
    }

    fn encode_flow(&self, encoder: &mut CommandEncoder, accum: &Flow, max_strength: f32) -> Texture {
        let w = accum.width;
        let h = accum.height;

        let out = self.make_tex(
            w, h,
            TextureFormat::Rg32Float,
            TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
            "encoded",
        );

        // Upload accumulator to GPU
        let accum_tex = self.make_tex(
            w, h,
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
            TexelCopyTextureInfo { texture: &accum_tex, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
            &accum_bytes,
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
            Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );

        let ubuf = self.device.create_buffer(&BufferDescriptor {
            label: Some("encode_ubuf"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&[max_strength, 0.0, 0.0, 0.0]));

        let bg = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bg_encode"),
            layout: &self.bgl_encode,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&Self::tex_view(&accum_tex)) },
                BindGroupEntry { binding: 1, resource: ubuf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&Self::tex_view(&out)) },
            ],
        });
        self.dispatch_16(encoder, &self.encode, &bg, w, h);
        out
    }

    fn readback_flow(&self, tex: &Texture, w: u32, h: u32) -> Flow {
        let bpr = ((w * 8) + 255) & !255;
        let staging = self.device.create_buffer(&BufferDescriptor {
            label: Some("readback"),
            size: (bpr * h) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
            TexelCopyBufferInfo {
                buffer: &staging,
                layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
            },
            Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
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
            out.data[idx][0] = f00[0].mul_add(w00, f10[0].mul_add(w10, f01[0].mul_add(w01, f11[0] * w11)));
            out.data[idx][1] = f00[1].mul_add(w00, f10[1].mul_add(w10, f01[1].mul_add(w01, f11[1] * w11)));
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
