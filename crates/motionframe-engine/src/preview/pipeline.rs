//! wgpu render pipeline for the preview tab.
//!
//! Drops a wgpu render pass inside the egui frame via `egui_wgpu::Callback`.
//! Color and motion atlases are uploaded once per generation; uniforms carry
//! atlas grid size, frame count, motion strength, and fractional time.

use crate::pipeline::ImageRgba8;

/// Uniform buffer data — must match the WGSL struct layout exactly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    /// Atlas grid dimensions (cols, rows).
    pub atlas_grid: [u32; 2],
    /// Number of frames in the atlas.
    pub frame_count: u32,
    /// Motion vector decode strength.
    pub motion_strength: f32,
    /// Fractional frame index (time).
    pub time: f32,
    /// 0 = normal pack, 1 = stagger-packed.
    pub stagger_pack: u32,
    /// 0 = R8G8 remap [0,1], 1 = `SideFX` Labs polar.
    pub mv_encoding: u32,
    /// Background fill: 0 = black, 1 = gray, 2 = white, 3 = checkerboard.
    /// Compositied beneath the preview wherever the color atlas alpha < 1.
    pub bg_mode: u32,
    /// 0 = atlas stores straight (non-premultiplied) alpha,
    /// 1 = atlas stores premultiplied alpha. Selects the composite formula
    /// in the preview shader.
    pub premultiplied_alpha: u32,
    /// 0 = motion-vector warped blend (default), 1 = traditional cross-fade.
    /// Cross-fade skips the per-pixel UV warp and just `mix(c0, c1, t_frac)`,
    /// so the artist can A/B the value the motion vectors are adding.
    pub blend_mode: u32,
    #[doc(hidden)]
    pub _pad2: u32,
    #[doc(hidden)]
    pub _pad3: u32,
}

/// GPU resources for the preview render pipeline.
pub struct PreviewResources {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// Nearest-filter sampler used for SideFX-encoded motion. Bilinear lerps
    /// across the polar 0/2π wrap and across the polar-flip bit, producing
    /// harsh seams; nearest avoids both.
    nearest_sampler: wgpu::Sampler,
    color_texture: Option<wgpu::Texture>,
    motion_texture: Option<wgpu::Texture>,
    bind_group: Option<wgpu::BindGroup>,
}

impl PreviewResources {
    /// Create pipeline resources for the given device and surface format.
    // allow(too_many_lines): wgpu setup is verbose but sequential;
    //   splitting would scatter related bind group layout definitions
    #[allow(clippy::too_many_lines)]
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader_src = include_str!("shader.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preview_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preview_bind_group_layout"),
            entries: &[
                // binding 0: uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: color texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 2: motion vector texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 3: bilinear sampler (color + R8G8 remap motion)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 4: nearest sampler (SideFX-encoded motion)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preview_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("preview_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preview_uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("preview_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("preview_nearest_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            sampler,
            nearest_sampler,
            color_texture: None,
            motion_texture: None,
            bind_group: None,
        }
    }

    /// Upload color and motion atlas textures and rebuild the bind group.
    fn upload_textures(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_atlas: &ImageRgba8,
        motion_atlas: &ImageRgba8,
    ) {
        // Color atlas: LINEAR/Unorm. egui_wgpu's preferred surface format is
        // non-sRGB Unorm (see `egui_wgpu::preferred_framebuffer_format`), which
        // means the framebuffer doesn't gamma-encode on store. If we sampled
        // as sRGB the GPU would gamma-decode on read, producing linear values
        // that the surface then displays as-if-sRGB → too dark. Sampling raw
        // sRGB-encoded bytes (Unorm) matches what egui does for `ui.image`.
        let color_tex = create_texture(
            device,
            queue,
            color_atlas,
            "preview_color_atlas",
            wgpu::TextureFormat::Rgba8Unorm,
        );
        // Motion atlas: must be LINEAR. R8G8 stores raw motion bytes where 128 ≈ 0.5 ≈
        // zero motion. With sRGB the GPU would gamma-decode 0.5 → 0.215, breaking the
        // `(raw - 0.5) * 2 * strength` decode and warping every still pixel.
        let motion_tex = create_texture(
            device,
            queue,
            motion_atlas,
            "preview_motion_atlas",
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let motion_view = motion_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preview_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&motion_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                },
            ],
        });

        self.color_texture = Some(color_tex);
        self.motion_texture = Some(motion_tex);
        self.bind_group = Some(bind_group);
    }
}

/// Create a wgpu texture from an `ImageRgba8`.
fn create_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img: &ImageRgba8,
    label: &str,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width: img.width,
        height: img.height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &img.data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * img.width),
            rows_per_image: Some(img.height),
        },
        size,
    );
    texture
}

/// Custom paint callback for the preview shader.
pub struct PreviewCallback {
    /// Uniforms to upload before drawing.
    pub uniforms: Uniforms,
}

impl egui_wgpu::CallbackTrait for PreviewCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(resources) = callback_resources.get::<PreviewResources>() {
            queue.write_buffer(
                &resources.uniform_buffer,
                0,
                bytemuck::bytes_of(&self.uniforms),
            );
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(resources) = callback_resources.get::<PreviewResources>() {
            if let Some(ref bind_group) = resources.bind_group {
                render_pass.set_pipeline(&resources.pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);
                render_pass.draw(0..3, 0..1); // fullscreen triangle
            }
        }
    }
}

/// Initialize preview resources and insert them into the renderer's callback resources.
pub fn init_preview_resources(render_state: &egui_wgpu::RenderState) {
    let resources = PreviewResources::new(&render_state.device, render_state.target_format);
    render_state
        .renderer
        .write()
        .callback_resources
        .insert(resources);
}

/// Upload atlas textures into the preview resources held by the renderer.
/// Called on the `Done` state transition after generation completes.
pub fn upload_preview_textures(
    render_state: &egui_wgpu::RenderState,
    color_atlas: &ImageRgba8,
    motion_atlas: &ImageRgba8,
) {
    let mut renderer = render_state.renderer.write();
    if let Some(resources) = renderer.callback_resources.get_mut::<PreviewResources>() {
        resources.upload_textures(
            &render_state.device,
            &render_state.queue,
            color_atlas,
            motion_atlas,
        );
    }
}

/// Paint the preview into the given UI rect.
///
/// `egui_wgpu` downcasts `epaint::PaintCallback::callback` to its own
/// `egui_wgpu::Callback` wrapper before invoking `prepare`/`paint`. Wrap via
/// `Callback::new_paint_callback` — handing it a bare `PreviewCallback` makes
/// the downcast fail and the render pass silently skip our draw.
pub fn paint_preview(ui: &mut egui::Ui, rect: egui::Rect, uniforms: Uniforms) {
    let cb = egui_wgpu::Callback::new_paint_callback(rect, PreviewCallback { uniforms });
    ui.painter().add(cb);
}
