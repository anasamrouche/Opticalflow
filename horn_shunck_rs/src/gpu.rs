use std::sync::{Arc, mpsc::TryRecvError};

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
    BufferUsages, Color, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device,
    DeviceDescriptor, Extent3d, FragmentState, FrontFace, IndexFormat, Instance,
    InstanceDescriptor, MultisampleState, PipelineCompilationOptions, PipelineLayoutDescriptor,
    PolygonMode, PowerPreference, PrimitiveState, PrimitiveTopology, Queue, RenderPipeline,
    RenderPipelineDescriptor, RequestAdapterOptions, Sampler, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, Surface, SurfaceConfiguration, TexelCopyBufferLayout, TexelCopyTextureInfo,
    Texture, TextureDescriptor, TextureFormat, TextureUsages, VertexBufferLayout, VertexState,
    VertexStepMode,
    util::{BufferInitDescriptor, DeviceExt},
};
use winit::{
    application::ApplicationHandler,
    keyboard::{KeyCode::KeyQ, PhysicalKey::Code},
    window::Window,
};

use crate::camera::{CaptureRequest, CaptureThread, LuminanceFrame, RgbaFrame};

const VERTICES: [Vertex; 4] = [
    Vertex::new([-1.0, 1.0, 0.0], [0.0, 0.0]),
    Vertex::new([1.0, 1.0, 0.0], [1.0, 0.0]),
    Vertex::new([-1.0, -1.0, 0.0], [0.0, 1.0]),
    Vertex::new([1.0, -1.0, 0.0], [1.0, 1.0]),
];

const INDICES: [u16; 6] = [0, 2, 1, 1, 2, 3];

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coord: [f32; 2],
}

impl Vertex {
    const fn new(position: [f32; 3], tex_coord: [f32; 2]) -> Self {
        Self {
            position,
            tex_coord,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct FlowParams {
    alpha_squared: f32,
}

impl FlowParams {
    pub(crate) fn new(alpha_squared: f32) -> Self {
        Self { alpha_squared }
    }
}

struct GPUContext {
    device: Device,
    queue: Queue,
    surface: Surface<'static>,
    surface_config: SurfaceConfiguration,
}

impl GPUContext {
    async fn request(window: Arc<Window>) -> Self {
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::None,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                apply_limit_buckets: false,
            })
            .await
            .unwrap();
        let size = window.inner_size();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            desired_maximum_frame_latency: 2,
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default())
            .await
            .unwrap();

        Self {
            device,
            queue,
            surface,
            surface_config,
        }
    }
}

struct CameraTarget {
    texture: Texture,
    camera_bind_group: BindGroup,
}

pub struct State {
    context: GPUContext,
    window: Arc<Window>,
    capture: CaptureThread,
    camera_bind_group_layout: BindGroupLayout,
    compute_bind_group_a_to_b: BindGroup,
    compute_bind_group_b_to_a: BindGroup,
    sampler: Sampler,
    camera_pipeline: RenderPipeline,
    compute_pipeline: ComputePipeline,
    camera_target: Option<CameraTarget>,
    vertices_buffer: Buffer,
    indices_buffer: Buffer,
    flow_buffer: Buffer,
    previous_luminance_buffer: Buffer,
    current_luminance_buffer: Buffer,
    latest_luminance: Option<LuminanceFrame>,
}

impl State {
    pub async fn new(window: Arc<Window>, capture: CaptureThread, params: FlowParams) -> Self {
        let context = GPUContext::request(window.clone()).await;
        context
            .surface
            .configure(&context.device, &context.surface_config);

        let sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let camera_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("camera bind group layout"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let camera_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("camera pipeline layout"),
                    bind_group_layouts: &[Some(&camera_bind_group_layout)],
                    immediate_size: 0,
                });

        let shader = context.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("camera shader"),
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let camera_pipeline = context
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("camera pipeline"),
                layout: Some(&camera_pipeline_layout),
                vertex: VertexState {
                    module: &shader,
                    entry_point: Some("vertex_main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[Some(VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                        step_mode: VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2],
                    })],
                },
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: Some("fragment_main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: context.surface_config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

        let compute_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Compute bind group layout"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 3,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 4,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let compute_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("compute pipeline layout"),
                    bind_group_layouts: &[Some(&compute_bind_group_layout)],
                    immediate_size: 0,
                });

        let compute_pipeline = context
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("compute pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: Some("compute_main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            });

        let params_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Params buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: BufferUsages::UNIFORM,
        });

        let vertices_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Vertices buffer"),
            contents: bytemuck::cast_slice(&VERTICES),
            usage: BufferUsages::VERTEX,
        });

        let indices_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Indices buffer"),
            contents: bytemuck::cast_slice(&INDICES),
            usage: BufferUsages::INDEX,
        });

        let previous_luminance_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Previous luminance buffer"),
            contents: bytemuck::cast_slice::<u8, u8>(&[0; 1920 * 1080]),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let current_luminance_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Current luminance buffer"),
            contents: bytemuck::cast_slice::<u8, u8>(&[0; 1920 * 1080]),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let flow_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Flow buffer"),
            contents: bytemuck::cast_slice::<u8, u8>(&[0; 1920 * 1080 * 8]),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let next_flow_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Next flow buffer"),
            contents: bytemuck::cast_slice::<u8, u8>(&[0; 1920 * 1080 * 8]),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let compute_bind_group_a_to_b = context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Compute bind group A to B"),
            layout: &compute_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &previous_luminance_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &current_luminance_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &flow_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &next_flow_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let compute_bind_group_b_to_a = context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Compute bind group B to A"),
            layout: &compute_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &previous_luminance_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &current_luminance_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &flow_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &next_flow_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        Self {
            context,
            window,
            capture,
            camera_bind_group_layout,
            compute_bind_group_a_to_b,
            compute_bind_group_b_to_a,
            sampler,
            camera_pipeline,
            compute_pipeline,
            camera_target: None,
            previous_luminance_buffer,
            current_luminance_buffer,
            flow_buffer,
            vertices_buffer,
            indices_buffer,
            latest_luminance: None,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.context.surface_config.width = width;
            self.context.surface_config.height = height;
            self.context
                .surface
                .configure(&self.context.device, &self.context.surface_config);
        }
    }

    fn create_camera_target(&self, width: u32, height: u32) -> CameraTarget {
        let texture = self.context.device.create_texture(&TextureDescriptor {
            label: Some("Camera texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let camera_bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("camera bind group"),
                layout: &self.camera_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(BufferBinding {
                            buffer: &self.flow_buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                ],
            });

        CameraTarget {
            texture,
            camera_bind_group,
        }
    }

    fn upload_frame(&mut self, frame: &RgbaFrame) -> anyhow::Result<()> {
        let expected = 4 * frame.width as usize * frame.height as usize;
        anyhow::ensure!(
            frame.data.len() == expected,
            "frame {}x{} : {} octets au lieu de {} (pas du RGBA8 ?)",
            frame.width,
            frame.height,
            frame.data.len(),
            expected
        );

        let recreate = match &self.camera_target {
            Some(target) => {
                target.texture.width() != frame.width || target.texture.height() != frame.height
            }
            None => true,
        };
        if recreate {
            eprintln!(
                "[diag] texture caméra créée : {}x{}",
                frame.width, frame.height
            );
            self.camera_target = Some(self.create_camera_target(frame.width, frame.height));
        }

        let target = self.camera_target.as_ref().expect("créée juste au-dessus");
        self.context.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &frame.data,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * frame.width),
                rows_per_image: Some(frame.height),
            },
            Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        let mut latest_colored = None;
        let mut previous_luminance: Option<LuminanceFrame> = None;
        loop {
            match self.capture.receiver.try_recv() {
                Ok(CaptureRequest::Frame(colored_frame, luminance_frame)) => {
                    latest_colored = Some(colored_frame);
                    previous_luminance = self.latest_luminance.replace(luminance_frame);
                }
                Ok(CaptureRequest::Error(err)) => anyhow::bail!("thread de capture : {err}"),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    anyhow::bail!("le thread de capture s'est arrêté")
                }
            }
        }
        if let Some(frame) = latest_colored {
            self.upload_frame(&frame)?;
        }

        let (output, needs_reconfigure) = match self.context.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => (surface_texture, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => (surface_texture, true),
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.context
                    .surface
                    .configure(&self.context.device, &self.context.surface_config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                anyhow::bail!("get_current_texture : erreur de validation de la surface");
            }
            wgpu::CurrentSurfaceTexture::Lost => anyhow::bail!("surface perdue"),
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });
        {
            if let (Some(previous_luminance), Some(next_luminance)) =
                (&previous_luminance, &self.latest_luminance)
            {
                self.context.queue.write_buffer(
                    &self.previous_luminance_buffer,
                    0,
                    previous_luminance.data.as_slice(),
                );
                self.context.queue.write_buffer(
                    &self.current_luminance_buffer,
                    0,
                    next_luminance.data.as_slice(),
                );
                encoder.clear_buffer(&self.flow_buffer, 0, None);
                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Compute pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.compute_pipeline);

                let (width, height) = (
                    self.latest_luminance.as_ref().unwrap().height,
                    self.latest_luminance.as_ref().unwrap().height,
                );
                for _ in 0..10 {
                    compute_pass.set_bind_group(0, &self.compute_bind_group_a_to_b, &[]);
                    compute_pass.dispatch_workgroups(
                        2 * width.div_ceil(16),
                        height.div_ceil(16),
                        1,
                    );
                    compute_pass.set_bind_group(0, &self.compute_bind_group_b_to_a, &[]);
                    compute_pass.dispatch_workgroups(
                        2 * width.div_ceil(16),
                        height.div_ceil(16),
                        1,
                    );
                }
            };
        }
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            if let Some(target) = &self.camera_target {
                render_pass.set_pipeline(&self.camera_pipeline);
                render_pass.set_bind_group(0, &target.camera_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.vertices_buffer.slice(..));
                render_pass.set_index_buffer(self.indices_buffer.slice(..), IndexFormat::Uint16);
                render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);
            }
        }
        self.context.queue.submit(std::iter::once(encoder.finish()));
        self.context.queue.present(output);

        if needs_reconfigure {
            self.context
                .surface
                .configure(&self.context.device, &self.context.surface_config);
        }

        Ok(())
    }
}

pub struct Application {
    state: Option<State>,
    params: FlowParams,
}

impl Application {
    pub fn new(params: FlowParams) -> Self {
        Self {
            state: None,
            params,
        }
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window_attributes = Window::default_attributes().with_title("Horn-Schunck");
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        nokhwa::nokhwa_initialize(|_| {});
        let capture = CaptureThread::spawn();

        self.state = Some(pollster::block_on(State::new(
            window.clone(),
            capture,
            self.params,
        )));
        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            winit::event::WindowEvent::CloseRequested => event_loop.exit(),
            winit::event::WindowEvent::Resized(size) => state.resize(size.width, size.height),
            winit::event::WindowEvent::RedrawRequested => {
                if let Err(err) = state.render() {
                    eprintln!("[erreur] {err:#}");
                    event_loop.exit();
                    return;
                }
                state.window.request_redraw();
            }
            winit::event::WindowEvent::KeyboardInput { event, .. } => {
                if let Code(KeyQ) = event.physical_key {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }
}
