use pyo3::prelude::*;

type Pixel = f32;


#[pymodule]
mod horn_schunck_rs {
    use anyhow::Ok;
    use pyo3::prelude::*;
    use numpy::{IntoPyArray, PyReadonlyArray3, ndarray::{Array2, Array3, ArrayView2, Axis}, PyArray3};
    use wgpu::{BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BufferUsages, ComputePassDescriptor, ComputePipelineDescriptor, Device, ExperimentalFeatures, Features, InstanceDescriptor, InstanceFlags, Limits, MemoryBudgetThresholds, PipelineCompilationOptions, PipelineLayoutDescriptor, Queue, RequestAdapterOptions, ShaderModule, ShaderModuleDescriptor, ShaderStages, util::{BufferInitDescriptor, DeviceExt}, wgc::device, wgt::{CommandEncoderDescriptor, DeviceDescriptor}};
    
    use crate::{utilities::{get_average, space_derive, time_derive}};

    struct GPU {
        device: Device,
        queue: Queue,
    }
    
    async fn set_up() -> anyhow::Result<GPU> {
        let instance = wgpu::Instance::new(
                &InstanceDescriptor{
                    backends: wgpu::Backends::PRIMARY,
                    flags: InstanceFlags::DEBUG | InstanceFlags::VALIDATION,
                    ..Default::default()
                }
            );
    
        let adapter = instance.request_adapter(
            &RequestAdapterOptions{
                power_preference: wgpu::PowerPreference::None,
                force_fallback_adapter: false,
                compatible_surface: None
            }
        ).await?;

        let (device, queue) = adapter.request_device(
            &DeviceDescriptor{
                label: Some("Device"),
                required_features: Features::empty(),
                experimental_features: ExperimentalFeatures::disabled(),
                required_limits: Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            }
        ).await?;

        Ok(GPU {device: device, queue: queue})
    }

    fn gauss_seidel(image1: ArrayView2<'_, f32>, image2: ArrayView2<'_, f32>, alpha_squared: f32, max_iter: u32) -> (Array2<f32>, Array2<f32>) {
        let image_height = image1.shape()[0];
        let image_width = image1.shape()[1];

        let mut u_field = Array2::<f32>::zeros((image_height, image_width));
        let mut v_field = Array2::<f32>::zeros((image_height, image_width));

        for _ in 0..max_iter {
            let mut x_derivative = Array2::<f32>::zeros((image_height, image_width));
            let mut y_derivative = Array2::<f32>::zeros((image_height, image_width));
            let mut time_derivative = Array2::<f32>::zeros((image_height, image_width));

            for x in 0..image_height {
                for y in 0..image_width {
                    let (dx, dy) = space_derive(image1, x, y);
                    x_derivative[[x, y]] = dx;
                    y_derivative[[x, y]] = dy;
                    time_derivative[[x, y]] = time_derive(image1, image2, x, y);
                }
            }

            for x in 0..image_height {
                for y in 0..image_width {
                    let u_average = get_average(u_field.view(), x, y);
                    let v_average = get_average(v_field.view(), x, y);

                    u_field[[x, y]] = u_average - x_derivative[[x, y]] * (x_derivative[[x, y]] * u_average + y_derivative[[x, y]] * v_average + time_derivative[[x, y]])/(alpha_squared + x_derivative[[x, y]].powf(2.0) + y_derivative[[x, y]].powf(2.0));
                    v_field[[x, y]] = v_average - y_derivative[[x, y]] * (x_derivative[[x, y]] * u_average + y_derivative[[x, y]] * v_average + time_derivative[[x, y]])/(alpha_squared + x_derivative[[x, y]].powf(2.0) + y_derivative[[x, y]].powf(2.0));
                }
            }
        }


        (
            u_field,
            v_field
        )
    }

    #[pyfunction]
    fn solve_gauss_seidel<'py>(
            py: Python<'_>,
            video: PyReadonlyArray3<f32, '_>,
            alpha_squared: f32,
            max_iter: u32,
        )
        -> (Py<PyArray3<f32>>, Py<PyArray3<f32>>) {
        let video_array = video.as_array();

        let (frame_count, frame_height, frame_width) = (video_array.shape()[0], video_array.shape()[1], video_array.shape()[2]);

        let mut u_field = Array3::<f32>::zeros((frame_count, frame_height, frame_width));
        let mut v_field = Array3::<f32>::zeros((frame_count, frame_height, frame_width));

        for k in 0..frame_count-1 {
            let current_frame = video_array.index_axis(Axis(0), k);
            let next_frame = video_array.index_axis(Axis(0), k+1);

            let (u, v) = gauss_seidel(current_frame, next_frame, alpha_squared, max_iter);

            u_field.index_axis_mut(Axis(0), k).assign(&u);
            v_field.index_axis_mut(Axis(0), k).assign(&v);
        }

        (
            u_field.into_pyarray(py).unbind(),
            v_field.into_pyarray(py).unbind()
        )
    }

    async fn solve_jacobi<'py>(
            py: Python<'_>,
            video: PyReadonlyArray3<f32, '_>,
            alpha_squared: f32,
            max_iter: u32,
    )
    // -> (Py<PyArray3<f32>>, Py<PyArray3<f32>>) 
    {
        let gpu = set_up().await.unwrap();

        let video_array = video.as_array();

        let (frame_count, frame_height, frame_width) = (video_array.shape()[0], video_array.shape()[1], video_array.shape()[2]);

        let mut u_field = Array3::<f32>::zeros((frame_count, frame_height, frame_width));
        let mut v_field = Array3::<f32>::zeros((frame_count, frame_height, frame_width));

        let num_iterations = frame_count/30usize;

        for k in 0..num_iterations {
            let lower_bound: usize = 30 * k;
            let mut upper_bound: usize;

            match frame_count {
                frame_count if frame_count <= 30 * (k+1) => {upper_bound = frame_count - lower_bound;}
                _ => {upper_bound = 30 * (k+1);}
            }

            
        }
    }

    // async fn jacobi(gpu: &GPU) {
    //     let u_slice = u_field.as_slice().unwrap();
    //     let v_slice = v_field.as_slice().unwrap();
    
    //     let u_buffer = gpu.device.create_buffer_init(
    //         &BufferInitDescriptor {
    //             label: Some("U field"),
    //             contents: bytemuck::cast_slice(u_slice),
    //             usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    //         }
    //     );
    
    //     let v_buffer = gpu.device.create_buffer_init(
    //         &BufferInitDescriptor {
    //             label: Some("V field"),
    //             contents: bytemuck::cast_slice(v_slice),
    //             usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    //         }
    //     );
    
    //     let u_layout = gpu.device.create_bind_group_layout(
    //         &BindGroupLayoutDescriptor {
    //             label: Some("U layout descriptor"),
    //             entries: &[
    //                 BindGroupLayoutEntry {
    //                     binding: 0,
    //                     visibility: ShaderStages::COMPUTE,
    //                     ty: wgpu::BindingType::Buffer {
    //                         ty: wgpu::BufferBindingType::Storage {
    //                             read_only: false
    //                         },
    //                         has_dynamic_offset: false,
    //                         min_binding_size: None
    //                     },
    //                     count: None
    //                 }
    //             ]
    //         }
    //     );
    
    //     let v_layout = gpu.device.create_bind_group_layout(
    //         &BindGroupLayoutDescriptor {
    //             label: Some("V layout descriptor"),
    //             entries: &[
    //                 BindGroupLayoutEntry {
    //                     binding: 1,
    //                     visibility: ShaderStages::COMPUTE,
    //                     ty: wgpu::BindingType::Buffer {
    //                         ty: wgpu::BufferBindingType::Storage {
    //                             read_only: false
    //                         },
    //                         has_dynamic_offset: false,
    //                         min_binding_size: None
    //                     },
    //                     count: None
    //                 }
    //             ]
    //         }
    //     );
    
    //     let u_bindgroup = gpu.device.create_bind_group(
    //         &BindGroupDescriptor {
    //             label: Some("U bind group"),
    //             layout: &u_layout,
    //             entries: &[
    //                 BindGroupEntry {
    //                     binding: 0,
    //                     resource: u_buffer.as_entire_binding()
    //                 }
    //             ]
    //         },
    //     );
    
    //     let v_bindgroup = gpu.device.create_bind_group(
    //         &BindGroupDescriptor {
    //             label: Some("V bind group"),
    //             layout: &v_layout,
    //             entries: &[
    //                 BindGroupEntry {
    //                     binding: 1,
    //                     resource: v_buffer.as_entire_binding()
    //                 }
    //             ]
    //         },
    //     );
    
    //     let pipeline_layout = gpu.device.create_pipeline_layout(
    //         &PipelineLayoutDescriptor {
    //             label: Some("Compute pipeline descriptor"),
    //             bind_group_layouts: &[
    //                 &u_layout,
    //                 &v_layout
    //             ],
    //             immediate_size: 0
    //         }
    //     );
    
    //     let compute_module = gpu.device.create_shader_module(
    //         ShaderModuleDescriptor {
    //             label: Some("Compute shader module"),
    //             source: wgpu::ShaderSource::Wgsl("./jacobi_method.wgsl".into())
    //         }
    //     );
    
    //     let compute_pipeline = gpu.device.create_compute_pipeline(
    //         &ComputePipelineDescriptor {
    //             label: Some("Compute pipeline"),
    //             layout: Some(&pipeline_layout),
    //             entry_point: Some("main"),
    //             module: &compute_module,
    //             compilation_options: PipelineCompilationOptions { 
    //                 constants: &[],
    //                 zero_initialize_workgroup_memory: false
    //             },
    //             cache: None
    //         }
    //     );
    
    //     let mut encoder = gpu.device.create_command_encoder(
    //         &CommandEncoderDescriptor {
    //             label: Some("Compute command encoder"),
    //         }
    //     );
    
    //     {
    //         let mut compute_pass = encoder.begin_compute_pass(
    //             &ComputePassDescriptor {
    //                 label: Some("Compute pass"),
    //                 timestamp_writes: None,
    //             }
    //         );
    //         compute_pass.set_pipeline(&compute_pipeline);
    //         compute_pass.set_bind_group(0, &u_bindgroup, &[]);
    //         compute_pass.set_bind_group(0, &v_bindgroup, &[]);
    //     }
    
    //     gpu.queue.submit(std::iter::once(encoder.finish()));

    // }

    fn gradient_descent(image1: ArrayView2<'_, f32>, image2: ArrayView2<'_, f32>, alpha_squared: f32, step: f32, max_iter: u32, norm_l2: bool) -> (Array2<f32>, Array2<f32>) {
        if norm_l2 {
            let image_height = image1.shape()[0];
            let image_width = image1.shape()[1];
    
            let mut u_field = Array2::<f32>::zeros((image_height, image_width));
            let mut v_field = Array2::<f32>::zeros((image_height, image_width));
    
            let get_cross_pattern = |field: &Array2<f32>, x_index: usize, y_index: usize| -> f32 {
                let x_previous = x_index.saturating_sub(1).clamp(0, image_height - 1);
                let x_next = (x_index + 1).min(image_height - 1);
    
                let y_previous = y_index.saturating_sub(1).clamp(0, image_width - 1);
                let y_next = (y_index + 1).min(image_width - 1);
    
                field[[x_previous, y_index]] + field[[x_next, y_index]] + field[[x_index, y_previous]] + field[[x_index, y_next]]
            };
    
            for _ in 0..max_iter {
                for x_index in 0..image_height {
                    for y_index in 0..image_width {
                        let (Ix, Iy) = space_derive(image1, x_index, y_index);
                        let It = time_derive(image1, image2, x_index, y_index);
    
                        u_field[[x_index, y_index]] -= step * 2.0 * (Ix * (Ix * u_field[[x_index, y_index]] + Iy * v_field[[x_index, y_index]] + It) - alpha_squared * get_cross_pattern(&u_field, x_index, y_index));
                        v_field[[x_index, y_index]] -= step * 2.0 * (Iy * (Ix * u_field[[x_index, y_index]] + Iy * v_field[[x_index, y_index]] + It) - alpha_squared * get_cross_pattern(&v_field, x_index, y_index));
                    }
                }
            }
    
            (u_field, v_field)
        }
        else {
            let image_height = image1.shape()[0];
            let image_width = image1.shape()[1];
    
            let mut u_field = Array2::<f32>::zeros((image_height, image_width));
            let mut v_field = Array2::<f32>::zeros((image_height, image_width));
    
            let get_cross_pattern = |field: &Array2<f32>, x_index: usize, y_index: usize| -> f32 {
                let x_previous = x_index.saturating_sub(2).clamp(0, image_height - 1);
                let x_next = (x_index + 2).min(image_height - 1);
    
                let y_previous = y_index.saturating_sub(2).clamp(0, image_width - 1);
                let y_next = (y_index + 2).min(image_width - 1);
    
                (field[[x_next, y_index]] - field[[x_index, y_index]]).signum() + (field[[x_index, y_index]] - field[[x_previous, y_index]]).signum() + (field[[x_index, y_next]] - field[[x_index, y_index]]).signum() + (field[[x_index, y_index]] - field[[x_index, y_previous]]).signum()
            };
    
            for _ in 0..max_iter {
                for x_index in 0..image_height {
                    for y_index in 0..image_width {
                        let (Ix, Iy) = space_derive(image1, x_index, y_index);
                        let It = time_derive(image1, image2, x_index, y_index);
    
                        u_field[[x_index, y_index]] -= step * 2.0 * (Ix * (Ix * u_field[[x_index, y_index]] + Iy * v_field[[x_index, y_index]] + It) - alpha_squared * get_cross_pattern(&u_field, x_index, y_index));
                        v_field[[x_index, y_index]] -= step * 2.0 * (Iy * (Ix * u_field[[x_index, y_index]] + Iy * v_field[[x_index, y_index]] + It) - alpha_squared * get_cross_pattern(&v_field, x_index, y_index));
                    }
                }
            }

            (u_field, v_field)
        }
    }

    #[pyfunction]
    fn solve_gradient_descent<'py>(
            py: Python<'_>,
            video: PyReadonlyArray3<f32, '_>,
            alpha_squared: f32,
            step: f32,
            max_iter: u32,
            norm_l2: bool
        )
        -> (Py<PyArray3<f32>>, Py<PyArray3<f32>>) {
            let video_array = video.as_array();

            let (frame_count, frame_height, frame_width) = (video_array.shape()[0], video_array.shape()[1], video_array.shape()[2]);

            let mut u_field = Array3::<f32>::zeros((frame_count, frame_height, frame_width));
            let mut v_field = Array3::<f32>::zeros((frame_count, frame_height, frame_width));

            for k in 0..frame_count-1 {
                let current_frame = video_array.index_axis(Axis(0), k);
                let next_frame = video_array.index_axis(Axis(0), k+1);

                let (u, v) = gradient_descent(current_frame, next_frame, alpha_squared, step, max_iter, norm_l2);

                u_field.index_axis_mut(Axis(0), k).assign(&u);
                v_field.index_axis_mut(Axis(0), k).assign(&v);
            }

            (
                u_field.into_pyarray(py).unbind(),
                v_field.into_pyarray(py).unbind()
            )
        }

}

mod utilities {
    use super::{Pixel};
    use numpy::ndarray::ArrayView2;

    
    pub fn space_derive(image: ArrayView2<'_, f32>, x: usize, y: usize) -> (Pixel, Pixel) {
        let image_height = image.shape()[0];
        let image_width = image.shape()[1];

        //Remplacement du bloc conditionnel par du clamping. Ça devrait permettre au compilateur d'appliquer des optimisations
        //Impossibles à mettre en place avex les blocs match (SIMD)
        let x_previous = x.saturating_sub(1);
        let x_next = (x + 1).min(image_height - 1);
        
        let y_previous = y.saturating_sub(1);
        let y_next = (y + 1).min(image_width - 1);

        //Le problème est que les bords de l'image sont un cas particulier à traiter.
        //Si on est à côté des bords, il ne faut plus diviser la différence.
        //Si on est aux bords, on applique une condition de Neumann pour que la dérivée soit nulle.
        let x_denominator = (x_next - x_previous) as f32;
        let y_denominator = (y_next - y_previous) as f32;

        let dx = if x_denominator > 0.0 { (image[[x_next, y]] - image[[x_previous, y]])/x_denominator } else { 0.0 };
        let dy = if y_denominator > 0.0 { (image[[x, y_next]] - image[[x, y_previous]])/y_denominator } else { 0.0 };

        (dx, dy)
    }

    pub fn time_derive(current_image: ArrayView2<'_, f32>, next_image: ArrayView2<'_, f32>, x: usize, y: usize) -> Pixel {
        next_image[[x, y]] - current_image[[x, y]]
    }
    
    pub fn get_average(image: ArrayView2<'_, f32>, x: usize, y: usize) -> Pixel {
        let image_height = image.shape()[0];
        let image_width = image.shape()[1];

        let get_clamped = |x_index: usize, y_index: usize| -> f32 {
            let x_clamped = x_index.clamp(0, image_height - 1);
            let y_clamped = y_index.clamp(0, image_width - 1);

            return image[[x_clamped, y_clamped]];
        };

        let closer_pixels = (
            get_clamped(x.saturating_sub(1), y) + get_clamped(x+1, y) + get_clamped(x, y.saturating_sub(1)) + get_clamped(x, y+1)
        )/6.0;
        let further_pixels = (
            get_clamped(x.saturating_sub(1), y.saturating_sub(1)) + get_clamped(x+1, y.saturating_sub(1)) + get_clamped(x+1, y+1) + get_clamped(x.saturating_sub(1), y+1)
        )/12.0;

        closer_pixels + further_pixels
    }

    // pub fn add_pixels(pixel1: &Pixel, pixel2: &Pixel) -> Pixel {
    //     let mut returned_DEFAULT_PIXEL: Pixel = DEFAULT_PIXEL;
    //     returned_DEFAULT_PIXEL
    // }

    // pub fn substract_DEFAULT_PIXELs(DEFAULT_PIXEL1: &DEFAULT_PIXEL, DEFAULT_PIXEL2: &DEFAULT_PIXEL) -> DEFAULT_PIXEL {
    //     let mut returned_DEFAULT_PIXEL: DEFAULT_PIXEL = DEFAULT_PIXEL;
    //     for k in 0..3 {
    //         returned_DEFAULT_PIXEL = DEFAULT_PIXEL1 - DEFAULT_PIXEL2;
    //     }
    //     returned_DEFAULT_PIXEL
    // }

    // pub fn divide_DEFAULT_PIXEL(DEFAULT_PIXEL: &DEFAULT_PIXEL, divisor: f32) -> DEFAULT_PIXEL {
    //     let mut returned_DEFAULT_PIXEL: DEFAULT_PIXEL = DEFAULT_PIXEL;
    //     for k in 0..3 {
    //         returned_DEFAULT_PIXEL = DEFAULT_PIXEL/divisor;
    //     }
    //     returned_DEFAULT_PIXEL
    // }
}

// mod kernel {
//     use pyo3::prelude::*;

//     struct Kernel<const S: usize> {
//         values: [[f32; S]; S]
//     }

//     struct Image<const S: usize> {
//         size: usize,
//         content: [[DEFAULT_PIXEL; S]; S]
//     }

//     impl<const S: usize> Image<S> {
//         fn convolution<const K: usize>(self, kernel: Kernel<K>) {

//         }
//     }
// }