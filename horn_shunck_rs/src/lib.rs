use pyo3::prelude::*;

type Pixel = f32;


#[pymodule]
mod horn_schunck_rs {
    use anyhow::Ok;
    use pyo3::prelude::*;
    use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray3, ToPyArray, ndarray::{Array2, Array3, ArrayView2, Axis}};
    use wgpu::{BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BufferUsages, ComputePassDescriptor, ComputePipelineDescriptor, Device, ExperimentalFeatures, Features, InstanceDescriptor, InstanceFlags, Limits, MemoryBudgetThresholds, PipelineCompilationOptions, PipelineLayoutDescriptor, Queue, RequestAdapterOptions, ShaderModule, ShaderModuleDescriptor, ShaderStages, util::{BufferInitDescriptor, DeviceExt}, wgc::device, wgt::{CommandEncoderDescriptor, DeviceDescriptor}};
    
    use crate::{utilities::{get_average, space_derive, time_derive}};

    // #[pyclass]
    // struct Adam {
    //     step_size: f32,
    //     beta1: f32,
    //     beta2: f32,
    //     tol: f32,
    //     first_moment: Vec<f32>,
    //     second_moment: Vec<f32>,
    //     step: u32
    // }

    // #[pymethods]
    // impl Adam {
    //     #[new]
    //     #[pyo3(signature =(step_size=1e-3, beta1=0.99, beta2=0.999, tol=1e-8))]
    //     fn new(step_size: f32, beta1: f32, beta2: f32, tol: f32) -> Self {
    //         Self { step_size, beta1, beta2, tol, first_moment: Vec::new(), second_moment: Vec::new(), step:0 }
    //     }
    // }

    // struct GPU {
    //     device: Device,
    //     queue: Queue,
    // }
    
    // async fn set_up() -> anyhow::Result<GPU> {
    //     let instance = wgpu::Instance::new(
    //             &InstanceDescriptor{
    //                 backends: wgpu::Backends::PRIMARY,
    //                 flags: InstanceFlags::DEBUG | InstanceFlags::VALIDATION,
    //                 ..Default::default()
    //             }
    //         );
    
    //     let adapter = instance.request_adapter(
    //         &RequestAdapterOptions{
    //             power_preference: wgpu::PowerPreference::None,
    //             force_fallback_adapter: false,
    //             compatible_surface: None
    //         }
    //     ).await?;

    //     let (device, queue) = adapter.request_device(
    //         &DeviceDescriptor{
    //             label: Some("Device"),
    //             required_features: Features::empty(),
    //             experimental_features: ExperimentalFeatures::disabled(),
    //             required_limits: Limits::default(),
    //             memory_hints: Default::default(),
    //             trace: wgpu::Trace::Off,
    //         }
    //     ).await?;

    //     Ok(GPU {device: device, queue: queue})
    // }

    fn gauss_seidel(image1: ArrayView2<'_, f32>, image2: ArrayView2<'_, f32>, alpha_squared: f32, max_iter: u32) -> (Array2<f32>, Array2<f32>) {
        let image_height = image1.shape()[0];
        let image_width = image1.shape()[1];

        let mut u_field = Array2::<f32>::zeros((image_height, image_width));
        let mut v_field = Array2::<f32>::zeros((image_height, image_width));
        
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
        for _ in 0..max_iter {
            for x in 0..image_height {
                for y in 0..image_width {
                    let u_average = get_average(u_field.view(), x, y);
                    let v_average = get_average(v_field.view(), x, y);

                    u_field[[x, y]] = u_average - x_derivative[[x, y]] * (x_derivative[[x, y]] * u_average + y_derivative[[x, y]] * v_average + time_derivative[[x, y]])/(alpha_squared + x_derivative[[x, y]].powi(2) + y_derivative[[x, y]].powi(2));
                    v_field[[x, y]] = v_average - y_derivative[[x, y]] * (x_derivative[[x, y]] * u_average + y_derivative[[x, y]] * v_average + time_derivative[[x, y]])/(alpha_squared + x_derivative[[x, y]].powi(2) + y_derivative[[x, y]].powi(2));
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

    fn gradient_descent(image1: ArrayView2<'_, f32>, image2: ArrayView2<'_, f32>, alpha_squared: f32, step: f32, max_iter: u32, tol: f32, normL2: bool) -> (Array2<f32>, Array2<f32>, u32) {
        if normL2 {
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

            let get_gradient_norm = |field: &Array2<f32>, x_index: usize, y_index: usize| -> f32 {
                let x_previous = x_index.saturating_sub(1).clamp(0, image_height - 1);
                let x_next = (x_index + 1).min(image_height - 1);
    
                let y_previous = y_index.saturating_sub(1).clamp(0, image_width - 1);
                let y_next = (y_index + 1).min(image_width - 1);
    
                (field[[x_next, y_index]] - field[[x_previous, y_index]]).powi(2)/4.0 + (field[[x_index, y_next]] - field[[x_index, y_previous]]).powi(2)/4.0
            };

            let mut count = 0;
            for _ in 0..max_iter {
                count += 1;
                let mut previous_evaluation: f32 = 0.0;
                let mut next_evaluation: f32 = 0.0;
                for x_index in 0..image_height {
                    for y_index in 0..image_width {
                        let (Ix, Iy) = space_derive(image1, x_index, y_index);
                        let It = time_derive(image1, image2, x_index, y_index);
                        
                        next_evaluation += (Ix * u_field[[x_index, y_index]] + Iy * v_field[[x_index, y_index]] + It).powi(2) + get_gradient_norm(&u_field, x_index, y_index) + get_gradient_norm(&v_field, x_index, y_index);
                        u_field[[x_index, y_index]] -= step * 2.0 * (Ix * (Ix * u_field[[x_index, y_index]] + Iy * v_field[[x_index, y_index]] + It) - alpha_squared * (get_cross_pattern(&u_field, x_index, y_index) - 4.0 * u_field[[x_index, y_index]]));
                        v_field[[x_index, y_index]] -= step * 2.0 * (Iy * (Ix * u_field[[x_index, y_index]] + Iy * v_field[[x_index, y_index]] + It) - alpha_squared * (get_cross_pattern(&v_field, x_index, y_index) - 4.0 * v_field[[x_index, y_index]]));
                    }
                }
                if (next_evaluation - previous_evaluation).abs() < tol {break};
                previous_evaluation = next_evaluation;
                next_evaluation = 0.0;
            }
    
            (u_field, v_field, count)
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

            let mut count = 0;
            for _ in 0..max_iter {
                count += 1;
                for x_index in 0..image_height {
                    for y_index in 0..image_width {
                        let (Ix, Iy) = space_derive(image1, x_index, y_index);
                        let It = time_derive(image1, image2, x_index, y_index);
    
                        u_field[[x_index, y_index]] -= step * 2.0 * (Ix * (Ix * u_field[[x_index, y_index]] + Iy * v_field[[x_index, y_index]] + It) - alpha_squared * (get_cross_pattern(&u_field, x_index, y_index) - 4.0 * u_field[[x_index, y_index]]));
                        v_field[[x_index, y_index]] -= step * 2.0 * (Iy * (Ix * u_field[[x_index, y_index]] + Iy * v_field[[x_index, y_index]] + It) - alpha_squared * (get_cross_pattern(&v_field, x_index, y_index) - 4.0 * v_field[[x_index, y_index]]));
                    }
                }
            }

            (u_field, v_field, count)
        }
    }

    #[pyfunction]
    fn solve_gradient_descent<'py>(
            py: Python<'_>,
            video: PyReadonlyArray3<f32, '_>,
            alpha_squared: f32,
            step: f32,
            max_iter: u32,
            tol: f32,
            normL2: bool
        )
        -> (Py<PyArray3<f32>>, Py<PyArray3<f32>>, Py<PyArray1<u32>>) {
            let video_array = video.as_array();

            let (frame_count, frame_height, frame_width) = (video_array.shape()[0], video_array.shape()[1], video_array.shape()[2]);

            let mut u_field = Array3::<f32>::zeros((frame_count, frame_height, frame_width));
            let mut v_field = Array3::<f32>::zeros((frame_count, frame_height, frame_width));

            let mut counts: Vec<u32> = Vec::new();
            for k in 0..frame_count-1 {
                let current_frame = video_array.index_axis(Axis(0), k);
                let next_frame = video_array.index_axis(Axis(0), k+1);

                let (u, v, count) = gradient_descent(current_frame, next_frame, alpha_squared, step, max_iter, tol, normL2);

                u_field.index_axis_mut(Axis(0), k).assign(&u);
                v_field.index_axis_mut(Axis(0), k).assign(&v);

                counts.push(count);
            }

            (
                u_field.into_pyarray(py).unbind(),
                v_field.into_pyarray(py).unbind(),
                counts.to_pyarray(py).unbind()
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
}