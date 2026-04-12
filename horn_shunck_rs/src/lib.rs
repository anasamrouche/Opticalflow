use pyo3::prelude::*;
use numpy::{PyArray3, ndarray::Array2};

type Fields = (Array2<f64>, Array2<f64>);
type Video = (Py<PyArray3<f64>>, Py<PyArray3<f64>>);

#[pymodule]
mod horn_schunck_rs {
    use ndarray::Array2;
    use pyo3::prelude::*;
    use numpy::{IntoPyArray, PyReadonlyArray3,  ndarray::{Array3, ArrayView2, Axis}};
    
    use crate::{Fields, Video, utilities::{downscale_recursively, expand, get_average, space_derive, time_derive}};

    fn gauss_seidel(image1: ArrayView2<'_, f64>, image2: ArrayView2<'_, f64>, alpha_squared: f64, max_iter: u32) -> Fields {
        let image_height = image1.shape()[0];
        let image_width = image1.shape()[1];

        let mut u_field = Array2::<f64>::zeros((image_height, image_width));
        let mut v_field = Array2::<f64>::zeros((image_height, image_width));
        
        let mut x_derivative = Array2::<f64>::zeros((image_height, image_width));
        let mut y_derivative = Array2::<f64>::zeros((image_height, image_width));
        let mut time_derivative = Array2::<f64>::zeros((image_height, image_width));
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
            video: PyReadonlyArray3<'_, f64>,
            alpha_squared: f64,
            max_iter: u32,
        )
        -> Video {
        let video_array = video.as_array();

        let (frame_count, frame_height, frame_width) = (video_array.shape()[0], video_array.shape()[1], video_array.shape()[2]);

        let mut u_field = Array3::<f64>::zeros((frame_count, frame_height, frame_width));
        let mut v_field = Array3::<f64>::zeros((frame_count, frame_height, frame_width));

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

    fn gradient_descent(image1: ArrayView2<'_, f64>, image2: ArrayView2<'_, f64>, alpha_squared: f64, step: f64, max_iter: u32, norm_l2: bool) -> Fields {
        if norm_l2 {
            let image_height = image1.shape()[0];
            let image_width = image1.shape()[1];
    
            let mut u_field = Array2::<f64>::zeros((image_height, image_width));
            let mut v_field = Array2::<f64>::zeros((image_height, image_width));
    
            let get_cross_pattern = |field: &Array2<f64>, x_index: usize, y_index: usize| -> f64 {
                let x_previous = x_index.saturating_sub(1).clamp(0, image_height - 1);
                let x_next = (x_index + 1).min(image_height - 1);
    
                let y_previous = y_index.saturating_sub(1).clamp(0, image_width - 1);
                let y_next = (y_index + 1).min(image_width - 1);
    
                field[[x_previous, y_index]] + field[[x_next, y_index]] + field[[x_index, y_previous]] + field[[x_index, y_next]]
            };

            for _ in 0..max_iter {
                for x_index in 0..image_height {
                    for y_index in 0..image_width {
                        let (ix, iy) = space_derive(image1, x_index, y_index);
                        let it = time_derive(image1, image2, x_index, y_index);
                        
                        u_field[[x_index, y_index]] -= step * 2.0 * (ix * (ix * u_field[[x_index, y_index]] + iy * v_field[[x_index, y_index]] + it) - alpha_squared * (get_cross_pattern(&u_field, x_index, y_index) - 4.0 * u_field[[x_index, y_index]]));
                        v_field[[x_index, y_index]] -= step * 2.0 * (iy * (ix * u_field[[x_index, y_index]] + iy * v_field[[x_index, y_index]] + it) - alpha_squared * (get_cross_pattern(&v_field, x_index, y_index) - 4.0 * v_field[[x_index, y_index]]));
                    }
                }
            }
    
            (u_field, v_field,)
        }
        else {
            let image_height = image1.shape()[0];
            let image_width = image1.shape()[1];
    
            let mut u_field = Array2::<f64>::zeros((image_height, image_width));
            let mut v_field = Array2::<f64>::zeros((image_height, image_width));
    
            let get_cross_pattern = |field: &Array2<f64>, x_index: usize, y_index: usize| -> f64 {
                let x_previous = x_index.saturating_sub(2).clamp(0, image_height - 1);
                let x_next = (x_index + 2).min(image_height - 1);
    
                let y_previous = y_index.saturating_sub(2).clamp(0, image_width - 1);
                let y_next = (y_index + 2).min(image_width - 1);
    
                (field[[x_next, y_index]] - field[[x_index, y_index]]).signum() + (field[[x_index, y_index]] - field[[x_previous, y_index]]).signum() + (field[[x_index, y_next]] - field[[x_index, y_index]]).signum() + (field[[x_index, y_index]] - field[[x_index, y_previous]]).signum()
            };

            for _ in 0..max_iter {
                for x_index in 0..image_height {
                    for y_index in 0..image_width {
                        let (ix, iy) = space_derive(image1, x_index, y_index);
                        let it = time_derive(image1, image2, x_index, y_index);
    
                        u_field[[x_index, y_index]] -= step * 2.0 * (ix * (ix * u_field[[x_index, y_index]] + iy * v_field[[x_index, y_index]] + it) - alpha_squared * (get_cross_pattern(&u_field, x_index, y_index) - 4.0 * u_field[[x_index, y_index]]));
                        v_field[[x_index, y_index]] -= step * 2.0 * (iy * (ix * u_field[[x_index, y_index]] + iy * v_field[[x_index, y_index]] + it) - alpha_squared * (get_cross_pattern(&v_field, x_index, y_index) - 4.0 * v_field[[x_index, y_index]]));
                    }
                }
            }

            (u_field, v_field)
        }
    }

    #[pyfunction]
    fn solve_gradient_descent<'py>(
            py: Python<'_>,
            video: PyReadonlyArray3<'_, f64>,
            alpha_squared: f64,
            step: f64,
            max_iter: u32,
            norm_l2: bool
        )
        -> Video  {
            let video_array = video.as_array();

            let (frame_count, frame_height, frame_width) = (video_array.shape()[0], video_array.shape()[1], video_array.shape()[2]);

            let mut u_field = Array3::<f64>::zeros((frame_count, frame_height, frame_width));
            let mut v_field = Array3::<f64>::zeros((frame_count, frame_height, frame_width));

            for k in 0..frame_count-1 {
                let current_frame = video_array.index_axis(Axis(0), k);
                let next_frame = video_array.index_axis(Axis(0), k+1);

                let (u, v) = gradient_descent(current_frame, next_frame, alpha_squared, step, max_iter, norm_l2);

                u_field.index_axis_mut(Axis(0), k).assign(&u);
                v_field.index_axis_mut(Axis(0), k).assign(&v);
            }

            (
                u_field.into_pyarray(py).unbind(),
                v_field.into_pyarray(py).unbind(),
            )
        }

    fn gauss_seidel_from_previous_uv(image1: ArrayView2<'_, f64>, image2: ArrayView2<'_, f64>, prev_u_field: Array2<f64>, prev_v_field: Array2<f64>, alpha_squared: f64, iter_max: u32) -> Fields {
        // Cette fonction sert à généraliser Gauss-Seidel dans les cas où un champ précédent a été calculé de façon pyramidale
        // Note pour plus tard : Je devrais peut-être l'utiliser à la place de la première méthode que j'ai mise en place car celle-ci est plus générale et un seule fonction rendrait le programme plus lisible.
        let image_height = image1.shape()[0];
        let image_width = image1.shape()[1];

        let mut u_field = prev_u_field;
        let mut v_field = prev_v_field;
        
        let mut x_derivative = Array2::<f64>::zeros((image_height, image_width));
        let mut y_derivative = Array2::<f64>::zeros((image_height, image_width));
        let mut time_derivative = Array2::<f64>::zeros((image_height, image_width));
        for x in 0..image_height {
            for y in 0..image_width {
                let (dx, dy) = space_derive(image1, x, y);
                x_derivative[[x, y]] = dx;
                y_derivative[[x, y]] = dy;
                time_derivative[[x, y]] = time_derive(image1, image2, x, y);
            }
        }
        for _ in 0..iter_max {
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

    fn pyramid(recursion_depth: u8, image1: ArrayView2<'_, f64>, image2: ArrayView2<'_, f64>, alpha_squared: f64, iter_max: u32) -> Fields {
        // Au cas où on irait trop loin avec les récursions au point où l'image devient trop petite pour le noyau gaussien
        if recursion_depth == 0 {
            return gauss_seidel(image1, image2, alpha_squared, iter_max);
        }

        let downscaled_images_1 = downscale_recursively(image1, recursion_depth);
        let downscaled_images_2 = downscale_recursively(image2, recursion_depth);

        let most_downscaled_idx = (recursion_depth - 1) as usize;
        
        let (mut u, mut v) = gauss_seidel(
            downscaled_images_1[most_downscaled_idx].view(), 
            downscaled_images_2[most_downscaled_idx].view(), 
            alpha_squared, 
            iter_max
        );

        for k in (0..most_downscaled_idx).rev() {
            let current_image_1 = downscaled_images_1[k].view();
            let current_image_2 = downscaled_images_2[k].view();
            
            let target_shape = (current_image_1.shape()[0], current_image_1.shape()[1]);
            
            let expanded_u = expand(u.view(), target_shape.0, target_shape.1);
            let expanded_v = expand(v.view(), target_shape.0, target_shape.1);

            let (new_u, new_v) = gauss_seidel_from_previous_uv(
                current_image_1, current_image_2, 
                expanded_u, expanded_v, 
                alpha_squared, iter_max
            );
            
            u = new_u;
            v = new_v;
        }

        let target_shape = (image1.shape()[0], image1.shape()[1]);
        let expanded_u = expand(u.view(), target_shape.0, target_shape.1);
        let expanded_v = expand(v.view(), target_shape.0, target_shape.1);

        gauss_seidel_from_previous_uv(image1, image2, expanded_u, expanded_v, alpha_squared, iter_max)
    }

    #[pyfunction]
    fn pyramidal_gauss_seidel<'py>(
            py: Python<'_>,
            video: PyReadonlyArray3<'_, f64>,
            alpha_squared: f64,
            max_iter: u32,
            recursion_depth: u8
        )
        -> Video {
        let video_array = video.as_array();

        let (frame_count, frame_height, frame_width) = (video_array.shape()[0], video_array.shape()[1], video_array.shape()[2]);

        let mut u_field = Array3::<f64>::zeros((frame_count, frame_height, frame_width));
        let mut v_field = Array3::<f64>::zeros((frame_count, frame_height, frame_width));

        for k in 0..frame_count-1 {
            let current_frame = video_array.index_axis(Axis(0), k);
            let next_frame = video_array.index_axis(Axis(0), k+1);

            let (u, v) = pyramid(recursion_depth, current_frame, next_frame, alpha_squared, max_iter);

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
    use ndarray::{Array2,};
    use numpy::ndarray::ArrayView2;

    
    pub fn space_derive(image: ArrayView2<'_, f64>, x: usize, y: usize) -> (f64, f64) {
        let image_height = image.shape()[0];
        let image_width = image.shape()[1];

        //Remplacement du bloc conditionnel par du clamping. Ça devrait permettre au compilateur d'appliquer des optimisations
        //Impossibles à mettre en place avex les blocs match (SIMD par exemple)
        let x_previous = x.saturating_sub(1);
        let x_next = (x + 1).min(image_height - 1);
        
        let y_previous = y.saturating_sub(1);
        let y_next = (y + 1).min(image_width - 1);

        //Le problème est que les bords de l'image sont un cas particulier à traiter.
        //Si on est à côté des bords, il ne faut plus diviser la différence.
        //Si on est aux bords, on applique une condition de Neumann pour que la dérivée soit nulle.
        let x_denominator = (x_next - x_previous) as f64;
        let y_denominator = (y_next - y_previous) as f64;

        let dx = if x_denominator > 0.0 { (image[[x_next, y]] - image[[x_previous, y]])/x_denominator } else { 0.0 };
        let dy = if y_denominator > 0.0 { (image[[x, y_next]] - image[[x, y_previous]])/y_denominator } else { 0.0 };

        (dx, dy)
    }

    pub fn time_derive(current_image: ArrayView2<'_, f64>, next_image: ArrayView2<'_, f64>, x: usize, y: usize) -> f64 {
        next_image[[x, y]] - current_image[[x, y]]
    }
    
    pub fn get_average(image: ArrayView2<'_, f64>, x: usize, y: usize) -> f64 {
        let image_height = image.shape()[0];
        let image_width = image.shape()[1];

        let get_clamped = |x_index: usize, y_index: usize| -> f64 {
            let x_clamped = x_index.clamp(0, image_height - 1);
            let y_clamped = y_index.clamp(0, image_width - 1);

            return image[[x_clamped, y_clamped]];
        };

        let closer_f64s = (
            get_clamped(x.saturating_sub(1), y) + get_clamped(x+1, y) + get_clamped(x, y.saturating_sub(1)) + get_clamped(x, y+1)
        )/6.0;
        let further_f64s = (
            get_clamped(x.saturating_sub(1), y.saturating_sub(1)) + get_clamped(x+1, y.saturating_sub(1)) + get_clamped(x+1, y+1) + get_clamped(x.saturating_sub(1), y+1)
        )/12.0;

        closer_f64s + further_f64s
    }

    pub fn get_cross(image: ArrayView2<'_, f64>, x_index: usize, y_index: usize) -> f64 {
        image[[x_index+1, y_index]] + image[[x_index-1, y_index]] + image[[x_index, y_index+1]] + image[[x_index, y_index-1]]
    }

    pub fn get_diagonal(image: ArrayView2<'_, f64>, x_index: usize, y_index: usize) -> f64 {
        image[[x_index+1, y_index+1]] + image[[x_index-1, y_index+1]] + image[[x_index+1, y_index-1]] + image[[x_index-1, y_index-1]]
    }

    pub fn expand_pixel(upscaled_image: &mut Array2<f64>, value: f64, x_center: usize, y_center: usize) {
        let h = upscaled_image.shape()[0];
        let w = upscaled_image.shape()[1];
        
        for dx in 0..=2 {
            for dy in 0..=2 {
                let x = x_center.saturating_add(dx).saturating_sub(1);
                let y = y_center.saturating_add(dy).saturating_sub(1);
                if x < h && y < w {
                    upscaled_image[[x, y]] = value;
                }
            }
        }
    }

    pub fn downscale(image: ArrayView2<'_, f64>) -> Array2<f64> {
        let (image_height, image_width) = (image.shape()[0], image.shape()[1]);
        let (new_width, new_height) = ((image_width-3)/3 + 1, (image_height-3)/3 + 1);

        let mut downscaled_image = Array2::<f64>::zeros((new_height, new_width));

        for x in 0..new_height {
            for y in 0..new_width {
                let (x_index, y_index) = (x*3+1, y*3+1);
                downscaled_image[[x, y]] = image[[x_index,y_index]]/4.0 + get_cross(image, x_index, y_index)/8.0 + get_diagonal(image, x_index, y_index)/16.0;
            }
        }

        downscaled_image
    }

    pub fn downscale_recursively(image: ArrayView2<'_, f64>, recursion_depth: u8) -> Vec<Array2<f64>> {
        let mut downscaled_images: Vec<Array2<f64>> = vec![downscale(image)];
        for k in 0..(recursion_depth as usize).saturating_sub(1) {
            downscaled_images.push(downscale(downscaled_images[k as usize].view()));
        }

        downscaled_images
    }

    pub fn expand(downscaled_image: ArrayView2<'_, f64>, target_height: usize, target_width: usize) -> Array2<f64> {
        let (downscaled_height, downscaled_width) = (downscaled_image.shape()[0], downscaled_image.shape()[1]);
        let mut expanded_image = Array2::<f64>::zeros((target_height, target_width));

        for x_index in 0..downscaled_height {
            for y_index in 0..downscaled_width {
                let (x_expanded_index, y_expanded_index) = (x_index*3 + 1, y_index*3 + 1);
                
                let scaled_value = downscaled_image[[x_index, y_index]] * 3.0;
                expand_pixel(&mut expanded_image, scaled_value, x_expanded_index, y_expanded_index);
            }
        }
        expanded_image
    }
}