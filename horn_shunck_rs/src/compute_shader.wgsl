@binding(0) @group(0) var<storage, read_write> u_buffer: array<f32>;
@binding(1) @group(0) var<storage, read_write> v_buffer: array<f32>;

struct Dimensions {
    height: u32,
    width: u32,
}

@binding(0) @group(2) var<uniform> dimensions: Dimensions;
@binding(3) @group(0) var<uniform> alpha: f32;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) position: vec3<u32>,
) {
    let x = position.x;
    let y = position.y;

    let index = dimensions.width * x + y;

    if x >= dimensions.width || y >= dimensions.height {return;}
    
    let x_derivative = x_derive(x, y);
    let y_derivative = y_derive(x, y);

    let average = u_laplace(x, y);

    let u_actual = average - x_derivative * (x_derivative * u_buffer[index] + y_derivative * v_buffer[index] + time_derivative)/(alpha + pow(x_derivative, 2) + pow(y_derivative, 2));
    let v_actual = average - y_derivative * (x_derivative * u_buffer[index] + y_derivative * v_buffer[index] + time_derivative)/(alpha + pow(x_derivative, 2) + pow(y_derivative, 2));
}

fn clamp_laplace(x: u32, y: u32) -> f32 {
    let safe_x = clamp(i32(x), 0i, i32(dimensions.width) - 1i);
    let safe_y = clamp(i32(y), 0i, i32(dimensions.height) - 1i);

    let index = safe_x * i32(dimensions.width) + safe_y;

    return u_buffer[index];
}

fn u_laplace(
    x: u32,
    y: u32
) -> f32 {
    let index = dimensions.width * x + y;

    if (x >= dimensions.width || y >= dimensions.height) { return 0; } //À vérifier, ne pas oublier !!!
    
    let left   = clamp_laplace(x - 1u, y);
    let right  = clamp_laplace(x + 1u, y);
    let top    = clamp_laplace(x, y - 1u);
    let bottom = clamp_laplace(x, y + 1u);
    
    let top_left     = clamp_laplace(x - 1u, y - 1u);
    let top_right    = clamp_laplace(x + 1u, y - 1u);
    let bottom_left  = clamp_laplace(x - 1u, y + 1u);
    let bottom_right = clamp_laplace(x + 1u, y + 1u);

    return (top+ right + bottom + left)/6f + (top_right + bottom_right + bottom_left + top_left)/12f;
}

fn derive_clamp_x(
    x: u32,
    y: u32
) -> f32 {
    let safe_x = clamp(x, 0u, dimensions.width-1u);
    let index= safe_x * dimensions.width + y;

    return u_buffer[index];
}

fn derive_clamp_y(
    x: u32,
    y: u32,
) -> f32 {
    let safe_y = clamp(y, 0u, dimensions.height-1u);
    let index= x * dimensions.width + safe_y;

    return u_buffer[index];
}

fn x_derive(
    x: u32,
    y: u32
) -> f32 {
    return (derive_clamp_x(x+1, y) - derive_clamp_x(x-1u, y))/2f;
}

fn y_derive(
    x: u32,
    y: u32
) -> f32 {
    return (derive_clamp_y(x,y+1u) - derive_clamp_y(x, y-1u))/2f;
}



