struct Params {
    alpha_squared: f32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex: vec2<f32>}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex: vec2<f32>}

const WIDTH: u32 = 1920u;
const HEIGHT: u32 = 1080u;

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(0)
var texture: texture_2d<f32>;

@group(0) @binding(1)
var camera_sampler: sampler;

@group(0) @binding(1)
var<storage, read> previous_luminance: array<u32>;

@group(0) @binding(2)
var<storage, read> current_luminance: array<u32>;

@group(0) @binding(3)
var<storage, read> flow: array<vec2<f32>>;

@group(0) @binding(4)
var<storage, read_write> next_flow: array<vec2<f32>>;

fn pixel_index(x: i32, y: i32) -> u32 {
    let cx = clamp(x, 0, i32(WIDTH) - 1);
    let cy = clamp(y, 0, i32(HEIGHT) - 1);
    return u32(cy) * WIDTH + u32(cx);
}

fn byte_at(word: u32, i: u32) -> f32 {
    return f32((word >> (8u * (i % 4u))) & 255u);
}

fn previous_at(x: i32, y: i32) -> f32 {
    let i = pixel_index(x, y);
    return byte_at(previous_luminance[i / 4u], i);
}

fn current_at(x: i32, y: i32) -> f32 {
    let i = pixel_index(x, y);
    return byte_at(current_luminance[i / 4u], i);
}

fn flow_at(x: i32, y: i32) -> vec2<f32> {
    return flow[pixel_index(x, y)];
}

@compute @workgroup_size(16, 16, 1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= WIDTH || id.y >= HEIGHT {
        return;
    }
    let x = i32(id.x);
    let y = i32(id.y);

    let p00 = previous_at(x, y);
    let p10 = previous_at(x + 1, y);
    let p01 = previous_at(x, y + 1);
    let p11 = previous_at(x + 1, y + 1);
    let c00 = current_at(x, y);
    let c10 = current_at(x + 1, y);
    let c01 = current_at(x, y + 1);
    let c11 = current_at(x + 1, y + 1);

    let ix = ((p10 - p00) + (p11 - p01) + (c10 - c00) + (c11 - c01)) / 4.0;
    let iy = ((p01 - p00) + (p11 - p10) + (c01 - c00) + (c11 - c10)) / 4.0;
    let it = ((c00 - p00) + (c10 - p10) + (c01 - p01) + (c11 - p11)) / 4.0;

    let direct = flow_at(x - 1, y) + flow_at(x + 1, y) + flow_at(x, y - 1) + flow_at(x, y + 1);
    let diagonal = flow_at(x - 1, y - 1) + flow_at(x + 1, y - 1) + flow_at(x - 1, y + 1) + flow_at(x + 1, y + 1);
    let mean = direct / 6.0 + diagonal / 12.0;

    let t = (ix * mean.x + iy * mean.y + it) / (params.alpha_squared + ix * ix + iy * iy);
    next_flow[pixel_index(x, y)] = vec2<f32>(mean.x - ix * t, mean.y - iy * t);
}

@vertex
fn vertex_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 1.0);
    out.tex = in.tex;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let x = min(u32(in.tex.x * f32(WIDTH)), WIDTH - 1u);
    let y = min(u32(in.tex.y * f32(HEIGHT)), HEIGHT - 1u);
    let f = flow[y * WIDTH + x];
    let intensity = clamp(sqrt(f.x * f.x + f.y * f.y), 0.0, 1.0)/10;
    return clamp(textureSample(texture, camera_sampler, in.tex) + vec4<f32>(0.0, f.y, f.x, intensity), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 1.0, 1.0));
}
