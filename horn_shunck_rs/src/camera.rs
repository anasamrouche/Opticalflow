use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender},
    },
    thread,
};

use nokhwa::{
    Camera, NokhwaError,
    pixel_format::{LumaFormat, RgbAFormat},
    utils::*,
};

pub struct RgbaFrame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

#[derive(Clone)]
pub struct LuminanceFrame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

pub enum CaptureRequest {
    Frame(RgbaFrame, LuminanceFrame),
    Error(String),
}

pub(crate) fn open_camera() -> Result<Camera, NokhwaError> {
    let mut camera = Camera::new(
        CameraIndex::Index(0),
        RequestedFormat::new::<RgbAFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
    )?;

    match camera.open_stream() {
        Ok(_) => {}
        Err(e) => eprintln!("Impossible to open camera stream {}", e),
    };
    Ok(camera)
}

pub struct CaptureThread {
    pub receiver: Receiver<CaptureRequest>,
    stop: Arc<AtomicBool>,
}

impl CaptureThread {
    pub fn spawn() -> Self {
        let (sender, receiver) = mpsc::sync_channel(2);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = stop.clone();

        thread::spawn(move || capture_loop(sender, stop_flag));

        Self { receiver, stop }
    }
}

fn capture_loop(sender: SyncSender<CaptureRequest>, stop: Arc<AtomicBool>) {
    let mut camera = match open_camera() {
        Ok(camera) => camera,
        Err(err) => {
            let _ = sender.send(CaptureRequest::Error(format!("{err}")));
            return;
        }
    };

    while !stop.load(Ordering::Relaxed) {
        let buffer = match camera.frame() {
            Ok(buffer) => buffer,
            Err(err) => {
                let _ = sender.send(CaptureRequest::Error(format!("{}", err)));
                return;
            }
        };
        let colored_image = match buffer.decode_image::<RgbAFormat>() {
            Ok(image) => image,
            Err(err) => {
                let _ = sender.send(CaptureRequest::Error(format!("{}", err)));
                return;
            }
        };
        let luminance_image = match buffer.decode_image::<LumaFormat>() {
            Ok(image) => image,
            Err(err) => {
                let _ = sender.send(CaptureRequest::Error(format!("{}", err)));
                return;
            }
        };

        let frame_color = RgbaFrame {
            width: colored_image.width(),
            height: colored_image.height(),
            data: colored_image.into_raw(),
        };
        let frame_luminance = LuminanceFrame {
            width: luminance_image.width(),
            height: luminance_image.height(),
            data: luminance_image.into_raw(),
        };
        match sender.send(CaptureRequest::Frame(frame_color, frame_luminance)) {
            Ok(_) => {}
            Err(err) => eprintln!("Error while sending frame: {err}"),
        }
    }
}

impl Drop for CaptureThread {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}
