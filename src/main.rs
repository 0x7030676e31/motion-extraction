#![feature(let_chains)]

use crossbeam_channel::{Receiver, Sender, bounded};
use jpeg_decoder::Decoder as JpegDecoder;
use minifb::{Window, WindowOptions};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use rscam::{Camera, Config, Frame};
use std::collections::VecDeque;
use std::io::Cursor;
use std::mem;
use std::thread::{self, JoinHandle};

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const FPS: u32 = 30;
// Number of frames to buffer: max offset (6) + 1
const BUF_LEN: usize = 13;

fn capture_thread(tx_capture: Sender<Frame>, rx_close: Receiver<()>) -> JoinHandle<()> {
    let config = Config {
        interval: (1, FPS),
        resolution: (WIDTH as u32, HEIGHT as u32),
        format: b"MJPG",
        ..Default::default()
    };

    let mut cam = Camera::new("/dev/video0").expect("Failed to open camera");
    cam.start(&config).expect("Failed to start camera");

    thread::spawn(move || {
        while rx_close.try_recv().is_err() {
            match cam.capture() {
                Ok(frame) => {
                    if tx_capture.send(frame).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    })
}

fn decode_thread(
    rx_capture: Receiver<Frame>,
    tx_decode: Sender<Vec<u32>>,
    rx_close: Receiver<()>,
) -> JoinHandle<()> {
    // Pre-allocate decode buffer to reuse
    let mut decode_buf = Vec::with_capacity(WIDTH * HEIGHT * 3);
    let mut rgb_u32 = Vec::with_capacity(WIDTH * HEIGHT);

    thread::spawn(move || {
        while rx_close.try_recv().is_err() {
            let frame = match rx_capture.recv() {
                Ok(f) => f,
                Err(_) => break,
            };

            decode_buf.clear();
            decode_buf.extend_from_slice(&frame);

            let mut decoder = JpegDecoder::new(Cursor::new(&decode_buf));
            let pixels = decoder.decode().expect("Failed to decode JPEG");

            rgb_u32.clear();
            for chunk in pixels.chunks_exact(3) {
                rgb_u32
                    .push(((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | (chunk[2] as u32));
            }

            // swap out and send the filled buffer without cloning
            let to_send = mem::replace(&mut rgb_u32, Vec::with_capacity(WIDTH * HEIGHT));
            if tx_decode.send(to_send).is_err() {
                break;
            }
        }
    })
}

fn main() {
    let (tx_cap, rx_cap) = bounded(2);
    let (tx_dec, rx_dec) = bounded(2);
    let (tx_close_cap, rx_close_cap) = bounded(1);
    let (tx_close_dec, rx_close_dec) = bounded(1);

    // Capture thread
    let cap_handle = capture_thread(tx_cap, rx_close_cap);
    // Decode thread
    let dec_handle = decode_thread(rx_cap, tx_dec, rx_close_dec);

    // Setup window
    let mut window = Window::new(
        "Motion Extraction (720p30)",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale: minifb::Scale::FitScreen,
            ..Default::default()
        },
    )
    .expect("Failed to create window");

    // History buffer initialization
    let mut history = VecDeque::from(vec![vec![0; WIDTH * HEIGHT]; BUF_LEN]);

    let mut fullscreen = false;
    let mut prev_dim = (WIDTH, HEIGHT);
    let mut prev_pos = (0, 0);

    // Pre-allocate diff buffer
    let mut diff_buf = vec![0u32; WIDTH * HEIGHT];

    // Main loop
    while window.is_open() {
        // Toggle fullscreen
        if window.is_key_down(minifb::Key::F11) {
            fullscreen = !fullscreen;

            if fullscreen {
                prev_dim = window.get_size();
                prev_pos = window.get_position();
            }

            drop(window);
            window = Window::new(
                "Motion Extraction (720p30)",
                if fullscreen { 1920 } else { prev_dim.0 },
                if fullscreen { 1080 } else { prev_dim.1 },
                WindowOptions {
                    resize: true,
                    scale: if fullscreen {
                        minifb::Scale::X1
                    } else {
                        minifb::Scale::FitScreen
                    },
                    borderless: fullscreen,
                    topmost: fullscreen,
                    ..Default::default()
                },
            )
            .expect("Failed to create window");

            if fullscreen {
                window.set_position(0, 0);
            } else {
                window.set_position(prev_pos.0, prev_pos.1);
            }
        }

        // Receive decoded frame
        let curr = match rx_dec.recv() {
            Ok(buf) => buf,
            Err(_) => break,
        };

        // Update history
        history.push_back(curr);
        if history.len() > BUF_LEN {
            history.pop_front();
        }

        let newest = &history[history.len() - 1];
        let frame_r = &history[history.len().saturating_sub(5)];
        let frame_g = &history[history.len().saturating_sub(9)];
        let frame_b = &history[history.len().saturating_sub(13)];

        // Compute diff into the pre-allocated diff_buf
        diff_buf.par_iter_mut().enumerate().for_each(|(i, d)| {
            let p = newest[i];
            let dr = (((p >> 16) & 0xFF) as i32 - ((frame_r[i] >> 16) & 0xFF) as i32).abs() as u32;
            let dg = (((p >> 8) & 0xFF) as i32 - ((frame_g[i] >> 8) & 0xFF) as i32).abs() as u32;
            let db = (((p) & 0xFF) as i32 - ((frame_b[i]) & 0xFF) as i32).abs() as u32;
            *d = (dr << 16) | (dg << 8) | db;
        });

        if window.update_with_buffer(&diff_buf, WIDTH, HEIGHT).is_err() {
            break;
        }
    }

    // Signal and join
    tx_close_dec.send(()).ok();
    tx_close_cap.send(()).ok();

    cap_handle.join().ok();
    dec_handle.join().ok();
}
