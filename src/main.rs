use std::collections::VecDeque;
use std::io::Cursor;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::{mem, thread};

use jpeg_decoder::Decoder as JpegDecoder;
use minifb::{Key, Scale, Window, WindowOptions};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use rscam::{Camera, Config, Frame};

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;

const FS_WIDTH: usize = 1920;
const FS_HEIGHT: usize = 1080;

const FPS: u32 = 30;

const CHANNEL_OFFSET: usize = 4;
const BUFFER_SIZE: usize = 2 + 2 * CHANNEL_OFFSET;

fn capture_thread(tx_capture: SyncSender<Frame>, rx_close: Receiver<()>) -> thread::JoinHandle<()> {
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
                Err(err) => {
                    eprintln!("Error capturing frame: {}", err);
                    break;
                }
            }
        }
    })
}

fn decode_thread(
    rx_capture: Receiver<Frame>,
    tx_decode: SyncSender<Vec<u32>>,
    rx_close: Receiver<()>,
) -> thread::JoinHandle<()> {
    let mut decode_buf = Vec::with_capacity(WIDTH * HEIGHT * 3);
    let mut decode_buf_u32 = Vec::with_capacity(WIDTH * HEIGHT);

    thread::spawn(move || {
        while rx_close.try_recv().is_err() {
            let frame = match rx_capture.recv() {
                Ok(frame) => frame,
                Err(_) => break,
            };

            decode_buf.clear();
            decode_buf.extend_from_slice(&frame);

            let mut decoder = JpegDecoder::new(Cursor::new(&decode_buf));
            let pixels = decoder.decode().expect("Failed to decode JPEG");

            decode_buf_u32.clear();
            for (idx, chunk) in pixels.chunks_exact(3).enumerate() {
                let r = chunk[0] as u32;
                let g = chunk[1] as u32;
                let b = chunk[2] as u32;

                let row = (idx + WIDTH - 1) / WIDTH + 1;
                if decode_buf_u32.len() < row * WIDTH {
                    decode_buf_u32.extend_from_slice(&vec![0u32; WIDTH]);
                }

                let index = row * WIDTH - (idx % WIDTH) - 1;
                decode_buf_u32[index] = (r << 16) | (g << 8) | b;
            }

            let pixels = mem::replace(&mut decode_buf_u32, Vec::with_capacity(WIDTH * HEIGHT));
            if tx_decode.send(pixels).is_err() {
                break;
            }
        }
    })
}

fn main() {
    let (tx_cap, rx_cap) = sync_channel(4);
    let (tx_dec, rx_dec) = sync_channel(4);

    let (tx_close_cap, rx_close_cap) = sync_channel(1);
    let (tx_close_dec, rx_close_dec) = sync_channel(1);

    let cap_handle = capture_thread(tx_cap, rx_close_cap);
    let dec_handle = decode_thread(rx_cap, tx_dec, rx_close_dec);

    let mut back_buffer = VecDeque::from(vec![vec![0; WIDTH * HEIGHT]; BUFFER_SIZE]);

    let mut fullscreen = false;
    let mut dimensions = (WIDTH, HEIGHT);
    let mut position = (0, 0);

    let mut diff_buf = vec![0u32; WIDTH * HEIGHT];

    let mut window = Window::new(
        "Motion Extraction (720p30)",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale: Scale::FitScreen,
            ..Default::default()
        },
    )
    .expect("Failed to create window");

    while window.is_open() {
        if window.is_key_released(Key::Escape) {
            break;
        }

        if window.is_key_released(Key::F11) {
            fullscreen = !fullscreen;

            if fullscreen {
                dimensions = window.get_size();
                position = window.get_position();
            }

            window = Window::new(
                "Motion Extraction (720p30)",
                if fullscreen { FS_WIDTH } else { dimensions.0 },
                if fullscreen { FS_HEIGHT } else { dimensions.1 },
                WindowOptions {
                    resize: !fullscreen,
                    borderless: fullscreen,
                    scale: Scale::FitScreen,
                    topmost: fullscreen,
                    ..Default::default()
                },
            )
            .expect("Failed to create window");

            if !fullscreen {
                window.set_position(position.0 - 4, position.1 - 46);
            } else {
                window.set_cursor_visibility(false);
            }
        }

        let curr = match rx_dec.recv() {
            Ok(buf) => buf,
            Err(_) => break,
        };

        back_buffer.push_back(curr);
        if back_buffer.len() > BUFFER_SIZE {
            back_buffer.pop_front();
        }

        let length = back_buffer.len();
        let newest = &back_buffer[length - 1];
        let frame_r = &back_buffer[length.saturating_sub(2)];
        let frame_g = &back_buffer[length.saturating_sub(2 + CHANNEL_OFFSET)];
        let frame_b = &back_buffer[length.saturating_sub(2 + CHANNEL_OFFSET + CHANNEL_OFFSET)];

        diff_buf.par_iter_mut().enumerate().for_each(|(i, pixel)| {
            let p = newest[i];
            let dr = ((p >> 16) & 0xFF).saturating_sub((frame_r[i] >> 16) & 0xFF);
            let dg = ((p >> 8) & 0xFF).saturating_sub((frame_g[i] >> 8) & 0xFF);
            let db = (p & 0xFF).saturating_sub(frame_b[i] & 0xFF);

            *pixel = (dr << 16) | (dg << 8) | db;
        });

        if let Err(err) = window.update_with_buffer(&diff_buf, WIDTH, HEIGHT) {
            eprintln!("Error updating window: {}", err);
            break;
        }
    }

    tx_close_cap
        .send(())
        .expect("Failed to send close signal to capture thread");
    tx_close_dec
        .send(())
        .expect("Failed to send close signal to decode thread");

    cap_handle.join().expect("Failed to join capture thread");
    dec_handle.join().expect("Failed to join decode thread");
}
