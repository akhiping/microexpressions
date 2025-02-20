import os
import cv2
import numpy as np
import tensorflow as tf
from magnet import MagNet3Frames
from configparser import ConfigParser, ExtendedInterpolation
from subprocess import call
import time
from scipy import signal
from collections import deque

def load_config(config_path):
    print(f"Starting to load config from {config_path}")
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(config_path)

    print("Config sections:", config.sections())
    for section in config.sections():
        print(f"Section: {section}, Items: {dict(config[section])}")

    arch_config = {}
    if 'architecture' in config:
        arch_config['n_channels'] = config.getint('architecture', 'n_channels', fallback=3)
        ynet_section = 'architecture:ynet_3frames'
        if ynet_section in config:
            print(f"Loading ynet_3frames section from config")
            ynet_config = {
                'enc_dims': 32,
                'texture_dims': 32,
                'shape_dims': 32,
                'use_shape_conv': False
            }
            for key in config[ynet_section]:
                value = config[ynet_section][key]
                try:
                    ynet_config[key] = int(value)
                except ValueError:
                    if value.lower() in ('true', 'false'):
                        ynet_config[key] = value.lower() == 'true'
                    else:
                        ynet_config[key] = value
            arch_config['ynet_3frames'] = ynet_config
        else:
            print(f"Warning: '{ynet_section}' not found in config file!")

    print("Loaded arch_config:", arch_config)

    if 'ynet_3frames' not in arch_config or not arch_config['ynet_3frames']:
        print("Error: 'ynet_3frames' section is empty or missing. Using defaults.")
        arch_config['ynet_3frames'] = {
            'enc_dims': 32,
            'num_enc_resblk': 3,
            'num_man_resblk': 1,
            'num_man_conv': 1,
            'num_man_aft_conv': 1,
            'num_dec_resblk': 9,
            'texture_dims': 32,
            'texture_downsample': True,
            'use_texture_conv': True,
            'shape_dims': 32,
            'num_texture_resblk': 2,
            'num_shape_resblk': 2,
            'use_shape_conv': False
        }

    print("Finished loading config")
    return arch_config

def preprocess_frame(frame):
    print("Preprocessing frame")
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
    frame = frame.astype(np.float32) / 127.5 - 1.0
    print("Frame preprocessed")
    return frame

def compute_heart_rate(signal_buffer, fps, window_size=180):
    """Estimate heart rate from green channel signal."""
    print(f"Computing heart rate with buffer size: {len(signal_buffer)}")
    if len(signal_buffer) < window_size:
        print(f"Not enough data yet: {len(signal_buffer)}/{window_size}")
        return None

    signal_data = np.array(signal_buffer)
    print(f"Signal mean: {np.mean(signal_data):.2f}, std: {np.std(signal_data):.2f}")

    # Bandpass filter (0.7-3 Hz, 42-180 BPM)
    nyquist = fps / 2
    b, a = signal.butter(2, [0.7 / nyquist, 3.0 / nyquist], btype='band')
    filtered = signal.filtfilt(b, a, signal_data)
    print(f"Filtered signal mean: {np.mean(filtered):.2f}, std: {np.std(filtered):.2f}")

    # FFT
    fft_data = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), 1.0 / fps)
    peak_idx = np.argmax(fft_data)
    peak_freq = freqs[peak_idx]
    bpm = peak_freq * 60
    print(f"Peak frequency: {peak_freq:.2f} Hz, BPM: {bpm:.1f}")

    return int(bpm) if 42 <= bpm <= 180 else None

def process_live_stream(out_dir, amplification_factor, model, velocity_mag=False):
    print("Initializing camera capture")
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("Error: Could not open See3CAM.")
        return 0

    print(f"Setting up output directory: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    frame_count = 0
    prev_frame = None
    signal_buffer = deque(maxlen=180)  # 3 seconds at 60 FPS
    fps = 60
    print("Starting live stream processing. Press 'q' to quit.")

    while True:
        print(f"Capturing frame {frame_count + 1}")
        start_capture = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Retrying...")
            continue
        capture_time = time.time() - start_capture
        print(f"Capture took {capture_time:.3f} seconds")

        # Extract green channel from ROI (center 40x40)
        roi = frame[40:80, 60:100, 1]  # Adjusted for 160x120
        green_mean = np.mean(roi)
        signal_buffer.append(green_mean)
        print(f"Green mean from ROI: {green_mean:.2f}")

        current_frame = preprocess_frame(frame)

        if prev_frame is None:
            print("Setting initial previous frame")
            prev_frame = current_frame
            frame_count += 1
            continue

        print("Preparing input frames")
        start_prep = time.time()
        dummy_amp = current_frame
        input_frames = np.concatenate([prev_frame, current_frame, dummy_amp], axis=-1)
        input_frames = np.expand_dims(input_frames, axis=0)
        prep_time = time.time() - start_prep
        print(f"Input prep took {prep_time:.3f} seconds")

        print(f"Running inference on frame {frame_count + 1}")
        start_inference = time.time()
        out_amp = model.sess.run(model.test_output, feed_dict={
            model.test_input: input_frames,
            model.test_amplification_factor: [amplification_factor]
        })
        inference_time = time.time() - start_inference
        print(f"Inference took {inference_time:.3f} seconds")

        print("Post-processing magnified frame")
        start_post = time.time()
        magnified_frame = np.squeeze(out_amp)
        magnified_frame = (magnified_frame + 1.0) * 127.5
        magnified_frame = magnified_frame.astype(np.uint8)
        post_time = time.time() - start_post
        print(f"Post-processing took {post_time:.3f} seconds")

        # Compute and display heart rate
        bpm = compute_heart_rate(signal_buffer, fps)
        text = f"Heart Rate: {bpm if bpm else 'Calculating...'} BPM"
        print(f"Overlaying text: {text}")
        cv2.putText(magnified_frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        print("Displaying live magnified frame with heart rate")
        start_display = time.time()
        cv2.imshow('Magnified Live Stream', magnified_frame)
        display_time = time.time() - start_display
        print(f"Display took {display_time:.3f} seconds")

        if velocity_mag:
            print("Updating previous frame for dynamic mode")
            prev_frame = current_frame

        frame_count += 1

        print(f"Frame {frame_count} completed. Total frame time: {(time.time() - start_capture):.3f} seconds")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User requested exit")
            break

    print("Releasing camera and closing windows")
    cap.release()
    cv2.destroyAllWindows()
    print(f"Live stream processing finished. Processed {frame_count} frames.")
    return frame_count

def main(arguments):
    import argparse
    print("Parsing command-line arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('video_name', type=str)
    parser.add_argument('amplification_factor', type=float)
    parser.add_argument('run_dynamic_mode', nargs='?', default='no', type=str)
    args = parser.parse_args(arguments)

    print("Loading configuration")
    config_path = 'configs/o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds3.conf'
    arch_config = load_config(config_path)

    print("Setting up paths")
    checkpoint_dir = os.path.join('data', 'training', args.experiment_name, 'checkpoint')
    out_dir = "/home/akhiping/Downloads/12dmodel/vid_output"

    print("Initializing TensorFlow session")
    sess = tf.Session()
    print("Building MagNet3Frames model")
    model = MagNet3Frames(sess, args.experiment_name, arch_config)

    image_width, image_height = 160, 120
    print(f"Setting up model for inference with width={image_width}, height={image_height}")
    model.setup_for_inference(checkpoint_dir, image_width, image_height)

    print("Starting live stream processing")
    velocity_mag = args.run_dynamic_mode.lower() == 'yes'
    frame_count = process_live_stream(out_dir, args.amplification_factor, model, velocity_mag)

    print("Closing TensorFlow session")
    sess.close()
    print("Script execution finished")

if __name__ == "__main__":
    import sys
    print("Starting script execution")
    main(sys.argv[1:])