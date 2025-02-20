import os
import cv2
import numpy as np
import tensorflow as tf
from magnet import MagNet3Frames
from configparser import ConfigParser, ExtendedInterpolation
from subprocess import call

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
    frame = cv2.resize(frame, (640, 360))  # Reduced resolution
    frame = frame.astype(np.float32) / 127.5 - 1.0
    print("Frame preprocessed")
    return frame

def process_live_stream(out_dir, amplification_factor, model, velocity_mag=False):
    print("Initializing camera capture")
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        print("Error: Could not open See3CAM.")
        return 0

    print(f"Setting up output directory: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    frame_count = 0
    prev_frame = None
    print("Starting live stream processing. Press 'q' to quit.")

    while True:
        print(f"Capturing frame {frame_count + 1}")
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Retrying...")
            continue

        current_frame = preprocess_frame(frame)

        if prev_frame is None:
            print("Setting initial previous frame")
            prev_frame = current_frame
            frame_count += 1
            continue

        print("Preparing input frames")
        dummy_amp = current_frame
        input_frames = np.concatenate([prev_frame, current_frame, dummy_amp], axis=-1)
        input_frames = np.expand_dims(input_frames, axis=0)

        print(f"Running inference on frame {frame_count + 1}")
        out_amp = model.sess.run(model.test_output, feed_dict={
            model.test_input: input_frames,
            model.test_amplification_factor: [amplification_factor]
        })

        print("Post-processing magnified frame")
        magnified_frame = np.squeeze(out_amp)
        magnified_frame = (magnified_frame + 1.0) * 127.5
        magnified_frame = magnified_frame.astype(np.uint8)

        print("Displaying magnified frame")
        cv2.imshow('Magnified Live Stream', magnified_frame)

        # Optional: Comment out to skip saving for max speed
        # print(f"Saving frame {frame_count + 1} to disk")
        # out_path = os.path.join(out_dir, f'frame_{frame_count:06d}.png')
        # cv2.imwrite(out_path, cv2.cvtColor(magnified_frame, cv2.COLOR_RGB2BGR))

        if velocity_mag:
            print("Updating previous frame for dynamic mode")
            prev_frame = current_frame

        frame_count += 1

        print(f"Frame {frame_count} completed")
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

    image_width, image_height = 640, 360
    print(f"Setting up model for inference with width={image_width}, height={image_height}")
    model.setup_for_inference(checkpoint_dir, image_width, image_height)

    print("Starting live stream processing")
    velocity_mag = args.run_dynamic_mode.lower() == 'yes'
    frame_count = process_live_stream(out_dir, args.amplification_factor, model, velocity_mag)

    if frame_count > 0:
        print(f"Combining {frame_count} frames into video")
        call(['ffmpeg', '-y', '-f', 'image2', '-r', '15', '-i',
              os.path.join(out_dir, 'frame_%06d.png'), '-c:v', 'libx264',
              os.path.join(out_dir, args.video_name + '_magnified.mp4')])
        print("Video creation completed")

    print("Closing TensorFlow session")
    sess.close()
    print("Script execution finished")

if __name__ == "__main__":
    import sys
    print("Starting script execution")
    main(sys.argv[1:])