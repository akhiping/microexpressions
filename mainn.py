import os
import cv2
import numpy as np
import tensorflow as tf
from magnet import MagNet3Frames
from configparser import ConfigParser, ExtendedInterpolation
from subprocess import call

def load_config(config_path):
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

    return arch_config

def preprocess_frame(frame):
    frame = cv2.resize(frame, (960, 544))
    frame = frame.astype(np.float32) / 127.5 - 1.0
    return frame

def process_live_stream(out_dir, amplification_factor, model, velocity_mag=False):
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 544)
    cap.set(cv2.CAP_PROP_FPS, 120)

    if not cap.isOpened():
        print("Error: Could not open See3CAM.")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    frame_count = 0
    prev_frame = None
    print("Streaming from See3CAM. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Retrying...")
            continue

        current_frame = preprocess_frame(frame)

        if prev_frame is None:
            prev_frame = current_frame
            frame_count += 1
            continue

        dummy_amp = current_frame
        input_frames = np.concatenate([prev_frame, current_frame, dummy_amp], axis=-1)  # Shape: (544, 960, 9)
        input_frames = np.expand_dims(input_frames, axis=0)  # Shape: (1, 544, 960, 9)

        out_amp = model.sess.run(model.test_output, feed_dict={
            model.test_input: input_frames,
            model.test_amplification_factor: [amplification_factor]
        })

        magnified_frame = np.squeeze(out_amp)
        magnified_frame = (magnified_frame + 1.0) * 127.5
        magnified_frame = magnified_frame.astype(np.uint8)

        cv2.imshow('Magnified Live Stream', magnified_frame)

        out_path = os.path.join(out_dir, f'frame_{frame_count:06d}.png')
        cv2.imwrite(out_path, cv2.cvtColor(magnified_frame, cv2.COLOR_RGB2BGR))

        if velocity_mag:
            prev_frame = current_frame

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame_count

def main(arguments):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('video_name', type=str)
    parser.add_argument('amplification_factor', type=float)
    parser.add_argument('run_dynamic_mode', nargs='?', default='no', type=str)
    args = parser.parse_args(arguments)

    config_path = 'configs/o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds3.conf'
    arch_config = load_config(config_path)

    checkpoint_dir = os.path.join('data', 'training', args.experiment_name, 'checkpoint')
    out_dir = "/home/akhiping/Downloads/12dmodel/vid_output"

    sess = tf.Session()
    model = MagNet3Frames(sess, args.experiment_name, arch_config)

    image_width, image_height = 960, 544
    model.setup_for_inference(checkpoint_dir, image_width, image_height)

    velocity_mag = args.run_dynamic_mode.lower() == 'yes'
    frame_count = process_live_stream(out_dir, args.amplification_factor, model, velocity_mag)

    if frame_count > 0:
        call(['ffmpeg', '-y', '-f', 'image2', '-r', '30', '-i',
              os.path.join(out_dir, 'frame_%06d.png'), '-c:v', 'libx264',
              os.path.join(out_dir, args.video_name + '_magnified.mp4')])

    sess.close()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])