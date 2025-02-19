import cv2
import os
import numpy as np
import tensorflow as tf

def get_live_frames(output_dir="data/vids/live", max_frames=1000):
    """Capture live frames from See3CAM and save to a directory."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(2)  # Adjust to /dev/video0 if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  # Match deep_motion_mag input size
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 544)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Retrying...")
            continue

        # Save frame
        frame_path = os.path.join(output_dir, f'{frame_count:06d}.png')
        cv2.imwrite(frame_path, frame)
        frame_count += 1

        if frame_count >= max_frames or cv2.waitKey(1) & 0xFF == ord('q'):  # Stop after max_frames or 'q'
            break

    cap.release()
    return [f for f in sorted(os.listdir(output_dir)) if f.endswith('.png')]

# Modify the main function
def main(arguments):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('video_name', type=str)
    parser.add_argument('amplification_factor', type=float)
    parser.add_argument('run_dynamic_mode', nargs='?', default='no', type=str)
    args = parser.parse_args(arguments)

    # Use live stream instead of pre-recorded
    video_name = "live"
    out_dir = "/home/akhiping/Downloads/12dmodel/vid_output"  # Your output directory
    frame_list = get_live_frames()

    # Existing code for loading model and processing
    checkpoint_dir = os.path.join('data', 'training', args.experiment_name, 'checkpoint')
    model = load_model(checkpoint_dir)  # You’ll need to define or adapt this

    for frame_file in frame_list:
        frame_path = os.path.join('data/vids/live', frame_file)
        frame = cv2.imread(frame_path)

        # Process frame with deep_motion_mag (adapt from existing run function)
        magnified_frame = process_frame(frame, model, args.amplification_factor, args.run_dynamic_mode)

        # Display frame with heart rate
        heart_rate = estimate_heart_rate(frame)  # Define this function below
        if heart_rate:
            cv2.putText(magnified_frame, f'Heart Rate: {heart_rate:.1f} BPM', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Live Magnified Stream', magnified_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Save magnified frame
        out_path = os.path.join(out_dir, frame_file)
        cv2.imwrite(out_path, magnified_frame)

    cv2.destroyAllWindows()

# Heart rate estimation function (simplified rPPG)
def estimate_heart_rate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi = gray[y:y+h, x:x+w]
        signal = np.mean(roi)  # Average intensity

        # Simple bandpass filter (0.7-4 Hz for heart rate, assuming 30 fps)
        from scipy.signal import butter, filtfilt
        fs = 30
        lowcut, highcut = 0.7, 4.0  # 42-240 BPM
        nyquist = fs / 2
        low, high = lowcut / nyquist, highcut / nyquist
        b, a = butter(2, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, [signal])  # Buffer needed for accuracy

        # Estimate heart rate (simplified)
        from scipy.fft import fft
        f = fft(filtered_signal)
        freqs = np.fft.fftfreq(len(filtered_signal), 1/fs)
        peak_freq = freqs[np.argmax(np.abs(f))]
        heart_rate = peak_freq * 60  # Convert to BPM

        return heart_rate
    return None

# Placeholder for process_frame (adapt from magnet.py or existing code)
def process_frame(frame, model, amplification_factor, dynamic_mode):
    # Resize and normalize frame to match model input
    frame = cv2.resize(frame, (960, 544))  # Match your earlier video size
    frame = frame.astype(np.float32) / 255.0

    # Apply motion magnification (simplified)
    # This is where you’d call the existing deep_motion_mag logic
    magnified = model.predict(frame)  # Adapt based on your model
    magnified = (magnified * 255).astype(np.uint8)

    return magnified

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])