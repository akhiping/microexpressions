import cv2
import numpy as np
import time
import sys
from scipy.signal import butter, lfilter, detrend

# Add the PyEVM directory to the Python path (if not installed via pip)
sys.path.append('/home/akhiping/Downloads/PyEVM-master/pyevm')

try:
    from evm import amplify_live_frames  # Adjust if the API is different
except ImportError as e:
    print(f"Failed to import PyEVM: {e}")
    exit()

# --- PyEVM Amplification Parameters ---
alpha = 50         # Amplification factor
lowcut = 0.4       # Low cutoff frequency (Hz)
highcut = 3.0      # High cutoff frequency (Hz)
pyramid_levels = 3 # Number of spatial pyramid levels

# --- Live Capture Parameters ---
fs = 120                         # Frames per second
duration = 5                     # Buffer duration in seconds
buffer_size = int(fs * duration) # e.g., 5 sec * 120 fps = 600 frames

# Function to compute heart rate from a signal using FFT
def compute_heart_rate(signal, fs):
    # Remove linear trend
    signal = detrend(signal)
    # Apply a Butterworth bandpass filter for frequencies corresponding to 48-180 BPM (0.8-3.0 Hz)
    b, a = butter(4, [0.8/(0.5*fs), 3.0/(0.5*fs)], btype='band')
    filtered = lfilter(b, a, signal)
    # Compute FFT
    fft_vals = np.fft.rfft(filtered)
    fft_freqs = np.fft.rfftfreq(len(filtered), d=1.0/fs)
    # Consider frequencies within the heart rate range
    idx = np.where((fft_freqs >= 0.8) & (fft_freqs <= 3.0))
    if len(idx[0]) == 0:
        return None
    fft_mag = np.abs(fft_vals[idx])
    peak_idx = np.argmax(fft_mag)
    peak_freq = fft_freqs[idx][peak_idx]
    bpm = peak_freq * 60
    return bpm

def main():
    # Open the camera (using device index 2 as in your original code; adjust if needed)
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set the desired resolution and fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, fs)

    # Buffers for live processing
    frame_buffer = []   # To store frames for magnification (in RGB)
    roi_signal = []     # To store average green intensity for heart rate computation
    latest_bpm = None   # Last computed heart rate

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert frame from BGR to RGB for PyEVM processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Append frame to the amplification buffer
        frame_buffer.append(frame_rgb)
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)

        # For rPPG, compute average green intensity (from the ROI)
        # For demonstration, we use the entire frame's green channel. 
        # In practice, you should use a properly defined ROI (e.g., forehead region).
        green_avg = np.mean(frame_rgb[:, :, 1])
        roi_signal.append(green_avg)
        if len(roi_signal) > buffer_size:
            roi_signal.pop(0)

        # When the buffer is full, process it
        if len(frame_buffer) >= buffer_size and len(roi_signal) >= buffer_size:
            try:
                # Normalize frames to [0,1] and amplify them using PyEVM
                video_buffer = np.array(frame_buffer, dtype=np.float32) / 255.0
                amplified_video = amplify_live_frames(video_buffer, fs, lowcut, highcut, 
                                                       levels=pyramid_levels, amplification=alpha)
                # Use the last amplified frame for display
                amplified_frame = amplified_video[-1]
                # Convert amplified frame from RGB back to BGR for display
                amplified_frame_bgr = cv2.cvtColor((amplified_frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print("Error during amplification:", e)
                amplified_frame_bgr = frame  # Fallback: use the original frame

            # Compute heart rate from roi_signal
            bpm = compute_heart_rate(np.array(roi_signal), fs)
            if bpm is not None:
                latest_bpm = bpm
                print("Estimated Heart Rate: {:.1f} BPM".format(bpm))
            else:
                latest_bpm = None

            # Slide the window by removing the first half of the buffers
            frame_buffer = frame_buffer[int(0.5 * buffer_size):]
            roi_signal = roi_signal[int(0.5 * buffer_size):]

        # For display, use the original frame converted back to BGR if it was processed in RGB
        # (Since we haven't modified the live frame, we use the original BGR frame)
        display_frame = frame.copy()

        # Overlay heart rate on the display
        if latest_bpm is not None:
            cv2.putText(display_frame, f"Heart Rate: {latest_bpm:.1f} BPM", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Calculating HR...", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Optionally, show both the live feed and the amplified output
        cv2.imshow("Live Feed", display_frame)
        cv2.imshow("Amplified Output", amplified_frame_bgr if 'amplified_frame_bgr' in locals() else frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
