import pyrealsense2 as rs
import numpy as np
import cv2
import time
from scipy.signal import butter, lfilter

# -------------------------------
# 1. Helper Functions for Filtering
# -------------------------------
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# -------------------------------
# 2. Eulerian Video Magnification Function
# -------------------------------
def eulerian_magnification(roi_buffer, fps, lowcut, highcut, amplification, order=1):
    """
    Apply a simplified Eulerian Video Magnification on a fixed-size buffer of ROI frames.
    All frames in roi_buffer must have the same dimensions.
    """
    video = np.array(roi_buffer, dtype=np.float32)  # shape: (N, h, w, 3)
    b, a = butter_bandpass(lowcut, highcut, fps, order=order)
    # Filter along the time axis (axis=0)
    filtered_video = lfilter(b, a, video, axis=0)
    # Amplify and add back to original video
    amplified_video = video + amplification * filtered_video
    amplified_video = np.clip(amplified_video, 0, 255).astype(np.uint8)
    return amplified_video

# -------------------------------
# 3. Initialize RealSense Camera and Face Detector
# -------------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -------------------------------
# 4. Parameters and Buffers for Breathing Detection
# -------------------------------
fps = 90                             # Frame rate (should match camera settings)
duration_seconds = 5                 # Buffer duration (adjust as needed)
buffer_size = fps * duration_seconds # Number of frames in buffer

torso_buffer = []    # Buffer for torso ROI frames
intensity_buffer = []  # Buffer for extracted signal (proxy for breathing)

# EVM parameters for breathing: typical breathing frequency is 0.1–0.5 Hz (6–30 breaths per minute)
evm_lowcut = 0.1   # Lower cutoff in Hz
evm_highcut = 0.5  # Upper cutoff in Hz
amplification = 50 # Amplification factor (tune as needed)

# Define a fixed ROI size for the torso region (width, height)
fixed_roi_size = (150, 100)  # Adjust based on your camera and subject distance

print("Press 'q' to quit.")

# -------------------------------
# 5. Main Loop: Capture, ROI Extraction, EVM Processing, and Breathing Rate Estimation
# -------------------------------
while True:
    try:
        frames = pipeline.wait_for_frames(timeout_ms=10000)
    except RuntimeError as e:
        print("Timeout waiting for frames:", e)
        continue

    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    display_frame = frame.copy()
    
    # Detect face as an anchor point
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4)
    
    if len(faces) > 0:
        # Use the first detected face for defining the ROI
        (x, y, w, h) = faces[0]
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Define a torso ROI below the face:
        # For example, starting from the bottom of the face (y+h) and extending by ~1.5 * face height.
        torso_y_start = y + h
        torso_y_end = torso_y_start + int(1.5 * h)
        torso_roi = frame[torso_y_start:torso_y_end, x:x + w]
        
        # Check if the ROI is valid and resize to a fixed size
        if torso_roi.size != 0:
            torso_roi_fixed = cv2.resize(torso_roi, fixed_roi_size)
            cv2.rectangle(display_frame, (x, torso_y_start), (x + w, torso_y_end), (0, 255, 0), 2)
            
            # Append ROI to buffer and compute a proxy signal (average green channel intensity)
            torso_buffer.append(torso_roi_fixed)
            avg_intensity = np.mean(torso_roi_fixed[:, :, 1])
            intensity_buffer.append(avg_intensity)
            
            # Maintain a fixed-size buffer
            if len(torso_buffer) > buffer_size:
                torso_buffer.pop(0)
            if len(intensity_buffer) > buffer_size:
                intensity_buffer.pop(0)
            
            # Once the buffer is full, process for breathing detection
            if len(torso_buffer) == buffer_size:
                try:
                    # Apply EVM on the torso ROI buffer to magnify breathing motions
                    magnified_video = eulerian_magnification(torso_buffer, fps, evm_lowcut, evm_highcut, amplification, order=1)
                    
                    # Extract a time-series from the magnified video.
                    # Here, we compute the average green channel intensity from each magnified frame.
                    magnified_intensity = [np.mean(roi_frame[:, :, 1]) for roi_frame in magnified_video]
                    
                    # Remove DC component and apply further bandpass filtering
                    signal = np.array(magnified_intensity) - np.mean(magnified_intensity)
                    filtered_signal = bandpass_filter(signal, evm_lowcut, evm_highcut, fps, order=6)
                    
                    # Perform FFT to find the dominant frequency
                    fft_vals = np.fft.rfft(filtered_signal)
                    fft_freq = np.fft.rfftfreq(len(filtered_signal), d=1.0 / fps)
                    
                    # Select frequencies within the breathing range
                    valid_idx = np.where((fft_freq >= evm_lowcut) & (fft_freq <= evm_highcut))
                    if valid_idx[0].size > 0:
                        fft_mag = np.abs(fft_vals[valid_idx])
                        peak_idx = np.argmax(fft_mag)
                        peak_freq = fft_freq[valid_idx][peak_idx]
                        breathing_bpm = peak_freq * 60  # Convert Hz to breaths per minute
                        cv2.putText(display_frame, f"Breathing Rate: {breathing_bpm:.1f} BPM", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "No valid breathing frequency", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    cv2.putText(display_frame, "EVM (Torso) Applied", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    # Display the last magnified torso ROI for visual feedback
                    cv2.imshow("Magnified Torso ROI", magnified_video[-1])
                except Exception as e:
                    print("Error during EVM processing:", e)
                    torso_buffer.clear()
                    intensity_buffer.clear()
    else:
        cv2.putText(display_frame, "No face detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("RealSense Breathing EVM", display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# 6. Clean Up
# -------------------------------
pipeline.stop()
cv2.destroyAllWindows()
