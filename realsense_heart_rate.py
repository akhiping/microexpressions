import pyrealsense2 as rs
import numpy as np
import cv2
import time
from scipy.signal import butter, lfilter

# -------------------------------
# 1. Define Bandpass Filter Functions
# -------------------------------
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# -------------------------------
# 2. Initialize RealSense Pipeline
# -------------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# -------------------------------
# 3. Load Face Detection Model
# -------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -------------------------------
# 4. Signal Processing Parameters
# -------------------------------
fps = 90
duration_seconds = 5        # Change this to 5 for quicker results during debugging
buffer_size = fps * duration_seconds
intensity_buffer = []        # Buffer for average green intensities

# Heart rate frequency range in Hz (0.8 Hz = 48 BPM, 3.0 Hz = 180 BPM)
lowcut = 0.8
highcut = 3.0

print("Press 'q' to quit.")

# -------------------------------
# 5. Main Loop for Video Capture & Processing
# -------------------------------
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    display_frame = frame.copy()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract forehead ROI (upper 20% of the detected face)
        roi_y_end = y + int(0.2 * h)
        roi = frame[y:roi_y_end, x:x + w]
        cv2.rectangle(display_frame, (x, y), (x + w, roi_y_end), (0, 255, 0), 2)

        # Compute the average intensity from the green channel in the ROI
        avg_intensity = np.mean(roi[:, :, 1])
        intensity_buffer.append(avg_intensity)

        # Keep a fixed-size sliding window
        if len(intensity_buffer) > buffer_size:
            intensity_buffer.pop(0)

        # Provide on-screen feedback regarding data collection
        cv2.putText(display_frame, f"Data Frames: {len(intensity_buffer)}/{buffer_size}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Compute heart rate only if enough data is collected
        if len(intensity_buffer) == buffer_size:
            # Remove the DC component
            signal = np.array(intensity_buffer) - np.mean(intensity_buffer)
            filtered_signal = bandpass_filter(signal, lowcut, highcut, fps, order=6)

            # FFT analysis
            fft_vals = np.fft.rfft(filtered_signal)
            fft_freq = np.fft.rfftfreq(len(filtered_signal), d=1.0 / fps)

            # Select frequencies in the expected heart rate band
            valid_idx = np.where((fft_freq >= lowcut) & (fft_freq <= highcut))
            if valid_idx[0].size > 0:
                fft_mag = np.abs(fft_vals[valid_idx])
                peak_idx = np.argmax(fft_mag)
                peak_freq = fft_freq[valid_idx][peak_idx]
                bpm = peak_freq * 60  # Convert Hz to BPM

                # Debug: Print heart rate values to console
                print(f"Computed BPM: {bpm:.1f}")

                # Overlay the heart rate on the display frame
                cv2.putText(display_frame, f"Heart Rate: {bpm:.1f} BPM", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Debug info if FFT yields no valid frequency bins
                print("No valid frequency bins found in FFT")
    else:
        cv2.putText(display_frame, "No face detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('RealSense Video Magnification & rPPG', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
pipeline.stop()
cv2.destroyAllWindows()
