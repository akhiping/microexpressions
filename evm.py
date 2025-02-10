import pyrealsense2 as rs
import numpy as np
import cv2
import time
from scipy.signal import butter, lfilter

# -------------------------------
# 1. Helper Functions for Filtering
# -------------------------------
def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Create a Butterworth bandpass filter.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth bandpass filter to the data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# -------------------------------
# 2. Eulerian Video Magnification Function
# -------------------------------
def eulerian_magnification(roi_buffer, fps, lowcut, highcut, amplification, order=1):
    """
    Applies a simplified Eulerian Video Magnification on a buffer of ROI frames.
    
    Parameters:
      roi_buffer: list of ROI frames (numpy arrays) with shape (h, w, 3)
      fps: frame rate of the video
      lowcut, highcut: frequency band of interest (e.g., heart rate frequencies)
      amplification: factor by which to amplify the temporal variations
      order: order of the Butterworth filter
      
    Returns:
      amplified_video: numpy array of magnified ROI frames (dtype=uint8)
    """
    # Convert list to a 4D numpy array: (num_frames, h, w, 3)
    video = np.array(roi_buffer, dtype=np.float32)
    # video shape: (N, h, w, 3)
    
    # Get filter coefficients for temporal filtering along the frame axis (axis=0)
    b, a = butter_bandpass(lowcut, highcut, fps, order=order)
    
    # Apply the temporal filter on the video data along the time axis (axis=0)
    filtered_video = lfilter(b, a, video, axis=0)
    
    # Amplify the filtered signal and add it back to the original video
    amplified_video = video + amplification * filtered_video
    
    # Clip values to valid range [0, 255] and convert to uint8
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
# 4. Parameters and Buffers
# -------------------------------
fps = 30                      # Frame rate (must match camera settings)
duration_seconds = 10         # Length of buffer for processing (adjust as needed)
buffer_size = fps * duration_seconds

# Buffers for ROI frames and the corresponding green channel intensities
roi_buffer = []               
intensity_buffer = []         

# EVM parameters (using the heart rate frequency band as an example)
evm_lowcut = 0.8              # Lower bound in Hz (~48 BPM)
evm_highcut = 3.0             # Upper bound in Hz (~180 BPM)
amplification = 50            # Amplification factor for subtle color changes

print("Press 'q' to quit.")

# -------------------------------
# 5. Main Loop: Capture, EVM Processing, and rPPG Extraction
# -------------------------------
while True:
    # Capture frame from RealSense camera
    #frames = pipeline.wait_for_frames()
    try:
        frames = pipeline.wait_for_frames(timeout_ms=10000)  # Increase timeout to 10 seconds
    except RuntimeError as e:
        print("Error: ", e)
        continue 
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    display_frame = frame.copy()
    
    # Face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4)
    
    if len(faces) > 0:
        # Use the first detected face for simplicity
        (x, y, w, h) = faces[0]
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Define the forehead region (upper ~20% of the face)
        roi_y_end = y + int(0.2 * h)
        roi = frame[y:roi_y_end, x:x + w]
        cv2.rectangle(display_frame, (x, y), (x + w, roi_y_end), (0, 255, 0), 2)
        
        # Append ROI to buffer and compute average green intensity
        roi_buffer.append(roi)
        avg_intensity = np.mean(roi[:, :, 1])
        intensity_buffer.append(avg_intensity)
        
        # Maintain fixed buffer size (sliding window)
        if len(roi_buffer) > buffer_size:
            roi_buffer.pop(0)
        if len(intensity_buffer) > buffer_size:
            intensity_buffer.pop(0)
        
        # Once sufficient frames are collected, apply EVM and compute heart rate
        if len(roi_buffer) == buffer_size:
            # Apply Eulerian Video Magnification to the ROI buffer
            magnified_video = eulerian_magnification(roi_buffer, fps, evm_lowcut, evm_highcut, amplification, order=1)
            
            # Compute the average green channel intensity for each magnified ROI frame
            magnified_intensity = [np.mean(roi_frame[:, :, 1]) for roi_frame in magnified_video]
            
            # Remove the DC component
            signal = np.array(magnified_intensity) - np.mean(magnified_intensity)
            
            # Apply an additional bandpass filter (optional, for further noise reduction)
            filtered_signal = bandpass_filter(signal, evm_lowcut, evm_highcut, fps, order=6)
            
            # Perform FFT to extract the dominant frequency
            fft_vals = np.fft.rfft(filtered_signal)
            fft_freq = np.fft.rfftfreq(len(filtered_signal), d=1.0 / fps)
            
            # Consider only frequencies in the valid heart rate range
            valid_idx = np.where((fft_freq >= evm_lowcut) & (fft_freq <= evm_highcut))
            if valid_idx[0].size > 0:
                fft_mag = np.abs(fft_vals[valid_idx])
                peak_idx = np.argmax(fft_mag)
                peak_freq = fft_freq[valid_idx][peak_idx]
                bpm = peak_freq * 60  # Convert Hz to BPM
                
                # Overlay the computed heart rate on the display frame
                cv2.putText(display_frame, f"Heart Rate: {bpm:.1f} BPM", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No valid frequency", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Provide feedback that EVM was applied (and display the last magnified ROI)
            cv2.putText(display_frame, "EVM Applied", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("Magnified ROI", magnified_video[-1])
    else:
        cv2.putText(display_frame, "No face detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("RealSense EVM & rPPG", display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# 6. Clean Up
# -------------------------------
pipeline.stop()
cv2.destroyAllWindows()
