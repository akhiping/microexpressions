import cv2
import numpy as np
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
    y = lfilter(b, a, data)
    return y

# -------------------------------
# 2. Eulerian Video Magnification Function
# -------------------------------
def eulerian_magnification(roi_buffer, fps, lowcut, highcut, amplification, order=1):
    """
    Applies a simplified Eulerian Video Magnification on a buffer of ROI frames.
    Assumes that all frames in roi_buffer are of the same dimensions.
    """
    # Convert list to a 4D numpy array: (num_frames, height, width, channels)
    video = np.array(roi_buffer, dtype=np.float32)
    # Get filter coefficients for temporal filtering along the frame axis (axis=0)
    b, a = butter_bandpass(lowcut, highcut, fps, order=order)
    # Apply the temporal filter along time (axis=0)
    filtered_video = lfilter(b, a, video, axis=0)
    # Amplify the filtered signal and add it back to the original video
    amplified_video = video + amplification * filtered_video
    # Clip values and convert back to uint8
    amplified_video = np.clip(amplified_video, 0, 255).astype(np.uint8)
    return amplified_video

# -------------------------------
# 3. Initialize see3cam_24cug Using OpenCV
# -------------------------------
# Assuming your see3cam_24cug is available as the default camera (index 0)
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Set the camera resolution and FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 120)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -------------------------------
# 4. Parameters and Buffers
# -------------------------------
fps = 120                              # Set to 120 fps
duration_seconds = 5                   # Duration of the buffer (in seconds)
buffer_size = fps * duration_seconds   # Total frames to accumulate (120 * 5 = 600)

roi_buffer = []        # Buffer to store ROI frames
intensity_buffer = []  # Buffer to store average green intensities

# EVM parameters
evm_lowcut = 0.8      # Lower cutoff frequency in Hz (~48 BPM)
evm_highcut = 3.0     # Upper cutoff frequency in Hz (~180 BPM)
amplification = 50    # Amplification factor for EVM

# Define a fixed ROI size (width, height) for consistency
fixed_roi_size = (100, 50)  # Adjust as needed

print("Press 'q' to quit.")

# -------------------------------
# 5. Main Loop: Capture, EVM Processing, and rPPG Extraction
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    display_frame = frame.copy()
    
    # Face detection using Haar cascades
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4)
    
    if len(faces) > 0:
        # Use the first detected face for simplicity
        (x, y, w, h) = faces[0]
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Define the forehead region (upper 20% of the face)
        roi_y_end = y + int(0.2 * h)
        roi = frame[y:roi_y_end, x:x + w]
        
        # Resize the ROI to a fixed size for consistency
        roi_fixed = cv2.resize(roi, fixed_roi_size)
        cv2.rectangle(display_frame, (x, y), (x + w, roi_y_end), (255, 0, 0), 2)
        
        # Append the fixed ROI to the buffer and compute average green intensity
        roi_buffer.append(roi_fixed)
        avg_intensity = np.mean(roi_fixed[:, :, 1])
        intensity_buffer.append(avg_intensity)
        
        # Maintain a fixed-size buffer
        if len(roi_buffer) > buffer_size:
            roi_buffer.pop(0)
        if len(intensity_buffer) > buffer_size:
            intensity_buffer.pop(0)
        
        # Once the buffer is full, process EVM and extract heart rate
        if len(roi_buffer) == buffer_size:
            try:
                # Apply Eulerian Video Magnification on the ROI buffer
                magnified_video = eulerian_magnification(roi_buffer, fps, evm_lowcut, evm_highcut, amplification, order=1)
                
                # Compute the average green channel intensity for each magnified frame
                magnified_intensity = [np.mean(frame_roi[:, :, 1]) for frame_roi in magnified_video]
                
                # Remove the DC component and further filter the signal
                signal = np.array(magnified_intensity) - np.mean(magnified_intensity)
                filtered_signal = bandpass_filter(signal, evm_lowcut, evm_highcut, fps, order=6)
                
                # Perform FFT to extract the dominant frequency (heart rate)
                fft_vals = np.fft.rfft(filtered_signal)
                fft_freq = np.fft.rfftfreq(len(filtered_signal), d=1.0 / fps)
                
                # Select frequencies within the expected heart rate range
                valid_idx = np.where((fft_freq >= evm_lowcut) & (fft_freq <= evm_highcut))
                if valid_idx[0].size > 0:
                    fft_mag = np.abs(fft_vals[valid_idx])
                    peak_idx = np.argmax(fft_mag)
                    peak_freq = fft_freq[valid_idx][peak_idx]
                    bpm = peak_freq * 60  # Convert Hz to BPM
                    cv2.putText(display_frame, f"Heart Rate: {bpm:.1f} BPM", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "No valid frequency", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.putText(display_frame, "EVM Applied", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                # Display the last magnified ROI frame for visual feedback
                cv2.imshow("Magnified ROI", magnified_video[-1])
            except Exception as e:
                print("Error during EVM processing:", e)
                # Optionally, clear the buffers if needed
                roi_buffer.clear()
                intensity_buffer.clear()
    
    else:
        cv2.putText(display_frame, "No face detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("See3Cam EVM & rPPG", display_frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# 6. Clean Up
# -------------------------------
cap.release()
cv2.destroyAllWindows()
