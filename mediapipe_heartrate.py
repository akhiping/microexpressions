import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, lfilter, detrend

# -------------------------------
# Helper Functions for Filtering
# -------------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def compute_heart_rate(signal, fs):
    """
    Detrends and bandpass-filters the signal, computes its FFT,
    and returns the dominant frequency (converted to BPM).
    """
    # Remove linear trend
    signal = detrend(signal)
    # Filter signal for heart rate frequencies (0.8 - 3.0 Hz, i.e., 48-180 BPM)
    filtered = bandpass_filter(signal, 0.8, 3.0, fs, order=4)
    
    fft_vals = np.fft.rfft(filtered)
    fft_freqs = np.fft.rfftfreq(len(filtered), 1.0/fs)
    
    # Focus on the frequency range of interest
    idx = np.where((fft_freqs >= 0.8) & (fft_freqs <= 3.0))
    if len(idx[0]) == 0:
        return None
    
    fft_mag = np.abs(fft_vals[idx])
    peak_idx = np.argmax(fft_mag)
    peak_freq = fft_freqs[idx][peak_idx]
    bpm = peak_freq * 60
    return bpm

# -------------------------------
# Main Function: Live rPPG using Mediapipe Face Mesh
# -------------------------------
def main():
    # Initialize Mediapipe Face Mesh for robust face landmark detection
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    # Open the see3cam using OpenCV (ensure it's connected and recognized)
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set resolution and frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 120)
    
    fs = 120                           # Sampling frequency (120 fps)
    duration = 5                       # Buffer duration in seconds
    buffer_size = fs * duration        # e.g., 5 sec * 120 fps = 600 frames
    roi_signal = []                    # Buffer for average green intensities
    latest_bpm = None                  # Latest computed heart rate

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        display_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            # Use the first detected face
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            # Convert landmark coordinates to pixel values
            landmarks = np.array([[int(pt.x * w), int(pt.y * h)] for pt in face_landmarks.landmark])
            # Determine the bounding box for the face
            x_min = np.min(landmarks[:,0])
            x_max = np.max(landmarks[:,0])
            y_min = np.min(landmarks[:,1])
            y_max = np.max(landmarks[:,1])
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Define ROI for forehead: use the upper third of the face bounding box
            roi_y_end = y_min + (y_max - y_min) // 3
            roi = frame[y_min:roi_y_end, x_min:x_max]
            # Resize ROI to a fixed size for consistency
            fixed_roi_size = (150, 100)
            roi_resized = cv2.resize(roi, fixed_roi_size)
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, roi_y_end), (255, 0, 0), 2)
            
            # Compute the average green channel intensity
            green_avg = np.mean(roi_resized[:, :, 1])
            roi_signal.append(green_avg)
            
            # Maintain fixed buffer size
            if len(roi_signal) > buffer_size:
                roi_signal.pop(0)
            
            # Process buffer when full
            if len(roi_signal) >= buffer_size:
                bpm = compute_heart_rate(np.array(roi_signal), fs)
                if bpm is not None:
                    latest_bpm = bpm
                    print("Estimated Heart Rate: {:.1f} BPM".format(bpm))
                else:
                    latest_bpm = None
                # Slide the window by removing the first half of the samples
                roi_signal = roi_signal[int(0.5 * buffer_size):]
        else:
            cv2.putText(display_frame, "No face detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Overlay the latest BPM on the video feed
        if latest_bpm is not None:
            cv2.putText(display_frame, f"Heart Rate: {latest_bpm:.1f} BPM", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Calculating...", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("Live rPPG", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
