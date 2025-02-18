import cv2
import numpy as np
from scipy.signal import butter, lfilter, detrend
import time

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Create a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a Butterworth bandpass filter."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def compute_heart_rate(signal, fs):
    """
    Given a time-series signal and sampling frequency (fs),
    remove trend, filter it, and compute the dominant frequency via FFT.
    """
    # Remove linear trend
    signal = detrend(signal)
    # Apply bandpass filtering for typical heart rate frequencies (0.8 - 3.0 Hz)
    filtered = bandpass_filter(signal, 0.8, 3.0, fs, order=4)
    
    # Compute FFT
    fft_vals = np.fft.rfft(filtered)
    fft_freqs = np.fft.rfftfreq(len(filtered), 1.0 / fs)
    
    # Only consider frequencies in the heart rate range
    idx = np.where((fft_freqs >= 0.8) & (fft_freqs <= 3.0))
    if len(idx[0]) == 0:
        return None
    
    fft_mag = np.abs(fft_vals[idx])
    peak_idx = np.argmax(fft_mag)
    peak_freq = fft_freqs[idx][peak_idx]
    bpm = peak_freq * 60
    return bpm

def main():
    # Initialize the see3cam_24cug using OpenCV
    cap = cv2.VideoCapture(2)  # Adjust the index if needed
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera properties for resolution and 120 fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 120)

    # Load face detection model (Haar cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Define parameters
    fs = 120                            # Sampling frequency in Hz (120 fps)
    duration = 10                       # Duration (in seconds) for one processing window
    buffer_size = fs * duration         # Number of frames to accumulate (e.g., 10 sec * 120 fps = 1200 frames)
    
    roi_signal = []                     # Buffer to store average green channel intensities

    # Fixed ROI size for consistency (width, height)
    fixed_roi_size = (100, 50)          # Adjust as needed

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Define the ROI as the forehead region (upper 20% of the face)
            roi_y_end = y + int(0.2 * h)
            roi = frame[y:roi_y_end, x:x+w]
            # Resize ROI to a fixed size to keep consistency
            roi_fixed = cv2.resize(roi, fixed_roi_size)
            cv2.rectangle(display_frame, (x, y), (x+w, roi_y_end), (255, 0, 0), 2)
            
            # Compute the average green intensity from the ROI
            green_avg = np.mean(roi_fixed[:, :, 1])
            roi_signal.append(green_avg)
            
            # Maintain the buffer length
            if len(roi_signal) > buffer_size:
                roi_signal.pop(0)

            # If buffer is full, compute heart rate
            if len(roi_signal) == buffer_size:
                bpm = compute_heart_rate(np.array(roi_signal), fs)
                if bpm is not None:
                    cv2.putText(display_frame, f"Heart Rate: {bpm:.1f} BPM", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print("Estimated Heart Rate: {:.1f} BPM".format(bpm))
                else:
                    cv2.putText(display_frame, "No valid frequency", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Reset the buffer to avoid reusing the same data
                roi_signal = []

        else:
            cv2.putText(display_frame, "No face detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the video feed
        cv2.imshow("See3Cam rPPG", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
