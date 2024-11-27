import cv2
import mediapipe as mp
import numpy as np
import csv

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Apply Gaussian Blur to the frame
def apply_gaussian_blur(frame):
    return cv2.GaussianBlur(frame, (5, 5), 0)

# Convert to grayscale
def convert_to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Histogram Equalization
def histogram_equalization(frame):
    return cv2.equalizeHist(frame)

# Canny Edge Detection
def canny_edge_detection(frame):
    return cv2.Canny(frame, 100, 200)

# Apply sharpening filter
def sharpen_image(frame):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])  # Sharpening kernel
    return cv2.filter2D(frame, -1, kernel)

# Mapping keys (1-5) to gesture labels
def log_landmarks(landmarks, label):
    with open("data/gesture_data.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(np.hstack((landmarks.flatten(), label)))

# Initialize MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Apply computational photography techniques
    # Uncomment one of the lines below to apply specific techniques

    # Optionally apply Gaussian Blur
    # frame = apply_gaussian_blur(frame)

    # Convert to grayscale
    # frame = convert_to_grayscale(frame)

    # Apply Histogram Equalization
    # frame = histogram_equalization(frame)

    # Apply Canny Edge Detection (for edge focus)
    # frame = canny_edge_detection(frame)

    # Apply Sharpening
    frame = sharpen_image(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_label = None
    if cv2.waitKey(1) & 0xFF == ord('1'):
        gesture_label = 1  # Label for "Thumbs Up"
    elif cv2.waitKey(1) & 0xFF == ord('2'):
        gesture_label = 2  # Label for "OK Sign"
    elif cv2.waitKey(1) & 0xFF == ord('3'):
        gesture_label = 3  # Label for "Victory"
    elif cv2.waitKey(1) & 0xFF == ord('4'):
        gesture_label = 4  # Label for "Fist"
    elif cv2.waitKey(1) & 0xFF == ord('5'):
        gesture_label = 5  # Label for "Stop Sign"

    if results.multi_hand_landmarks and gesture_label:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
            log_landmarks(landmarks, gesture_label)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Data Collection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
