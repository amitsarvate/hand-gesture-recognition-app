import cv2
import mediapipe as mp
import numpy as np
import csv

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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
    elif cv2.waitKey(1) & 0xFF == ord('a'):
        gesture_label = 6  # Label for  'A'
    elif cv2.waitKey(1) & 0xFF == ord('b'):
        gesture_label = 7  # Label for  'B'
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        gesture_label = 8  # Label for  'C'
    elif cv2.waitKey(1) & 0xFF == ord('d'):
        gesture_label = 9 # Label for "D"
    elif cv2.waitKey(1) & 0xFF == ord('e'):
        gesture_label = 10 # Label for "e"
    elif cv2.waitKey(1) & 0xFF == ord('f'):
        gesture_label = 11  # Label for 'f'
    elif cv2.waitKey(1) & 0xFF == ord('g'):
        gesture_label = 12  # Label for 'g'
    elif cv2.waitKey(1) & 0xFF == ord('h'):
        gesture_label = 13  # Label for 'h'
    elif cv2.waitKey(1) & 0xFF == ord('i'):
        gesture_label = 14  # Label for 'i'
    elif cv2.waitKey(1) & 0xFF == ord('j'):
        gesture_label = 15  # Label for 'j'
    elif cv2.waitKey(1) & 0xFF == ord('k'):
        gesture_label = 16  # Label for 'k'
    elif cv2.waitKey(1) & 0xFF == ord('l'):
        gesture_label = 17  # Label for 'l'
    elif cv2.waitKey(1) & 0xFF == ord('m'):
        gesture_label = 18  # Label for 'm'
    elif cv2.waitKey(1) & 0xFF == ord('n'):
        gesture_label = 19  # Label for 'n'
    elif cv2.waitKey(1) & 0xFF == ord('o'):
        gesture_label = 20  # Label for 'o'
    elif cv2.waitKey(1) & 0xFF == ord('p'):
        gesture_label = 21  # Label for 'p'
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        gesture_label = 22  # Label for 'q'
    elif cv2.waitKey(1) & 0xFF == ord('r'):
        gesture_label = 23  # Label for 'r'
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        gesture_label = 24  # Label for 's'
    elif cv2.waitKey(1) & 0xFF == ord('t'):
        gesture_label = 25  # Label for 't'
    elif cv2.waitKey(1) & 0xFF == ord('u'):
        gesture_label = 26  # Label for 'u'
    elif cv2.waitKey(1) & 0xFF == ord('v'):
        gesture_label = 27  # Label for 'v'
    elif cv2.waitKey(1) & 0xFF == ord('w'):
        gesture_label = 28  # Label for 'w'
    elif cv2.waitKey(1) & 0xFF == ord('x'):
        gesture_label = 29  # Label for 'x'
    elif cv2.waitKey(1) & 0xFF == ord('y'):
        gesture_label = 30  # Label for 'y'
    elif cv2.waitKey(1) & 0xFF == ord('z'):
        gesture_label = 31  # Label for 'z'


    if results.multi_hand_landmarks and gesture_label:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
            log_landmarks(landmarks, gesture_label)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Data Collection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
