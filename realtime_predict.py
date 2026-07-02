"""
realtime_predict.py
--------------------
Loads the trained gesture classifier and runs live predictions from the
webcam feed, so you can verify accuracy before wiring it into the Aura
music instrument.

Usage:
    python realtime_predict.py
"""

import cv2
import joblib
import mediapipe as mp

from collect_gesture_data import normalize_landmarks  # reuse the same normalization


def main():
    clf = joblib.load("gesture_model.pkl")
    encoder = joblib.load("label_encoder.pkl")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        label_text = "no hand detected"
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = normalize_landmarks(hand_landmarks)
            pred_idx = clf.predict([features])[0]
            pred_proba = clf.predict_proba([features])[0][pred_idx]
            label_text = f"{encoder.inverse_transform([pred_idx])[0]}  ({pred_proba:.2f})"

        cv2.putText(frame, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Gesture Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
