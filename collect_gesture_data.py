"""
collect_gesture_data.py
------------------------
Records hand-landmark samples for one gesture at a time and appends them
to a CSV dataset used later to train the gesture classifier.

Usage:
    python collect_gesture_data.py --label open_palm --samples 200
    python collect_gesture_data.py --label fist --samples 200
    python collect_gesture_data.py --label one --samples 200
    ... repeat for every gesture class ...

Controls while running:
    SPACE  -> start/pause capturing
    q      -> quit early

Output:
    gesture_data.csv  (appended, created if missing)
    Columns: label, x0,y0,z0, x1,y1,z1, ... x20,y20,z20   (63 landmark features)
"""

import argparse
import csv
import os
import cv2
import mediapipe as mp

FEATURE_COLUMNS = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def normalize_landmarks(landmarks):
    """
    Make landmarks translation- and scale-invariant so the classifier
    generalizes across hand position/distance from camera:
      1. Shift everything so the wrist (landmark 0) is the origin.
      2. Scale by the distance wrist -> middle-finger MCP (landmark 9),
         a stable reference length across gestures.
    """
    pts = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
    wx, wy, wz = pts[0]
    shifted = [(x - wx, y - wy, z - wz) for x, y, z in pts]

    ref_x, ref_y, ref_z = shifted[9]
    scale = max((ref_x ** 2 + ref_y ** 2 + ref_z ** 2) ** 0.5, 1e-6)

    normalized = []
    for x, y, z in shifted:
        normalized.extend([x / scale, y / scale, z / scale])
    return normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Gesture name, e.g. fist, one, two, open_palm, pinch, rock_on")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to record")
    parser.add_argument("--output", default="gesture_data.csv", help="CSV file to append to")
    args = parser.parse_args()

    file_exists = os.path.isfile(args.output)
    csv_file = open(args.output, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(FEATURE_COLUMNS)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(0)
    capturing = False
    collected = 0

    print(f"Ready to record gesture '{args.label}'. Press SPACE to start, q to quit.")

    while cap.isOpened() and collected < args.samples:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if capturing:
                row = [args.label] + normalize_landmarks(hand_landmarks)
                writer.writerow(row)
                collected += 1

        cv2.putText(frame, f"Label: {args.label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Collected: {collected}/{args.samples}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        status = "CAPTURING" if capturing else "PAUSED (press SPACE)"
        cv2.putText(frame, status, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("Gesture Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            capturing = not capturing
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"Done. Saved {collected} samples for '{args.label}' to {args.output}")


if __name__ == "__main__":
    main()
