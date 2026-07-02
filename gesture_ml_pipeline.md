# Gesture Recognition ML Pipeline — Aura Music Instrument

This document covers the full ML component of the Aura gesture-music project:
why the pipeline is built this way, the complete code, and how to run it end to end.
This is what upgrades the project from a rule-based CV demo (finger counting) into
a genuine trained ML system, which is the part worth writing up in your report.

---

## 1. Theory

### 1.1 Why replace finger-counting with a trained classifier?
Finger-counting (checking if a fingertip's y-coordinate is above its knuckle) is a
hardcoded rule, not machine learning. A trained classifier instead **learns decision
boundaries from real examples of your hand**, which means:
- It can recognize gestures that aren't just "N fingers up" (e.g. rock-on, pinch, OK sign).
- It's more robust to natural variation in how you hold your hand.
- It gives you an actual model to evaluate (accuracy, confusion matrix) for your report.

### 1.2 What are the features?
MediaPipe Hands outputs 21 landmarks per hand, each with (x, y, z) coordinates —
**63 raw numbers** per frame. These raw coordinates are in image space, so the same
gesture looks numerically different depending on where your hand is in the frame or
how close it is to the camera.

To fix that, every sample is **normalized** before training:
1. **Translation invariance** — subtract the wrist landmark (landmark 0) from every
   point, so the wrist becomes the origin (0,0,0).
2. **Scale invariance** — divide every coordinate by the distance from the wrist to
   the middle-finger MCP joint (landmark 9), a stable reference length. This means a
   gesture looks the same numerically whether your hand is close to or far from the camera.

This turns raw pixel-dependent landmarks into a **63-dimensional feature vector**
that represents hand *shape*, independent of position or distance.

### 1.3 Why Random Forest?
For a small-to-medium tabular dataset like this (a few hundred–thousand samples,
63 numeric features), Random Forest is a strong default:
- Handles non-linear decision boundaries between gesture classes well.
- Needs far less data and tuning than a neural network.
- Naturally resistant to overfitting via ensembling many decision trees.
- Gives interpretable feature importances if you want to discuss which landmarks
  matter most in your report.

(You can swap in an SVM or a small MLP later as a comparison — the pipeline
below is structured so that's a one-line change in `train_gesture_classifier.py`.)

### 1.4 Suggested gesture set
| Label | Description | Suggested note/chord mapping |
|---|---|---|
| `fist` | closed hand | rest / mute |
| `one` | index finger up | C4 |
| `two` | index + middle up | E4 |
| `three` | index + middle + ring up | G4 |
| `four` | four fingers up, thumb tucked | C5 |
| `open_palm` | all five fingers spread | C major chord |
| `pinch` | thumb + index touching | pitch bend / filter control |
| `rock_on` | index + pinky up, others down | custom effect (e.g. distortion/echo) |

You choose the final set — 6–8 classes is a good range for a course project:
enough to be interesting, small enough to collect data for in one sitting.

### 1.5 Pipeline overview
```
webcam --> MediaPipe Hands --> 21 landmarks --> normalize --> 63-d feature vector
                                                                     |
                        collect_gesture_data.py  (labeled CSV rows) |
                                                                     v
                                              train_gesture_classifier.py
                                                    (Random Forest model)
                                                                     |
                                                                     v
                                              realtime_predict.py (live test)
                                                                     |
                                                                     v
                                    wire prediction into Aura's note-mapping logic
```

---

## 2. Setup

```bash
pip install opencv-python mediapipe scikit-learn pandas joblib
```

Files in this pipeline:
- `collect_gesture_data.py` — records labeled landmark samples from your webcam
- `train_gesture_classifier.py` — trains and evaluates the Random Forest model
- `realtime_predict.py` — runs the trained model live for testing

---

## 3. Code

### 3.1 `collect_gesture_data.py`
Records normalized landmark vectors for one gesture at a time and appends them to `gesture_data.csv`.

```python
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
```

### 3.2 `train_gesture_classifier.py`
Trains the Random Forest on the collected CSV and reports accuracy + confusion matrix.

```python
"""
train_gesture_classifier.py
----------------------------
Trains a classifier on the landmark dataset produced by collect_gesture_data.py
and saves the trained model + label encoder for use in realtime_predict.py.

Usage:
    python train_gesture_classifier.py --data gesture_data.csv
"""

import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="gesture_data.csv")
    parser.add_argument("--model_out", default="gesture_model.pkl")
    parser.add_argument("--encoder_out", default="label_encoder.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} samples across {df['label'].nunique()} gesture classes:")
    print(df["label"].value_counts())

    X = df.drop(columns=["label"]).values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # RandomForest is a strong default for small/medium tabular landmark data:
    # robust to feature scale differences, handles non-linear gesture boundaries,
    # and needs far less data/tuning than a neural net for this problem size.
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(clf, args.model_out)
    joblib.dump(encoder, args.encoder_out)
    print(f"\nSaved model to {args.model_out}")
    print(f"Saved label encoder to {args.encoder_out}")


if __name__ == "__main__":
    main()
```

### 3.3 `realtime_predict.py`
Runs the trained model live on webcam input so you can sanity-check accuracy before wiring it into Aura.

```python
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
```

---

## 4. Execution guide

1. **Install dependencies**
   ```bash
   pip install opencv-python mediapipe scikit-learn pandas joblib
   ```

2. **Collect data for each gesture** — run once per gesture class, aim for 200–300 samples each,
   varying hand angle/distance slightly between runs for a more robust model:
   ```bash
   python collect_gesture_data.py --label fist --samples 250
   python collect_gesture_data.py --label one --samples 250
   python collect_gesture_data.py --label two --samples 250
   python collect_gesture_data.py --label three --samples 250
   python collect_gesture_data.py --label four --samples 250
   python collect_gesture_data.py --label open_palm --samples 250
   python collect_gesture_data.py --label pinch --samples 250
   python collect_gesture_data.py --label rock_on --samples 250
   ```
   All samples append into a single `gesture_data.csv`.

3. **Train the classifier**
   ```bash
   python train_gesture_classifier.py --data gesture_data.csv
   ```
   This prints per-class precision/recall/F1 and a confusion matrix — copy this
   output directly into your project report's evaluation section.

4. **Test live**
   ```bash
   python realtime_predict.py
   ```
   Confirm each gesture is recognized correctly and confidence scores look reasonable
   (>0.8 for clean gestures). If a class is consistently confused with another,
   collect more samples for that class and retrain.

5. **Integrate into Aura**
   Once accuracy is solid, replace the rule-based `countFingers()` logic in the
   Aura web app with calls to this model. The cleanest way to bridge Python (model)
   and the browser (Aura's UI) is a small local WebSocket or Flask/FastAPI endpoint:
   the Python side runs `realtime_predict.py`-style inference and streams the
   predicted gesture label to the browser, which then triggers the note/chord
   exactly as it does now.

---

## 5. What to report

- **Problem framing**: gesture-to-music interface, accessibility + NIME motivation (see earlier discussion).
- **Data**: number of samples per class, collection protocol (webcam, MediaPipe landmarks, normalization).
- **Model**: Random Forest, hyperparameters used, why chosen over rule-based approach.
- **Evaluation**: accuracy, confusion matrix, per-class precision/recall — include the actual output from `train_gesture_classifier.py`.
- **System integration**: how predictions map to notes/chords and drive the reactive 3D visual.
- **Limitations & future work**: single-hand only, lighting sensitivity, could extend to two-hand gestures or continuous gesture sequences (melodies) with an RNN/LSTM.
