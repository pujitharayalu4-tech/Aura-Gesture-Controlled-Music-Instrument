# Aura — Gesture-Controlled Music Instrument

Play music with your bare hands. Aura tracks both hands through your webcam,
turns finger gestures into notes and chords, and renders a reactive
particle "VR" background that pulses, breathes, and sparkles in sync with
what you play.

No headset, no controllers — just your hands and a browser.

---

## ✨ Features

- **Two-hand play** — right hand plays the melody, left hand plays the bass line
- **Gesture-to-note mapping** — finger count (1–5) triggers notes and chords
- **Pinch to bend tone** — hold a pinch and vary its width to sweep a filter/reverb in real time
- **Tap gesture** — a quick pinch snap fires a bell-like chime and a burst of golden sparkle particles, fairy-tale style
- **Reactive 3D background** — a Three.js particle field that continuously responds to the actual audio output (not just on note triggers), plus color shifts per note and a glowing core
- **Runs entirely in-browser** — no installs beyond a local server; camera and audio processing happen client-side

---

## 🧠 Why this project

Gesture-controlled instruments sit at the intersection of computer vision,
human-computer interaction, and accessible design — this is a hands-on,
demoable exploration of that space (see the [New Interfaces for Musical
Expression](https://www.nime.org/) field for the academic context). It's also
a practical touchless-interface pattern and an approachable way to make music
without needing to learn a traditional instrument first.

---

## 🛠️ Tech stack

| Layer | Tool |
|---|---|
| Hand tracking | [MediaPipe Hands](https://developers.google.com/mediapipe) |
| Audio synthesis | [Tone.js](https://tonejs.github.io/) |
| 3D visuals | [Three.js](https://threejs.org/) |
| Everything else | Vanilla JavaScript, HTML, CSS |

---

## 🖐️ Gesture guide

**Right hand — melody**
| Gesture | Note |
|---|---|
| 1 finger | C4 |
| 2 fingers | D4 |
| 3 fingers | E4 |
| 4 fingers | G4 |
| Open palm | C major chord |

**Left hand — bass**
| Gesture | Note |
|---|---|
| 1 finger | C3 |
| 2 fingers | D3 |
| 3 fingers | E3 |
| 4 fingers | G3 |
| Open palm | C3 pad |

**Either hand**
| Gesture | Effect |
|---|---|
| Hold a pinch, vary width | Tone bend (filter + reverb sweep) |
| Quick pinch snap | ✨ Chime + sparkle burst |

---

## 🚀 Running it locally

Hand tracking and camera access require a local server — opening the HTML
file directly (`file://`) will not work in most browsers.

```bash
# from the folder containing gesture-music-vr.html
python3 -m http.server 8000
```

Then open **http://localhost:8000/gesture-music-vr.html** in Chrome.

Click **Enter the aura**, allow camera and microphone/audio permissions, and
show one or both hands to the camera.

> If the camera doesn't activate, check the camera icon in your browser's
> address bar and make sure this page is set to **Allow**, then reload.

---

## 🤖 ML extension (in progress)

The current build recognizes gestures with simple geometric rules (comparing
fingertip and knuckle positions). A companion pipeline trains an actual
classifier on hand-landmark data instead:

- `collect_gesture_data.py` — records labeled landmark samples via webcam
- `train_gesture_classifier.py` — trains a Random Forest on normalized
  landmark features and reports accuracy/confusion matrix
- `realtime_predict.py` — runs the trained model live for testing

See `gesture_ml_pipeline.md` for the full theory, code, and evaluation guide.
This is what lets the system recognize custom gestures beyond simple finger
counts (like the rock-on sign) and gives a trained model to report on.

---

## 📁 Project structure

```
gesture-music-vr.html          # main app — hand tracking, audio, visuals
collect_gesture_data.py        # ML: webcam data collection
train_gesture_classifier.py    # ML: model training + evaluation
realtime_predict.py            # ML: live inference test
gesture_ml_pipeline.md         # ML: theory, code, execution guide
README.md                      # this file
```

---

## 🔭 Future work

- Swap rule-based gesture detection for the trained classifier
- Two-hand combination gestures (e.g. both palms open = special effect)
- Record/export a played sequence as an audio file
- Deploy to a real WebXR headset for true VR hand tracking

---

## 📄 License

Personal/academic project — feel free to fork and extend for your own coursework.
