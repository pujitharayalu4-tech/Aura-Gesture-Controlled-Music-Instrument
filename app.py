import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os  # Import os module for file existence checks

# Import the normalization function from your collect_gesture_data.py
# Ensure collect_gesture_data.py is in the same directory or accessible in your Python path
try:
    from collect_gesture_data import normalize_landmarks
except ImportError:
    st.error("Could not import `normalize_landmarks` from `collect_gesture_data.py`. "
             "Please ensure `collect_gesture_data.py` is in the same directory "
             "or accessible in your Python path.")
    st.stop()

# --- Configuration ---
MODEL_PATH = "gesture_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
# Minimum confidence score for a prediction to be considered valid.
# Adjust this value based on your model's performance and desired strictness.
MIN_CONFIDENCE_THRESHOLD = 0.8

# --- Load Model and Encoder ---
try:
    # Check if model and encoder files exist before loading
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at '{MODEL_PATH}'. "
                 "Please run `train_gesture_classifier.py` first to generate it.")
        st.stop()
    if not os.path.exists(ENCODER_PATH):
        st.error(f"Label encoder file not found at '{ENCODER_PATH}'. "
                 "Please run `train_gesture_classifier.py` first to generate it.")
        st.stop()

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    st.success("Gesture recognition model and encoder loaded successfully.")
except Exception as e:
    st.error(f"Error loading model or encoder: {e}")
    st.stop()

# --- MediaPipe Setup ---
# Initialize MediaPipe Hands
# We set max_num_hands to 1 for simplicity in this example,
# but you can adjust it if your trained model supports multiple hands.
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1, # Adjust if your model was trained for multiple hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# --- Streamlit App Layout ---
st.set_page_config(page_title="Aura Gesture Recognition", layout="wide")
st.title("Aura Gesture Recognition")
st.markdown("""
This application demonstrates real-time gesture recognition using your webcam.
The system uses a pre-trained **Random Forest classifier** to identify hand gestures.
Make sure your hand is clearly visible to the camera.
""")

# --- Camera Input and Image Processing ---
# Use st.camera_input for a simple camera feed integration.
# It returns a file-like object containing the image bytes when feed is available.
camera_feed = st.camera_input("Your webcam feed", placeholder="Waiting for camera access...")

# Placeholder for displaying the annotated image
annotated_image_placeholder = st.empty()
# Placeholder for displaying the prediction status
status_placeholder = st.empty()

if camera_feed is not None:
    # To read the image from the camera feed:
    bytes_data = camera_feed.getvalue()
    
    # Convert the image bytes to an OpenCV image format (NumPy array)
    try:
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Check if image decoding was successful
        if cv2_img is None:
            status_placeholder.error("Failed to decode camera image. Please retry.")
            
        else:
            # Flip the image horizontally (like a mirror view)
            cv2_img = cv2.flip(cv2_img, 1)
            
            # Convert the BGR image to RGB format, as MediaPipe expects RGB
            rgb_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe Hands
            results = hands.process(rgb_image)

            # Create a copy of the original image to draw landmarks on
            annotated_image = cv2_img.copy()

            detected_gesture = "No hand detected"
            prediction_confidence = 0.0
            
            if results.multi_hand_landmarks:
                # Process each detected hand (assuming max_num_hands=1 here)
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the hand landmarks and connections
                    mp_draw.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                    )

                    # Normalize landmarks for the current hand
                    try:
                        features = normalize_landmarks(hand_landmarks)
                        # Ensure we have the correct number of features (63: 21 landmarks * 3 coords)
                        if len(features) == 63:
                            # Predict the gesture
                            prediction_raw = model.predict([features])
                            # Get the probability for the predicted class
                            current_prediction_probs = model.predict_proba([features])[0]
                            predicted_class_index = model.classes_.tolist().index(prediction_raw[0])
                            probability = current_prediction_probs[predicted_class_index]
                            
                            # Apply confidence threshold
                            if probability >= MIN_CONFIDENCE_THRESHOLD:
                                predicted_label_raw = label_encoder.inverse_transform(prediction_raw)[0]
                                detected_gesture = f"{predicted_label_raw}"
                                prediction_confidence = probability
                            else:
                                detected_gesture = "Uncertain"
                                prediction_confidence = probability
                        else:
                            st.warning(f"Feature extraction returned {len(features)} features, expected 63.")
                            detected_gesture = "Feature error"
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        detected_gesture = "Prediction error"
            
            # Display the annotated image in Streamlit
            # Use use_column_width=True to fit the image to the column width
            annotated_image_placeholder.image(annotated_image, channels="BGR", use_column_width=True)

            # Update status and prediction display
            status_text = f"**Gesture:** {detected_gesture}"
            if detected_gesture in ["No hand detected", "Feature error", "Prediction error"]:
                pass # No confidence to show for these states
            elif detected_gesture == "Uncertain":
                status_text += f" (Confidence: {prediction_confidence:.2f})"
            else: # A recognized gesture
                status_text += f" ({prediction_confidence:.2f})"
            
            status_placeholder.markdown(status_text)

    except Exception as e:
        status_placeholder.error(f"An error occurred during image processing: {e}")
else:
    # Display a placeholder image if camera feed is not yet available
    # A placeholder image can be a URL or a local path if you have one.
    # Using a placeholder URL for now.
    annotated_image_placeholder.image(
        "https://via.placeholder.com/640x480.png?text=Waiting+for+camera...", 
        caption="Camera feed placeholder", 
        use_column_width=True
    )
    status_placeholder.info("Please ensure your camera is accessible and grant permissions to start.")

# --- Sidebar Information ---
st.sidebar.header("About")
st.sidebar.info("""
This app uses a trained model to recognize hand gestures for the Aura Music Instrument.
Make sure to collect sufficient data for each gesture using `collect_gesture_data.py` 
and train the classifier with `train_gesture_classifier.py` before running this app.
""")
st.sidebar.header("Model Performance")
st.sidebar.info(f"""
- Model: **Random Forest**
- Minimum prediction confidence: **{MIN_CONFIDENCE_THRESHOLD:.2f}**
""")