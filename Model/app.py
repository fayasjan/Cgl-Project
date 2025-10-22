from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import numpy as np
import os
import time
import cv2  # OpenCV for video processing
import mediapipe as mp  # MediaPipe for pose estimation
import joblib  # For loading the StandardScaler
import pickle  # For loading the gallery embeddings

# --- Configuration ---
MODEL_FILENAME = 'siamese_model_ext107_bs8_e100.keras'
SCALER_FILENAME = 'scaler.joblib'
GALLERY_FILENAME = 'gallery_embeddings.pkl'
UPLOAD_FOLDER = 'uploads'
EXPECTED_FRAMES = 107
EXPECTED_FEATURES = 99  # 33 keypoints * 3 coordinates
AUTH_THRESHOLD = 0.8  # Start with a reasonable threshold and adjust after testing

# --- Custom Function Definitions (Needed for Model Loading) ---
def euclidean_distance(vectors):
    """Calculates the euclidean distance between two vectors."""
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred):
    """Calculates the contrastive loss."""
    margin = 1.0  # Must match the margin used during training
    y_true = tf.cast(y_true, tf.float32)
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# --- Initialize Global Variables ---
pose_detector = None
scaler = None
siamese_model = None
tower_model = None  # This will be our extracted model for generating embeddings
gallery_embeddings = {}

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Startup Loading Sequence ---
def load_dependencies():
    """Loads all necessary models and files at server startup."""
    global pose_detector, scaler, siamese_model, tower_model, gallery_embeddings

    print("\n--- Initializing Server Dependencies ---")
    try:
        mp_pose = mp.solutions.pose
        pose_detector = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        print("✅ MediaPipe Pose Detector initialized.")
    except Exception as e:
        print(f"❌ CRITICAL: Failed to initialize MediaPipe Pose Detector: {e}")

    try:
        scaler = joblib.load(SCALER_FILENAME)
        print(f"✅ Scaler loaded from {SCALER_FILENAME}.")
    except Exception as e:
        print(f"❌ CRITICAL: Failed to load scaler from {SCALER_FILENAME}: {e}")

    try:
        siamese_model = load_model(
            MODEL_FILENAME,
            custom_objects={
                'euclidean_distance': euclidean_distance,
                'contrastive_loss': contrastive_loss
            }
        )
        print(f"✅ Full Siamese model loaded from {MODEL_FILENAME}.")
    except Exception as e:
        print(f"❌ CRITICAL: Failed to load Siamese model from {MODEL_FILENAME}: {e}")

    if siamese_model:
        try:
            TOWER_MODEL_NAME = 'functional'
            tower_layer = siamese_model.get_layer(TOWER_MODEL_NAME)
            tower_model = Model(inputs=tower_layer.input, outputs=tower_layer.output, name="embedding_generator")
            print(f"✅ Tower model ('{TOWER_MODEL_NAME}') extracted successfully.")
        except ValueError:
            print(f"❌ CRITICAL: Could not find the tower layer named '{TOWER_MODEL_NAME}'.")
            tower_model = None

    try:
        with open(GALLERY_FILENAME, 'rb') as f:
            gallery_embeddings = pickle.load(f)
        print(f"✅ Gallery loaded with {len(gallery_embeddings)} users from {GALLERY_FILENAME}.")
    except Exception as e:
        print(f"❌ WARNING: Could not load gallery from {GALLERY_FILENAME}: {e}")
    
    print("--- Server is Ready to Accept Requests ---\n")

# --- Preprocessing Helper Functions ---
def preprocess_video(video_path):
    """Processes video: extracts keypoints, samples, normalizes, and reshapes."""
    if scaler is None or pose_detector is None:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames_with_keypoints = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_world_landmarks:
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark])
            frames_with_keypoints.append(keypoints)
    cap.release()

    if not frames_with_keypoints:
        return None

    num_available = len(frames_with_keypoints)
    if num_available < EXPECTED_FRAMES:
        indices = np.linspace(0, num_available - 1, EXPECTED_FRAMES, dtype=int)
    else:
        indices = np.round(np.linspace(0, num_available - 1, EXPECTED_FRAMES)).astype(int)
    
    sampled_frames = [frames_with_keypoints[i] for i in indices]
    sequence_3d = np.array(sampled_frames)

    try:
        normalized_sequence = scaler.transform(sequence_3d.reshape(-1, 3)).reshape(sequence_3d.shape)
        reshaped_for_model = normalized_sequence.reshape(1, EXPECTED_FRAMES, EXPECTED_FEATURES)
        return reshaped_for_model.astype(np.float32)
    except Exception:
        return None

# --- Authentication Logic ---
def perform_authentication(probe_embedding, gallery):
    """Compares a probe embedding against all gallery embeddings."""
    min_distance = float('inf')
    matched_user = "Unknown"

    print("--- Comparing Probe to Gallery ---")
    probe_np = probe_embedding.flatten()

    for user_id, user_embeddings in gallery.items():
        for i, ref_embedding in enumerate(user_embeddings):
            ref_np = np.asarray(ref_embedding).flatten()
            distance = np.linalg.norm(probe_np - ref_np)
            print(f"   Dist({user_id}, Sample {i+1}): {distance:.4f}")
            if distance < min_distance:
                min_distance = distance
                matched_user = user_id

    authenticated_np = min_distance < AUTH_THRESHOLD # This is a numpy.bool_
    
    print("------------------------------------")
    print(f"🔍 Min Distance: {min_distance:.4f} (Closest Match: {matched_user})")
    print(f"🔑 Auth Decision (Threshold {AUTH_THRESHOLD}): {'ACCESS GRANTED' if authenticated_np else 'ACCESS DENIED'}")
    print("------------------------------------")

    # --- FIX IS HERE ---
    # Convert the NumPy boolean to a standard Python bool before returning.
    return {
        "authenticated": bool(authenticated_np), # <-- This is the correct fix!
        "user_id": matched_user if authenticated_np else None,
        "distance": float(min_distance)
    }

# --- API Endpoints ---
@app.route('/')
def index():
    return "Gait Authentication Backend is Running!"

@app.route('/authenticate', methods=['POST'])
def handle_authentication():
    if not all([tower_model, scaler, pose_detector]):
        return jsonify({"error": "Server is not ready. A required component failed to load."}), 503

    if 'gait_video' not in request.files:
        return jsonify({"error": "No video file part in the request"}), 400
    video_file = request.files['gait_video']

    if video_file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = f"{int(time.time())}_{video_file.filename}"
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        video_file.save(video_path)
        print(f"\n--- New Authentication Request ---")
        print(f"💾 Video saved to: {video_path}")

        preprocessed_data = preprocess_video(video_path)
        if preprocessed_data is None:
            raise ValueError("Preprocessing failed and returned None.")

        print("🧠 Generating embedding for probe video...")
        probe_embedding = tower_model.predict(preprocessed_data)

        result = perform_authentication(probe_embedding, gallery_embeddings)
        return jsonify(result)

    except Exception as e:
        print(f"❌ Error during authentication process: {e}")
        return jsonify({"error": "An internal error occurred", "details": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"🗑 Temp file deleted: {video_path}")

@app.route('/register', methods=['POST'])
def handle_registration():
    if not all([tower_model, scaler, pose_detector]):
        return jsonify({"error": "Server is not ready. A required component failed to load."}), 503

    if 'gait_video' not in request.files or 'user_id' not in request.form:
        return jsonify({"error": "Missing 'gait_video' or 'user_id' in the request"}), 400
    
    video_file = request.files['gait_video']
    user_id = request.form['user_id']
    if not all([video_file.filename, user_id]):
        return jsonify({"error": "Empty video file or user_id"}), 400

    filename = f"{int(time.time())}_{user_id}_{video_file.filename}"
    video_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        video_file.save(video_path)
        print(f"\n--- New Registration Request for user '{user_id}' ---")
        print(f"💾 Video saved to: {video_path}")
        
        preprocessed_data = preprocess_video(video_path)
        if preprocessed_data is None:
            raise ValueError("Preprocessing failed and returned None.")

        print(f"🧠 Generating embedding for new user '{user_id}'...")
        new_embedding = tower_model.predict(preprocessed_data)[0]

        gallery_embeddings.setdefault(user_id, []).append(new_embedding)
        print(f"✅ Embedding added for '{user_id}'. Total samples for user: {len(gallery_embeddings[user_id])}")

        with open(GALLERY_FILENAME, 'wb') as f:
            pickle.dump(gallery_embeddings, f)
        print(f"💾 Updated gallery saved to {GALLERY_FILENAME}")
        
        return jsonify({"status": "success", "user_id": user_id, "message": "User registered successfully"})

    except Exception as e:
        print(f"❌ Error during registration process: {e}")
        return jsonify({"error": "An internal error occurred", "details": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"🗑 Temp file deleted: {video_path}")

# --- Run the Server ---
if __name__ == '__main__':
    load_dependencies()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

