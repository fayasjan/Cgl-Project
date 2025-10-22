import cv2
import numpy as np
import mediapipe as mp
import os # Added for os.path.basename

# Note: The backend needs to initialize MediaPipe Pose similarly to how it's done in the notebook
# Example initialization (should be done once globally in the backend):
# mp_pose = mp.solutions.pose
# pose_detector = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# --- Define FRAMES_PER_VIDEO (Must match the value used for training) ---
FRAMES_PER_VIDEO = 107

def extract_keypoints_from_frame(frame, pose_detector):
    """Extract 3D keypoints (33, 3) from a single frame."""
    # Ensure pose_detector is passed or accessible globally
    results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_world_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark])
    return None

def uniform_sample_frames(frames, target_count):
    """Uniformly samples or pads (repeats) frames to reach target_count."""
    if not frames: return []
    num_available = len(frames)
    if num_available == 0: return []

    if num_available < target_count:
        # Upsample by repeating frames
        indices = np.linspace(0, num_available - 1, target_count, dtype=int)
    else:
        # Downsample evenly
        indices = np.round(np.linspace(0, num_available - 1, target_count)).astype(int)

    # Ensure indices are valid
    indices = np.clip(indices, 0, num_available - 1)
    return [frames[i] for i in indices]

def process_video_for_backend(video_path, pose_detector, target_frames):
    """
    Extracts keypoint frames from a video and applies uniform sampling.
    Returns a NumPy array of shape (target_frames, 33, 3) or None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    frames_with_keypoints = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        keypoints = extract_keypoints_from_frame(frame, pose_detector)
        if keypoints is not None:
            frames_with_keypoints.append(keypoints)
    cap.release()

    if frames_with_keypoints:
        sampled_frames = uniform_sample_frames(frames_with_keypoints, target_frames)
        if len(sampled_frames) == target_frames:
             return np.array(sampled_frames)
        else:
             print(f"Warning: Sampling failed for {os.path.basename(video_path)}. Expected {target_frames}, got {len(sampled_frames)}.")
             return None
    else:
        print(f"Warning: No valid keypoints found in {os.path.basename(video_path)}.")
        return None

# Example Usage (How your friend might use it in the backend):
# 1. Initialize pose_detector once globally
# mp_pose = mp.solutions.pose
# pose_detector = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
#
# 2. In the request handler:
# sequence_data = process_video_for_backend(uploaded_video_path, pose_detector, FRAMES_PER_VIDEO)
# if sequence_data is not None:
#     # Proceed with normalization and prediction
#     pass
# else:
#     # Handle preprocessing failure
#     pass