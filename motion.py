import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to extract keypoints from MediaPipe output
def extract_keypoints(landmarks, frame_shape):
    keypoints = []
    for landmark in landmarks:
        keypoints.append([landmark.x * frame_shape[1], landmark.y * frame_shape[0], landmark.z])
    return np.array(keypoints)

# Function to check if the body is still
def is_still(keypoints, previous_keypoints, threshold=50):
    """Check if the body is still by comparing keypoints."""
    if previous_keypoints is None:
        return True  # If there's no previous keypoints, assume stillness
    
    movement = np.linalg.norm(keypoints - previous_keypoints, axis=1)
    return np.all(movement < threshold)  # True if all movements are below the threshold

# Function to classify activity based on keypoints
def classify_activity(keypoints, previous_keypoints):
    # Assuming the following indices for keypoints:
    left_shoulder = keypoints[11]  # Left shoulder
    right_shoulder = keypoints[12]  # Right shoulder
    left_knee = keypoints[25]  # Left knee
    right_knee = keypoints[26]  # Right knee
    left_ankle = keypoints[27]  # Left ankle (toe)
    right_ankle = keypoints[28]  # Right ankle (toe)

    # Calculate Y-coordinates for keypoints
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    knee_y = (left_knee[1] + right_knee[1]) / 2
    ankle_y = (left_ankle[1] + right_ankle[1]) / 2

    # Check for horizontal alignment (sleeping)
    if abs(shoulder_y - knee_y) < 50 and abs(knee_y - ankle_y) < 50:
        return "Sleeping"

    # Check for walking based on shoulder distance
    shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_distance > 100:  # Threshold distance for walking
        return "Walking"

    # Check if the user is still
    if is_still(keypoints, previous_keypoints):
        return "Sitting"

    return "Standing"  # Default state if not walking, sleeping, or sitting

# Function to smooth status over time (averaging predictions)
def smooth_predictions(prediction_history):
    """Returns the most frequent prediction over a window of time."""
    return max(set(prediction_history), key=prediction_history.count)

# Function to draw room boundaries on the frame
def draw_room_boundaries(frame):
    cv2.rectangle(frame, (ROOM_BOUNDARIES['x_min'], ROOM_BOUNDARIES['y_min']),
                  (ROOM_BOUNDARIES['x_max'], ROOM_BOUNDARIES['y_max']), (0, 255, 0), 2)

def main():
    global ROOM_BOUNDARIES  # Access room boundaries globally
    cap = cv2.VideoCapture(0)

    # Frame dimensions
    frame_width = 1280
    frame_height = 720

    # Set room boundaries equal to frame dimensions
    ROOM_BOUNDARIES = {
        'x_min': 0,
        'x_max': frame_width,  # 1280
        'y_min': 0,
        'y_max': frame_height,  # 720
    }

    # Queue to hold predictions over a period for smoothing
    prediction_history = deque(maxlen=90)  # Store the last 90 frames (~3 seconds of predictions)

    previous_keypoints = None  # Initialize previous keypoints

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # Get frame dimensions
            frame_height, frame_width, _ = frame.shape

            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            # Extract landmarks and draw them on the frame
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                # Extract body keypoints
                keypoints = extract_keypoints(result.pose_landmarks.landmark, frame.shape)

                # Classify activity based on keypoints
                activity = classify_activity(keypoints, previous_keypoints)

                # Add activity to the prediction history
                prediction_history.append(activity)

                # Smooth the prediction over time
                smoothed_activity = smooth_predictions(prediction_history)

                # Display the smoothed activity on the frame
                cv2.putText(frame, f"Activity: {smoothed_activity}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Update previous keypoints for the next iteration
                previous_keypoints = keypoints

            # Draw room boundaries on the frame
            draw_room_boundaries(frame)

            # Resize the frame to 1280x720
            frame = cv2.resize(frame, (1280, 720))
            cv2.imshow('Body Tracking and Activity Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
