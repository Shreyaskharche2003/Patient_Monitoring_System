import cv2
import dlib
import numpy as np

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to check if the lips and nose are covered (indicating a mask is worn)
def is_mask_worn(landmarks, frame):
    # Get the landmarks for the mouth, chin, and nose
    mouth_points = [landmarks.part(i) for i in range(48, 68)]  # Landmarks for the mouth (48-67)
    chin_point = landmarks.part(8)  # Chin point
    nose_points = [landmarks.part(i) for i in range(27, 36)]  # Landmarks for the nose (27-35)

    # Combine all relevant points for mask detection
    mask_area_points = mouth_points + [chin_point] + nose_points

    # Create an empty mask for analysis
    mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Draw filled contour on the mask image
    mask_contour = np.array([(p.x, p.y) for p in mask_area_points], dtype=np.int32)
    cv2.fillPoly(mask_img, [mask_contour], 255)

    # Use edge detection for better transparency detection (Oxygen masks have soft, transparent edges)
    edges = cv2.Canny(frame, 50, 150)
    edge_mean_color = cv2.mean(edges, mask=mask_img)[:3]

    # Calculate the mean color in the mask area of the original frame
    mean_color = cv2.mean(frame, mask=mask_img)[:3]

    # Calculate brightness based on the mean color (using only color for better contrast detection)
    brightness = (mean_color[0] + mean_color[1] + mean_color[2]) / 3

    # Using edge brightness to refine mask detection
    edge_brightness = (edge_mean_color[0] + edge_mean_color[1] + edge_mean_color[2]) / 3

    # Dynamic threshold adjustment for mask detection based on brightness and edge details
    threshold_brightness = 80  # This threshold can be adjusted based on lighting and background
    edge_threshold = 30  # Adjust edge threshold to help detect mask more effectively

    # Decide if the mask is worn based on a combination of brightness and edge detection
    mask_worn = (brightness < threshold_brightness) and (edge_brightness < edge_threshold)

    return mask_worn, mask_area_points

# Access webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frames from the webcam
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    for face in faces:
        # Predict facial landmarks for the detected face
        landmarks = predictor(gray, face)

        # Check if the user is wearing an oxygen mask
        mask_worn, mask_area_points = is_mask_worn(landmarks, frame)

        # Draw mask area landmarks
        for point in mask_area_points:
            cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)  # Draw points around mask area

        # Draw contours around the mouth, chin, and nose
        mask_contour = np.array([(p.x, p.y) for p in mask_area_points], dtype=np.int32)
        cv2.polylines(frame, [mask_contour], isClosed=True, color=(255, 0, 0), thickness=1)

        # Display status on the video feed
        label = "Mask Removed" if mask_worn else "Mask Worn"
        color = (0, 255, 0) if not mask_worn else (0, 0, 255)
        cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show the frame with the mask status
    cv2.imshow("Oxygen Mask Detection", frame)

    # Check if 'q' is pressed or if the window is closed
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Oxygen Mask Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
