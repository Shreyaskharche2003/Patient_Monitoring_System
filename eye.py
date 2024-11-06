import cv2
import dlib
import numpy as np
import time
from twilio.rest import Client

# Twilio credentials (replace with your own)
account_sid = ""
auth_token = ""
twilio_phone_number = ''
recipient_phone_number = ''

# Initialize Twilio client
client = Client(account_sid, auth_token)

# Function to make a call using Twilio
def make_call():
    call = client.calls.create(
        to=recipient_phone_number,
        from_=twilio_phone_number,
        url='http://demo.twilio.com/docs/voice.xml'  # Sample Twilio URL for the call
    )       
    print(f"Call initiated, SID: {call.sid}")

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# EAR threshold to determine eye state
EYE_AR_THRESH = 0.2

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(np.array([eye_points[1].x, eye_points[1].y]) - np.array([eye_points[5].x, eye_points[5].y]))
    B = np.linalg.norm(np.array([eye_points[2].x, eye_points[2].y]) - np.array([eye_points[4].x, eye_points[4].y]))
    C = np.linalg.norm(np.array([eye_points[0].x, eye_points[0].y]) - np.array([eye_points[3].x, eye_points[3].y]))
    ear = (A + B) / (2.0 * C)
    return ear

# Access webcam
cap = cv2.VideoCapture(0)

# Variables to count open and closed eye states over 15 seconds
open_count = 0
closed_count = 0
start_time = time.time()

# Duration to check (15 seconds)
DURATION = 15

while True:
    # Read frames from the webcam
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    for face in faces:
        # Predict facial landmarks for the detected face
        landmarks = predictor(gray, face)

        # Get coordinates for the left and right eye using Dlib's 68-point model
        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the EARs of both eyes
        avg_ear = (left_ear + right_ear) / 2.0

        # Determine if the eyes are open or closed and update the counts
        if avg_ear < EYE_AR_THRESH:
            eye_status = "Eyes Closed"
            closed_count += 1
        else:
            eye_status = "Eyes Open"
            open_count += 1

        # Display status on the video feed
        cv2.putText(frame, eye_status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw circles around eyes
        for point in left_eye:
            cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)
        for point in right_eye:
            cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)

    # Check if 15 full seconds have passed
    elapsed_time = int(time.time() - start_time)
    if elapsed_time >= DURATION:
        # Compare open and closed counts
        if open_count > closed_count:
            make_call()  # Make the call if eyes were open more
        else:
            print("No call made, eyes were closed more than open.")

        # Reset the counts and timer after 15 seconds
        open_count = 0
        closed_count = 0
        start_time = time.time()

    # Show the frame with the eye status
    cv2.imshow("Eye State Detection", frame)

    # Check if 'q' is pressed or if the window is closed
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Eye State Detection", cv2.WND_PROP_VISIBLE) < 1:
        breaks

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
