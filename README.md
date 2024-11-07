# Patient_Monitoring_System
This project is a comprehensive patient monitoring solution with three main modules: eye.py, mask.py, and motion.py. It uses image processing and alerting to monitor patients in ICU or coma, ensuring their safety through automated detection and notifications. The frontend is built with HTML, CSS, and JavaScript, with the main entry point as index.html .

# Project Structure
- **eye.py**
  Detects when a patient in a coma or ICU opens their eyes. Since these patients are generally expected to keep their eyes closed, any eye-opening triggers an alert. The system sends a notification call via Twilio to notify medical staff.

- **mask.py**
  Monitors whether the patient is wearing their oxygen mask. If the system detects that the mask has been removed, it sends an alert notification to the designated contact, ensuring that staff can respond quickly.

- **motion.py**
  Tracks the patient's activity in their room, identifying states such as walking, standing, sitting, or sleeping. This provides real-time insight into the patient's movements, supporting better supervision and care.

# Frontend
The frontend is designed with HTML, CSS, and JavaScript to create a user-friendly interface, with index.html serving as the main page. It presents data and alerts in an organized and accessible manner, allowing medical staff to quickly assess patient status.

# Setup
- Install Dependencies from project_requirements.txt

# Run the Project
Execute each script as needed to initiate monitoring:

- python3 eye.py
- python3 mask.py
- python3 motion.py

# Usage
- Launch the system and open index.html in a browser to view the monitoring dashboard.
- Each module works independently, monitoring and alerting based on the specific requirements of patient care.
# License
- This project is open-source and available for contributions.

