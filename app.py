from flask import Flask, jsonify, request
from flask_cors import CORS  # Add this import
import subprocess

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route('/')
def home():
    return "Welcome to the Final Project API!"

@app.route('/eye', methods=['GET'])
def run_eye_detection():
    try:
        result = subprocess.run(['python', 'eye.py'], capture_output=True, text=True, check=True)
        return jsonify(result=result.stdout.strip())
    except subprocess.CalledProcessError as e:
        return jsonify(error=f"Script failed with exit code {e.returncode}: {e.output.strip()}"), 500
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/mask', methods=['GET'])
def run_mask_detection():
    try:
        result = subprocess.run(['python', 'mask.py'], capture_output=True, text=True)
        return jsonify(result=result.stdout.strip())
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/motion', methods=['POST'])
def store_motion_data():
    data = request.get_json()
    return jsonify(message="Motion data received.", data=data)

if __name__ == '__main__':
    app.run(debug=True, port=5007)  # Ensure you're running on the desired port
