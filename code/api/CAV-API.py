# API for interfacing with CAV functions
# Author: James Crossley

# Imports
from imucontrol.imus import cav_imus
from motorcontrol.Motor import motor
from steeringControl.steeringControl import Steering
from flask import Flask, jsonify, request
import signal

# Initialisations
steering = Steering()
motor.start()
cav_imus.start()

# Signal handling and cleanup
def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    motor.close()
    steering.close()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Flask setup and default route
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "CAV API is running."
    #return "CAV API is running.\n Login with POST request at /login with JSON {'username': 'your_username'} to grab clientID."

# Motor calls
@app.route('/motor_status', methods=['GET'])
def get_motor_status():
    # Returns current motor speed and direction
    status = {
        "motorSpeed": motor.motorSpeed,
        "motorDirection": motor.motorDirection
    }
    return jsonify(status)

@app.route('/motor_speed', methods=['GET', 'POST'])
def set_motor_speed():
    if request.method == 'GET':
        # Returns current motor speed
        return jsonify({"motorSpeed": motor.motorSpeed})
    elif request.method == 'POST':
        # Receives JSON with 'speed' key to set motor speed
        data = request.get_json()
        speed = data.get('speed')
        speed = float(speed)
        try:
            motor.setMotorSpeed(speed)
            return jsonify({"status": "success", "message": f"Motor speed set to {speed}."})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400
    
# Steering calls
@app.route('/steering_angle', methods=['GET', 'POST'])
def steering_angle():
    if request.method == 'GET':
        # Returns current steering angle
        return jsonify({"steeringAngle": steering.currentAngle})
    elif request.method == 'POST':
        # Receives JSON with 'angle' key to set steering angle
        data = request.get_json()
        angle = data.get('angle')
        angle = float(angle)
        try:
            steering.setSteeringAngle(angle)
            return jsonify({"status": "success", "message": f"Steering angle set to {angle}."})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400