# Author: James Crossley, 21480395
# Description: A simple class structure to hold motor data and provide a simple interface to access it.

import serial
import logging

class motor:
    # Class attributes
    port = "/dev/ttyACM0"
    baud = 115200
    timeout = 1
    ser = None  # Serial object to communicate with onboard arduino

    # Initialise logging
    logging.basicConfig(filename="~/logs/motor.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Motor class initialized.")

    def setMotorSpeed(speed: int):
        # Set the motor speed. Speed should be between -100 and 100.
        if -100 <= speed <= 100:
            command = f'S{speed}'
            logging.info(f"Setting motor speed to {speed}. Command: {command}")
            motor.ser.write(command.encode())
        else:
            logging.error("Invalid speed value of {speed}. Must be between -100 and 100.")

    def motorStop():
        # Stop the motor
        command = 'S0'
        logging.info("Stopping motor. Command: S0")
        motor.ser.write(command.encode())

    def start():
        # Initialize the serial connection
        try:
            motor.ser = serial.Serial(motor.port, motor.baud, timeout=motor.timeout)
            logging.info(f"Serial port {motor.port} opened at {motor.baud} baud.")
        except serial.SerialException as e:
            logging.error(f"Error opening serial port: {e}")
            return False
        return True
    
    def close():
        # Close the serial connection
        if motor.ser and motor.ser.is_open:
            motor.ser.close()
            logging.info("Serial port closed.")
        else:
            logging.warning("Serial port was not open or already closed.")  