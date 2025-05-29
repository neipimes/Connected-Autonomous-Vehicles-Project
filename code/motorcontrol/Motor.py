# Author: James Crossley, 21480395
# Description: A simple class structure to hold motor data and provide a simple interface to access it.

import serial
import logging
import os, time

class motor:
    # Class attributes
    port = "/dev/ttyACM0"
    baud = 115200
    timeout = 1
    ser = None  # Serial object to communicate with onboard arduino

    # Initialise logging
    logging.basicConfig(filename=os.path.expanduser("~/logs/motor.log"), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Motor class initialized.")

    def setMotorSpeed(speed: int):
        # Set the motor speed. Speed should be between -100 and 100.
        if -100 <= speed <= 100:
            motor.ser.flush()
            command = f'S{speed}\n'
            logging.info(f"Setting motor speed to {speed}. Command: {command}")
            # Before we send a new speed command, we need to stop the motor first.
            motor.motorStop()
            # Wait for a short period to ensure the motor stops before sending the new speed command.
            time.sleep(0.2)
            for i in range(3):
                # Send the command to the motor 3 times to ensure it is received.
                # There are constant inconsistencies with the serial connection, so this is a workaround.
                if motor.ser.is_open:
                    logging.info(f"Sending command to motor: {command.strip()}")
                    motor.ser.write(command.encode())
                    motor.ser.flush()
                else:
                    logging.error("Serial port is not open. Cannot send command.")
                    return
                time.sleep(0.1)
        else:
            logging.error(f"Invalid speed value of {speed}. Must be between -100 and 100.")

    def motorStop():
        # Stop the motor
        command = 'S0\n'
        logging.info("Stopping motor. Command: S0")
        motor.ser.write(command.encode())
        motor.ser.flush()

    def importConfig():
        # Import the motor configuration from a file located in ~/configs/motor.conf 
        try:
            with open(os.path.expanduser("~/configs/motor.conf"), "r") as config_file:
                for line in config_file:
                    if line.startswith("port="):
                        motor.port = line.split("=")[1].strip()
                    elif line.startswith("baud="):
                        motor.baud = int(line.split("=")[1].strip())
                    elif line.startswith("timeout="):
                        motor.timeout = int(line.split("=")[1].strip())
            logging.info("Motor configuration imported successfully.")
            return True
        except Exception as e:
            logging.error(f"Error importing motor configuration: {e}")
            return False
        
    def saveConfig():
        # Save the motor configuration to a file located in ~/configs/motor.conf 
        try:
            with open(os.path.expanduser("~/configs/motor.conf"), "w") as config_file:
                config_file.write(f"port={motor.port}\nbaud={motor.baud}\ntimeout={motor.timeout}")
                logging.info("Motor configuration saved successfully.")
                return True
        except Exception as e:
            logging.error(f"Error saving motor configuration: {e}")
            return False

    def start():
        # Initialize the serial connection
        try:
            motor.ser = serial.Serial(motor.port, motor.baud, timeout=motor.timeout)
            time.sleep(8) # Wait for the serial connection to fully initialise.
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
            return True
        else:
            logging.warning("Serial port was not open or already closed.")
            return False