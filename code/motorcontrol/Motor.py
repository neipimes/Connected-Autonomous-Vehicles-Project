# Author: James Crossley, 21480395
# Description: A simple class structure to hold motor data and provide a simple interface to access it.

import serial
import logging
import os, time

class Motor:
    _instance = None
    _logger = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Motor, cls).__new__(cls)
            # Set up logger for this instance
            cls._logger = logging.getLogger("MotorLogger")
            if not cls._logger.hasHandlers():
                log_path = os.path.expanduser("~/logs/motor.log")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                handler = logging.FileHandler(log_path)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                cls._logger.addHandler(handler)
                cls._logger.setLevel(logging.INFO)
            cls._logger.info("Motor singleton instance created.")
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.port = "/dev/ttyACM0"
            self.baud = 115200
            self.timeout = 1
            self.ser = None  # Serial object to communicate with onboard arduino
            self.initialized = True

    def setMotorSpeed(self, speed: float):
        # Set the motor speed. Speed should be between -100 and 100.
        if -100 <= speed <= 100:
            # Check the speed value is a max of 2 decimal places.
            if isinstance(speed, float):
                speed = round(speed, 2)
            elif isinstance(speed, int):
                speed = float(speed)
            else:
                self._logger.error(f"Invalid speed type: {type(speed)}. Must be float or int. Max 2 decimal places.")
                return

            #speedModified = int(speed * 100) # Convert speed to a value between -10000 and 10000
            if self.ser:
                self.ser.flush()
            #command = f'S{speedModified}\n' Old command for finer control, but controller doesn't support it currently.
            command = f'S{int(speed)}\n'
            self._logger.info(f"Setting motor speed to {speed}. Command: {command}")
            # Before we send a new speed command, we need to stop the motor first.
            self.motorStop()
            # Wait for a short period to ensure the motor stops before sending the new speed command.
            #time.sleep(0.1)
            if self.ser and self.ser.is_open:
                self._logger.info(f"Sending command to motor: {command.strip()}")
                self.ser.write(command.encode())
                self.ser.flush()
    
                if speed < 0: # Changing the motor direction to reverse has some technicalities, so we need to handle it separately.
                    # The initial command changes the state in the ESC, with the second command sending the speed.
                    self._logger.info(f"Motor set to reverse, sending speed command {command.strip()} after state change.")
                    time.sleep(0.5)  
                    self.ser.write(command.encode())
                    self.ser.flush()
            
            else:
                self._logger.error("Serial port is not open. Cannot send command.")
        else:
            self._logger.error(f"Invalid speed value of {speed}. Must be between -100 and 100.")

    def motorStop(self):
        # Stop the motor
        command = 'S0\n'
        self._logger.info("Stopping motor. Command: S0")
        if self.ser:
            self.ser.write(command.encode())
            self.ser.flush()

    def importConfig(self):
        # Import the motor configuration from a file located in ~/configs/motor.conf 
        try:
            with open(os.path.expanduser("~/configs/motor.conf"), "r") as config_file:
                for line in config_file:
                    if line.startswith("port="):
                        self.port = line.split("=")[1].strip()
                    elif line.startswith("baud="):
                        self.baud = int(line.split("=")[1].strip())
                    elif line.startswith("timeout="):
                        self.timeout = int(line.split("=")[1].strip())
            self._logger.info("Motor configuration imported successfully.")
            return True
        except Exception as e:
            self._logger.error(f"Error importing motor configuration: {e}")
            return False
        
    def saveConfig(self):
        # Save the motor configuration to a file located in ~/configs/motor.conf 
        try:
            with open(os.path.expanduser("~/configs/motor.conf"), "w") as config_file:
                config_file.write(f"port={self.port}\nbaud={self.baud}\ntimeout={self.timeout}")
                self._logger.info("Motor configuration saved successfully.")
                return True
        except Exception as e:
            self._logger.error(f"Error saving motor configuration: {e}")
            return False

    def start(self):
        # Initialize the serial connection
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            time.sleep(8) # Wait for the serial connection to fully initialise.
            self._logger.info(f"Serial port {self.port} opened at {self.baud} baud.")
        except serial.SerialException as e:
            if self.ser and self.ser.is_open: # If the serial port is already open, we can assume the motor is already connected.
                self._logger.info("Serial port already open. Connected to motor.")
                return True
            else:
                self._logger.error(f"Error opening serial port: {e}")
                return False
        return True
    
    def close(self):
        # Close the serial connection
        if self.ser and self.ser.is_open:
            self.ser.close()
            self._logger.info("Serial port closed.")
            return True
        else:
            self._logger.warning("Serial port was not open or already closed.")
            return False

# Singleton instance for external use
motor = Motor()