# Name: imus.py
# Author: James Crossley, 21480395
# Date: 25/04/2025

# Description: This file contains the IMU class which is used to interface with the IMU sensors. It uses the mpu9250_jmdev library to communicate with the sensors.
# The IMU class has methods to read the accelerometer, gyroscope, and magnetometer data. It also has methods to calibrate the sensor.
# This class is a wrapper around the mpu9250_jmdev library and provides a simple interface to the 4 IMU sensors that sit on the CAV's chassis.

import os
import logging
import numpy as np
from mpu9250_jmdev.mpu_9250 import MPU9250
from mpu9250_jmdev.registers import *

# Configure logging
logging.basicConfig(filename="imu.log", level=logging.INFO, format="%(asctime)s - %(message)s")

class CAV_imus:
    imu1 = MPU9250(address_ak=None, 
        address_mpu_master=MPU9050_ADDRESS_68, 
        address_mpu_slave=None, 
        bus=1, 
        gfs=GFS_250, # Max of 250 degrees per second
        afs=AFS_2G, # Max of 2g
        mfs=None, 
        mode=None)

    imu2 = None # This IMU is not used in the current configuration

    imu3 = MPU9250(address_ak=None, 
        address_mpu_master=MPU9050_ADDRESS_69, # IMU3 has its AD0 pin set to high, so its I2C address is 0x69
        address_mpu_slave=None, 
        bus=1, 
        gfs=GFS_250, # Max of 250 degrees per second
        afs=AFS_2G, # Max of 2g
        mfs=None, 
        mode=None)

    imu4 = MPU9250(address_ak=None, 
        address_mpu_master=MPU9050_ADDRESS_69, # IMU4 has its AD0 pin set to high, so its I2C address is 0x69
        address_mpu_slave=None, 
        bus=8, 
        gfs=GFS_250, # Max of 250 degrees per second
        afs=AFS_2G, # Max of 2g
        mfs=None, 
        mode=None)

    imu5 = MPU9250(address_ak=None, 
        address_mpu_master=MPU9050_ADDRESS_68,
        address_mpu_slave=None, 
        bus=8, 
        gfs=GFS_250, # Max of 250 degrees per second
        afs=AFS_2G, # Max of 2g
        mfs=None, 
        mode=None)

    imuList = [imu1, imu2, imu3, imu4, imu5]
    imuAliases = {"Front": imu1, "Back": imu5, "Left": imu3, "Right": imu4}
    imuNoises = []

    def logIMUConfiguration(imu_name, gfs=None, afs=None, abias=None, gbias=None):
        log_message = f"{imu_name} configuration updated:"
        if gfs is not None and afs is not None:
            log_message += f" GFS: {gfs}, AFS: {afs};"
        if abias is not None:
            log_message += f" Accelerometer Bias: {abias};"
        if gbias is not None:
            log_message += f" Gyroscope Bias: {gbias};"
        logging.info(log_message)

    def calibrateAll(numOfSamples):
        # Calibrate all IMUs in the imuList which are not None, averaging the middle 99% of the returned a and g bias values across numOfSamples
        bias_data = []
        CAV_imus.imuNoises = []
        try:
            for imu in CAV_imus.imuList:
                if imu is not None:
                    imu.configureMPU6500(gfs=GFS_250, afs=AFS_2G)
                    aBiasSamples = []
                    gBiasSamples = []
                    for _ in range(numOfSamples):
                        imu.calibrateMPU6500()
                        aBiasSamples.append(imu.abias)
                        gBiasSamples.append(imu.gbias)

                    # Convert to numpy arrays for easier processing
                    aBiasSamples = np.array(aBiasSamples)
                    gBiasSamples = np.array(gBiasSamples)

                    # Calculate the middle 99% range
                    aBiasFiltered = np.percentile(aBiasSamples, [0.5, 99.5], axis=0)
                    gBiasFiltered = np.percentile(gBiasSamples, [0.5, 99.5], axis=0)

                    # Average the middle 99% values
                    avgABias = np.mean(aBiasSamples[(aBiasSamples >= aBiasFiltered[0]) & (aBiasSamples <= aBiasFiltered[1])], axis=0)
                    avgGBias = np.mean(gBiasSamples[(gBiasSamples >= gBiasFiltered[0]) & (gBiasSamples <= gBiasFiltered[1])], axis=0)

                    # Calculate bias noise
                    aBiasSorted = np.sort(aBiasSamples, axis=0)
                    gBiasSorted = np.sort(gBiasSamples, axis=0)
                    aBiasNoise = np.max(np.abs((aBiasSorted[[0, -1]] - avgABias) / avgABias), axis=0).tolist()
                    gBiasNoise = np.max(np.abs((gBiasSorted[[0, -1]] - avgGBias) / avgGBias), axis=0).tolist()

                    imu.abias = avgABias.tolist()
                    imu.gbias = avgGBias.tolist()
                    imu.configureMPU6500(gfs=GFS_250, afs=AFS_2G)
                    CAV_imus.logIMUConfiguration(
                        f"IMU{CAV_imus.imuList.index(imu) + 1}",
                        gfs=GFS_250,
                        afs=AFS_2G,
                        abias=imu.abias,
                        gbias=imu.gbias
                    )
                    bias_data.append((avgABias.tolist(), avgGBias.tolist()))
                    CAV_imus.imuNoises.append((aBiasNoise, gBiasNoise))
                else:
                    # If IMU is None, append zeros
                    bias_data.append(([0, 0, 0], [0, 0, 0]))
                    CAV_imus.imuNoises.append(([0, 0, 0], [0, 0, 0]))

            # Save bias data and noise to imu.conf
            with open("imu.conf", "w") as file:
                for idx, (aBias, gBias) in enumerate(bias_data):
                    file.write(f"IMU{idx + 1} Accelerometer Bias: {aBias}\n")
                    file.write(f"IMU{idx + 1} Gyroscope Bias: {gBias}\n")
                    file.write(f"IMU{idx + 1} A Bias Noise: {CAV_imus.imuNoises[idx][0]}\n")
                    file.write(f"IMU{idx + 1} G Bias Noise: {CAV_imus.imuNoises[idx][1]}\n")
        except Exception as e:
            logging.error(f"Error during calibration: {e}")

    def importSavedBias():
        # Read bias values and noise from imu.conf and apply them to the respective IMUs
        if not os.path.exists("imu.conf"):
            error_message = "Error: imu.conf file not found."
            print(error_message)
            logging.error(error_message)
            return

        try:
            with open("imu.conf", "r") as file:
                lines = file.readlines()

            CAV_imus.imuNoises = []
            for idx, imu in enumerate(CAV_imus.imuList):
                if imu is not None:
                    try:
                        aBiasLine = lines[idx * 4].strip()
                        gBiasLine = lines[idx * 4 + 1].strip()
                        aNoiseLine = lines[idx * 4 + 2].strip()
                        gNoiseLine = lines[idx * 4 + 3].strip()

                        # Parse accelerometer bias
                        if "Accelerometer Bias:" in aBiasLine:
                            aBias = list(map(float, aBiasLine.split(":")[1].strip(" []").split(",")))
                        else:
                            raise ValueError("Invalid accelerometer bias format.")

                        # Parse gyroscope bias
                        if "Gyroscope Bias:" in gBiasLine:
                            gBias = list(map(float, gBiasLine.split(":")[1].strip(" []").split(",")))
                        else:
                            raise ValueError("Invalid gyroscope bias format.")

                        # Parse accelerometer noise
                        if "A Bias Noise:" in aNoiseLine:
                            aNoise = list(map(float, aNoiseLine.split(":")[1].strip(" []").split(",")))
                        else:
                            raise ValueError("Invalid accelerometer noise format.")

                        # Parse gyroscope noise
                        if "G Bias Noise:" in gNoiseLine:
                            gNoise = list(map(float, gNoiseLine.split(":")[1].strip(" []").split(",")))
                        else:
                            raise ValueError("Invalid gyroscope noise format.")

                        # Apply biases and noise to the IMU
                        imu.abias = aBias
                        imu.gbias = gBias
                        imu.configureMPU6500(gfs=GFS_250, afs=AFS_2G)
                        CAV_imus.logIMUConfiguration(
                            f"IMU{idx + 1}",
                            gfs=GFS_250,
                            afs=AFS_2G,
                            abias=imu.abias,
                            gbias=imu.gbias
                        )
                        CAV_imus.imuNoises.append((aNoise, gNoise))
                    except (ValueError, IndexError) as e:
                        error_message = f"Error: Invalid data for IMU{idx + 1}. {e}"
                        print(error_message)
                        logging.error(error_message)
                        CAV_imus.imuNoises.append(([0, 0, 0], [0, 0, 0]))
                else:
                    warning_message = f"Warning: IMU{idx + 1} is not configured. Skipping bias application."
                    print(warning_message)
                    logging.warning(warning_message)
                    CAV_imus.imuNoises.append(([0, 0, 0], [0, 0, 0]))
        except Exception as e:
            error_message = f"Error: Failed to read or parse imu.conf. {e}"
            print(error_message)
            logging.error(error_message)

