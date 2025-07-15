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
from imucontrol.imu import imu

class CAV_imus:
    _instance = None
    _logger = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CAV_imus, cls).__new__(cls)
            # Set up logger for this instance
            cls._logger = logging.getLogger("CAV_imusLogger")
            if not cls._logger.hasHandlers():
                log_path = os.path.expanduser("~/logs/imu.log")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                handler = logging.FileHandler(log_path)
                formatter = logging.Formatter('%(asctime)s - %(message)s')
                handler.setFormatter(formatter)
                cls._logger.addHandler(handler)
                cls._logger.setLevel(logging.INFO)
            cls._logger.info("CAV_imus singleton instance created.")
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.mpu1 = MPU9250(address_ak=None, 
                address_mpu_master=MPU9050_ADDRESS_68, 
                address_mpu_slave=None, 
                bus=1, 
                gfs=GFS_250, # Max of 250 degrees per second
                afs=AFS_2G, # Max of 2g
                mfs=None, 
                mode=None)

            self.mpu2 = None # This IMU is not used in the current configuration

            self.mpu3 = MPU9250(address_ak=None, 
                address_mpu_master=MPU9050_ADDRESS_69, # IMU3 has its AD0 pin set to high, so its I2C address is 0x69
                address_mpu_slave=None, 
                bus=1, 
                gfs=GFS_250, # Max of 250 degrees per second
                afs=AFS_2G, # Max of 2g
                mfs=None, 
                mode=None)

            self.mpu4 = MPU9250(address_ak=None, 
                address_mpu_master=MPU9050_ADDRESS_69, # IMU4 has its AD0 pin set to high, so its I2C address is 0x69
                address_mpu_slave=None, 
                bus=8, 
                gfs=GFS_250, # Max of 250 degrees per second
                afs=AFS_2G, # Max of 2g
                mfs=None, 
                mode=None)

            self.mpu5 = MPU9250(address_ak=None, 
                address_mpu_master=MPU9050_ADDRESS_68,
                address_mpu_slave=None, 
                bus=8, 
                gfs=GFS_250, # Max of 250 degrees per second
                afs=AFS_2G, # Max of 2g
                mfs=None, 
                mode=None)
            
            # Forward and Left: Positive values. Back and Right: Negative values.
            self.imu1 = imu("Front", self.mpu1, [0, 0, 0], [0, 0, 0], 0, 1, -1, -1) # Front IMU
            self.imu2 = None # This IMU is not used in the current configuration
            self.imu3 = imu("Left", self.mpu3, [0, 0, 0], [0, 0, 0], 1, 0, 1, -1) # Left IMU
            self.imu4 = imu("Right", self.mpu4, [0, 0, 0], [0, 0, 0], 1, 0, -1, 1) # Right IMU
            self.imu5 = imu("Back", self.mpu5, [0, 0, 0], [0, 0, 0], 0, 1, 1, 1) # Back IMU

            self.imuList = [self.imu1, self.imu3, self.imu4, self.imu5]
            self.imuAliases = {"Front": self.imu1, "Back": self.imu5, "Left": self.imu3, "Right": self.imu4}

            # Track the IMUs that have the lowest acceleration noise values for each axis
            # This is used to determine which IMU to use for each axis when calculating the average data.
            self.imuFB = None
            self.imuLR = None

            self.initialized = True

    def logIMUConfiguration(self, imu_name, gfs=None, afs=None, abias=None, gbias=None):
        log_message = f"{imu_name} configuration updated:"
        if gfs is not None and afs is not None:
            log_message += f" GFS: {gfs}, AFS: {afs};"
        if abias is not None:
            log_message += f" Accelerometer Bias: {abias};"
        if gbias is not None:
            log_message += f" Gyroscope Bias: {gbias};"
        self._logger.info(log_message)

    def updateLowestNoiseIMUs(self):
        # Determine and assign the IMUs with the lowest acceleration noise values for FB and LR axes
        try:
            lowestFBNoise = float('inf')
            lowestLRNoise = float('inf')
            self.imuFB = None
            self.imuLR = None

            for imu_obj in self.imuList:
                if imu_obj is not None:
                    if imu_obj.getFBAcellNoise() < lowestFBNoise:  # Assuming index 0 corresponds to FB axis
                        lowestFBNoise = imu_obj.getFBAcellNoise()
                        self.imuFB = imu_obj
                    if imu_obj.getLRAcellNoise() < lowestLRNoise:  # Assuming index 1 corresponds to LR axis
                        lowestLRNoise = imu_obj.getLRAcellNoise()
                        self.imuLR = imu_obj

            self._logger.info(f"IMU with lowest FB noise: {self.imuFB.pos if self.imuFB else 'None'}")
            self._logger.info(f"IMU with lowest LR noise: {self.imuLR.pos if self.imuLR else 'None'}")
        except Exception as e:
            self._logger.error(f"Error updating lowest noise IMUs: {e}")

    def calibrateAll(self, numOfSamples):
        # Calibrate all IMUs in the imuList which are not None, averaging the middle 95% of the returned a and g bias values across numOfSamples
        try:
            for imu_obj in self.imuList:
                if imu_obj is not None:
                    imu_obj.mpu.configureMPU6500(gfs=GFS_250, afs=AFS_2G)
                    aBiasSamples = []
                    gBiasSamples = []
                    for _ in range(numOfSamples):
                        imu_obj.mpu.calibrateMPU6500()
                        aBiasSamples.append(imu_obj.mpu.abias)
                        gBiasSamples.append(imu_obj.mpu.gbias)

                    # Convert to numpy arrays for easier processing
                    aBiasSamples = np.array(aBiasSamples)
                    gBiasSamples = np.array(gBiasSamples)

                    # Calculate the middle 95% range for each axis
                    avgABias = []
                    avgGBias = []
                    aBiasNoise = []
                    gBiasNoise = []

                    for axis in range(3):  # Iterate over x, y, z axes
                        aBiasFiltered = np.percentile(aBiasSamples[:, axis], [0.5, 95.5])
                        gBiasFiltered = np.percentile(gBiasSamples[:, axis], [0.5, 95.5])

                        # Average the middle 95% values for this axis
                        avgABias.append(np.mean(aBiasSamples[(aBiasSamples[:, axis] >= aBiasFiltered[0]) & 
                                                              (aBiasSamples[:, axis] <= aBiasFiltered[1]), axis]))
                        avgGBias.append(np.mean(gBiasSamples[(gBiasSamples[:, axis] >= gBiasFiltered[0]) & 
                                                              (gBiasSamples[:, axis] <= gBiasFiltered[1]), axis]))

                        # Calculate bias noise for this axis
                        aBiasSorted = np.sort(aBiasSamples[:, axis])
                        gBiasSorted = np.sort(gBiasSamples[:, axis])
                        aBiasNoise.append(np.max(np.abs((aBiasSorted[[0, -1]] - avgABias[-1]) / avgABias[-1])))
                        gBiasNoise.append(np.max(np.abs((gBiasSorted[[0, -1]] - avgGBias[-1]) / avgGBias[-1])))

                    # Update MPU9250 biases
                    imu_obj.mpu.abias = avgABias
                    imu_obj.mpu.gbias = avgGBias
                    imu_obj.mpu.configureMPU6500(gfs=GFS_250, afs=AFS_2G)

                    # Update imu object noise values
                    imu_obj.aNoiseVals = aBiasNoise
                    imu_obj.gNoiseVals = gBiasNoise

                    self.logIMUConfiguration(
                        f"IMU{self.imuList.index(imu_obj) + 1}",
                        gfs=GFS_250,
                        afs=AFS_2G,
                        abias=imu_obj.mpu.abias,
                        gbias=imu_obj.mpu.gbias
                    )

            # Update the IMUs with the lowest noise values
            self.updateLowestNoiseIMUs()

            # Save bias data and noise to imu.conf
            self.saveConfig()
        except Exception as e:
            self._logger.error(f"Error during calibration: {e}")

    def saveConfig(self):
        # Save IMU configuration, biases, noise values, and aliases to ~/configs/imu.conf
        try:
            config_path = os.path.expanduser("~/configs/imu.conf")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as file:
                for idx, imu_obj in enumerate(self.imuList):
                    if imu_obj is not None:
                        # Save IMU creation information
                        file.write(f"IMU{idx + 1} Position: {imu_obj.pos}\n")
                        file.write(f"IMU{idx + 1} FB Index: {imu_obj.fbIndex}\n")
                        file.write(f"IMU{idx + 1} LR Index: {imu_obj.lrIndex}\n")
                        file.write(f"IMU{idx + 1} FB Mod: {imu_obj.fbMod}\n")
                        file.write(f"IMU{idx + 1} LR Mod: {imu_obj.lrMod}\n")
                        # Save bias and noise values
                        file.write(f"IMU{idx + 1} Accelerometer Bias: {imu_obj.mpu.abias}\n")
                        file.write(f"IMU{idx + 1} Gyroscope Bias: {imu_obj.mpu.gbias}\n")
                        file.write(f"IMU{idx + 1} A Bias Noise: {imu_obj.aNoiseVals}\n")
                        file.write(f"IMU{idx + 1} G Bias Noise: {imu_obj.gNoiseVals}\n")
                # Save imuAliases
                file.write("IMU Aliases:\n")
                for alias, imu_obj in self.imuAliases.items():
                    file.write(f"{alias}: IMU{self.imuList.index(imu_obj) + 1}\n")
            self._logger.info("IMU configuration and aliases saved to ~/configs/imu.conf.")
        except Exception as e:
            self._logger.error(f"Error saving configuration: {e}")

    def importSavedConfig(self):
        # Import IMU configuration, biases, noise values, and aliases from ~/configs/imu.conf
        config_path = os.path.expanduser("~/configs/imu.conf")
        if not os.path.exists(config_path):
            error_message = "Error: ~/configs/imu.conf file not found."
            print(error_message)
            self._logger.error(error_message)
            return
        
        try:
            with open(config_path, "r") as file:
                lines = file.readlines()

            self.imuAliases = {}
            for idx, imu_obj in enumerate(self.imuList):
                if imu_obj is not None:
                    try:
                        # Parse IMU creation information
                        posLine = lines[idx * 9].strip()
                        fbIndexLine = lines[idx * 9 + 1].strip()
                        lrIndexLine = lines[idx * 9 + 2].strip()
                        fbModLine = lines[idx * 9 + 3].strip()
                        lrModLine = lines[idx * 9 + 4].strip()
                        aBiasLine = lines[idx * 9 + 5].strip()
                        gBiasLine = lines[idx * 9 + 6].strip()
                        aNoiseLine = lines[idx * 9 + 7].strip()
                        gNoiseLine = lines[idx * 9 + 8].strip()

                        # Extract position
                        if "Position:" in posLine:
                            imu_obj.pos = posLine.split(":")[1].strip()
                        else:
                            raise ValueError("Invalid position format.")

                        # Extract FB and LR indices
                        if "FB Index:" in fbIndexLine:
                            imu_obj.fbIndex = int(fbIndexLine.split(":")[1].strip())
                        else:
                            raise ValueError("Invalid FB index format.")

                        if "LR Index:" in lrIndexLine:
                            imu_obj.lrIndex = int(lrIndexLine.split(":")[1].strip())
                        else:
                            raise ValueError("Invalid LR index format.")

                        # Extract FB and LR modifiers
                        if "FB Mod:" in fbModLine:
                            imu_obj.fbMod = int(fbModLine.split(":")[1].strip())
                        else:
                            raise ValueError("Invalid FB modifier format.")

                        if "LR Mod:" in lrModLine:
                            imu_obj.lrMod = int(lrModLine.split(":")[1].strip())
                        else:
                            raise ValueError("Invalid LR modifier format.")

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
                        
                        # Apply biases to MPU9250
                        imu_obj.mpu.abias = aBias
                        imu_obj.mpu.gbias = gBias
                        imu_obj.mpu.configureMPU6500(gfs=GFS_250, afs=AFS_2G)

                        # Apply noise values to imu object
                        imu_obj.aNoiseVals = aNoise
                        imu_obj.gNoiseVals = gNoise

                        self.logIMUConfiguration(
                            f"IMU{idx + 1}",
                            gfs=GFS_250,
                            afs=AFS_2G,
                            abias=imu_obj.mpu.abias,
                            gbias=imu_obj.mpu.gbias
                        )
                    except (ValueError, IndexError) as e:
                        error_message = f"Error: Invalid data for IMU{idx + 1}. {e}"
                        print(error_message)
                        self._logger.error(error_message)

            # Update the IMUs with the lowest noise values
            self.updateLowestNoiseIMUs()

            # Parse imuAliases
            aliasStartIndex = len(self.imuList) * 9
            for line in lines[aliasStartIndex:]:
                if "IMU Aliases:" in line:
                    continue
                alias, imuRef = line.strip().split(":")
                alias = alias.strip()
                imuIndex = int(imuRef.strip().split("IMU")[1]) - 1
                self.imuAliases[alias] = self.imuList[imuIndex]
            self._logger.info("IMU configuration and aliases imported from ~/configs/imu.conf.")

        except Exception as e:
            error_message = f"Error: Failed to read or parse ~/configs/imu.conf. {e}"
            print(error_message)
            self._logger.error(error_message)

    def getAvgData(self):
        # Iterate over the IMUs, get their measurements for a direction, modify according to layout,
        # and return the average of the measurements for that direction balanced by thier respective noise values.

        # FB, LR and Turn Angle Values
        allAccFBData = []
        allAccLRData = []
        allAccFBNoise = []
        allAccLRNoise = []
        allTurnAngleData = []
        allTurnAngleNoise = []

        # TODO: May have to restructure this to grab the data for each axis in the same loop to avoid 
        #       measurements being read out of sync from each other.
        #       This means the individual axes will be slightly more out of sync, but individual axis
        #       accuracy will increase.

        # TODO: Averaging across all IMUs is not perfect, as the measurement of the most reliable IMU should be enough to determine an accurate values.
        #       This is different for the turn angle, where averaging does seem to make the result more accurate (Noise is more random).
        #       For acceleration, using the most reliable (or top 2) IMU(s) should be enough to determine an accurate value. Could be worse over longer periods of time however.
        
        for imu_obj in self.imuList:
            if imu_obj is not None:
                allAccFBData.append(imu_obj.getFBAccelData())
                allAccFBNoise.append(imu_obj.getFBAcellNoise())
                allAccLRData.append(imu_obj.getLRAccelData())
                allAccLRNoise.append(imu_obj.getLRAcellNoise())
                allTurnAngleData.append(imu_obj.getTurnAngle())
                allTurnAngleNoise.append(imu_obj.getTurnAngleNoise())
        
        for i in range(len(allAccFBNoise)):
            # Invert the noise values to be a multiplier for the data value, then
            # raise to a power to increase the effect of the noise value on the data value.
            allAccFBNoise[i] = (1 - allAccFBNoise[i]) ** 20 
            allAccLRNoise[i] = (1 - allAccLRNoise[i]) ** 20
            allTurnAngleNoise[i] = (1 - allTurnAngleNoise[i]) ** 20 
        
        weightsFB = []
        weightsLR = []
        weightsTA = []
        for i in range(len(allAccFBNoise)):
            # Divide the noise value by the sum of the noise values to get a weighted multiplier for the data value.
            weightsFB.append(allAccFBNoise[i] / sum(allAccFBNoise))
            weightsLR.append(allAccLRNoise[i] / sum(allAccLRNoise))
            weightsTA.append(allTurnAngleNoise[i] / sum(allTurnAngleNoise))

        avgFB = None
        avgLR = None
        avgTA = None
        for i in range(len(allAccFBData)):
            # Multiply the data value by the weighted noise value to get a weighted data value.
            if avgFB is None:
                avgFB = allAccFBData[i] * weightsFB[i]
            else:
                avgFB = avgFB + allAccFBData[i] * weightsFB[i]
            
            if avgLR is None:
                avgLR = allAccLRData[i] * weightsLR[i]
            else:
                avgLR = avgLR + allAccLRData[i] * weightsLR[i]

            if avgTA is None:
                avgTA = allTurnAngleData[i] * weightsTA[i]
            else:
                avgTA = avgTA + allTurnAngleData[i] * weightsTA[i]
                

        return (avgFB, avgLR, avgTA) # Return the average values for FB, LR and Turn Angle.

    def start(self):
        # Setup function to be run to initialise the IMUs.
        
        config_path = os.path.expanduser("~/configs/imu.conf")
        if not os.path.exists(config_path):
            # Run an initial calibration and save to ~/configs/imu.conf
            self._logger.info("imu.conf not found. Running initial calibration and saving configuration.")
            self.calibrateAll(50)

        self.importSavedConfig()

# Singleton instance for external use
cav_imus = CAV_imus()