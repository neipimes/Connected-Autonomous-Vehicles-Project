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

# Configure logging
logging.basicConfig(filename="~/logs/imu.log", level=logging.INFO, format="%(asctime)s - %(message)s")

class CAV_imus:
    mpu1 = MPU9250(address_ak=None, 
        address_mpu_master=MPU9050_ADDRESS_68, 
        address_mpu_slave=None, 
        bus=1, 
        gfs=GFS_250, # Max of 250 degrees per second
        afs=AFS_2G, # Max of 2g
        mfs=None, 
        mode=None)

    mpu2 = None # This IMU is not used in the current configuration

    mpu3 = MPU9250(address_ak=None, 
        address_mpu_master=MPU9050_ADDRESS_69, # IMU3 has its AD0 pin set to high, so its I2C address is 0x69
        address_mpu_slave=None, 
        bus=1, 
        gfs=GFS_250, # Max of 250 degrees per second
        afs=AFS_2G, # Max of 2g
        mfs=None, 
        mode=None)

    mpu4 = MPU9250(address_ak=None, 
        address_mpu_master=MPU9050_ADDRESS_69, # IMU4 has its AD0 pin set to high, so its I2C address is 0x69
        address_mpu_slave=None, 
        bus=8, 
        gfs=GFS_250, # Max of 250 degrees per second
        afs=AFS_2G, # Max of 2g
        mfs=None, 
        mode=None)

    mpu5 = MPU9250(address_ak=None, 
        address_mpu_master=MPU9050_ADDRESS_68,
        address_mpu_slave=None, 
        bus=8, 
        gfs=GFS_250, # Max of 250 degrees per second
        afs=AFS_2G, # Max of 2g
        mfs=None, 
        mode=None)
    
    # Forward and Left: Positive values. Back and Right: Negative values.
    imu1 = imu("Front", mpu1, [0, 0, 0], [0, 0, 0], 0, 1, -1, -1) # Front IMU
    imu2 = None # This IMU is not used in the current configuration
    imu3 = imu("Left", mpu3, [0, 0, 0], [0, 0, 0], 1, 0, 1, -1) # Left IMU
    imu4 = imu("Right", mpu4, [0, 0, 0], [0, 0, 0], 1, 0, -1, 1) # Right IMU
    imu5 = imu("Back", mpu5, [0, 0, 0], [0, 0, 0], 0, 1, 1, 1) # Back IMU

    imuList = [imu1, imu3, imu4, imu5]
    imuAliases = {"Front": imu1, "Back": imu5, "Left": imu3, "Right": imu4}

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
        try:
            for imu_obj in CAV_imus.imuList:
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

                    CAV_imus.logIMUConfiguration(
                        f"IMU{CAV_imus.imuList.index(imu_obj) + 1}",
                        gfs=GFS_250,
                        afs=AFS_2G,
                        abias=imu_obj.mpu.abias,
                        gbias=imu_obj.mpu.gbias
                    )

            # Save bias data and noise to imu.conf
            CAV_imus.saveConfig()
        except Exception as e:
            logging.error(f"Error during calibration: {e}")

    def saveConfig():
        # Save IMU configuration, biases, noise values, and aliases to imu.conf
        try:
            with open("imu.conf", "w") as file:
                for idx, imu_obj in enumerate(CAV_imus.imuList):
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
                for alias, imu_obj in CAV_imus.imuAliases.items():
                    file.write(f"{alias}: IMU{CAV_imus.imuList.index(imu_obj) + 1}\n")
            logging.info("IMU configuration and aliases saved to imu.conf.")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")

    def importSavedConfig():
        # Import IMU configuration, biases, noise values, and aliases from imu.conf
        if not os.path.exists("imu.conf"):
            error_message = "Error: imu.conf file not found."
            print(error_message)
            logging.error(error_message)
            return
        
        try:
            with open("imu.conf", "r") as file:
                lines = file.readlines()

            CAV_imus.imuAliases = {}
            for idx, imu_obj in enumerate(CAV_imus.imuList):
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

                        CAV_imus.logIMUConfiguration(
                            f"IMU{idx + 1}",
                            gfs=GFS_250,
                            afs=AFS_2G,
                            abias=imu_obj.mpu.abias,
                            gbias=imu_obj.mpu.gbias
                        )
                    except (ValueError, IndexError) as e:
                        error_message = f"Error: Invalid data for IMU{idx + 1}. {e}"
                        print(error_message)
                        logging.error(error_message)

            # Parse imuAliases
            aliasStartIndex = len(CAV_imus.imuList) * 9
            for line in lines[aliasStartIndex:]:
                if "IMU Aliases:" in line:
                    continue
                alias, imuRef = line.strip().split(":")
                alias = alias.strip()
                imuIndex = int(imuRef.strip().split("IMU")[1]) - 1
                CAV_imus.imuAliases[alias] = CAV_imus.imuList[imuIndex] # TODO: Removing imu2 from imuList causes a fuck up here where imu5 is not added to the aliases.
                # E.g. IMU5 is in index 3, not index 4 as it would be if index 1 didn't contain IMU3.
                # TODO: Could have the second entry be None in the base global imuList, and then drop the None values when enumerating in above loop.
                # This means we could still have working alias matching here.
            logging.info("IMU configuration and aliases imported from imu.conf.")

        except Exception as e:
            error_message = f"Error: Failed to read or parse imu.conf. {e}"
            print(error_message)
            logging.error(error_message)

    def getAvgData():
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
        for imu_obj in CAV_imus.imuList:
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

    def start():
        # Setup function to be run to initialise the IMUs.
        # Currently will only run the importSavedConfig function to load the bias values from the imu.conf file.
        CAV_imus.importSavedConfig()