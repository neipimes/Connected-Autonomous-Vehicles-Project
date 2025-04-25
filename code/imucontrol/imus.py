# Name: imus.py
# Author: James Crossley, 21480395
# Date: 25/04/2025

# Description: This file contains the IMU class which is used to interface with the IMU sensors. It uses the mpu9250_jmdev library to communicate with the sensors.
# The IMU class has methods to read the accelerometer, gyroscope, and magnetometer data. It also has methods to calibrate the sensor and to get the sensor's temperature.
# This class is a wrapper around the mpu9250_jmdev library and provides a simple interface to the 4 IMU sensors that sit on the CAV's chassis.

from mpu9250_jmdev.mpu_9250 import MPU9250
from mpu9250_jmdev.registers import *

# TODO: Add a method for configuring setup with a config file.

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

    def calibrateAll(self, numOfSamples):
        # Calibrate all IMUs in the imuList which are not None, averaging the returned a and g bias values across numOfSamples
        # This is a blocking function, so it will wait for the calibration to finish before returning (approx 1 second per calibration run * number of IMUs)
        for imu in CAV_imus.imuList:
            if imu is not None:
                imu.configureMPU6500(gfs=GFS_250, afs=AFS_2G)
                avgABias = [0, 0, 0]
                avgGBias = [0, 0, 0]
                for i in range(0, numOfSamples):
                    imu.calibrateMPU6500()
                    aBias = imu.abias
                    gBias = imu.gbias
                    for i in range(0, 3):
                        avgABias[i] += aBias[i]
                        avgGBias[i] += gBias[i]
                for i in range(0, 3):
                    avgABias[i] /= numOfSamples
                    avgGBias[i] /= numOfSamples