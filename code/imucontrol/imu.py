# Name: imu.py
# Author: James Crossley, 21480395
# Description: A simple class structure to hold IMU data and provide a simple interface to access it.

from mpu9250_jmdev.mpu_9250 import MPU9250
from mpu9250_jmdev.registers import *

class imu:
    def __init__(self, pos: str, mpu: MPU9250, aNoiseVals: list, gNoiseVals: list, fbIndex: int, lrIndex: int, fbMod: int, lrMod: int):
        # Validate fbMod and lrMod
        if fbMod not in [-1, 1]:
            raise ValueError(f"Invalid fbMod value: {fbMod}. Must be 1 or -1.")
        if lrMod not in [-1, 1]:
            raise ValueError(f"Invalid lrMod value: {lrMod}. Must be 1 or -1.")

        # Validate fbIndex and lrIndex
        if fbIndex not in [0, 1, 2]:
            raise ValueError(f"Invalid fbIndex value: {fbIndex}. Must be 0, 1, or 2.")
        if lrIndex not in [0, 1, 2]:
            raise ValueError(f"Invalid lrIndex value: {lrIndex}. Must be 0, 1, or 2.")

        self.pos = pos
        self.mpu = mpu
        self.aNoiseVals = aNoiseVals
        self.gNoiseVals = gNoiseVals
        self.fbIndex = fbIndex
        self.lrIndex = lrIndex
        self.fbMod = fbMod
        self.lrMod = lrMod

    def getAccelData(self):
        return self.mpu.readAccelerometerMaster()
    
    def getGyroData(self):
        return self.mpu.readGyroscopeMaster()
    
    def getFBAccelData(self):
        return (self.getAccelData()[self.fbIndex] * self.fbMod)
    
    def getLRAccelData(self):
        return (self.getAccelData()[self.lrIndex] * self.lrMod)
    
    def getFBAcellNoise(self):
        return self.aNoiseVals[self.fbIndex]
    
    def getLRAcellNoise(self):
        return self.aNoiseVals[self.lrIndex]
    
    def getTurnAngle(self):
        return self.mpu.readGyroscopeMaster()[2]
    
    def getTurnAngleNoise(self):
        return self.gNoiseVals[2]