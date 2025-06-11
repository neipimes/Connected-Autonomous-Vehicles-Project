import numpy as np

class Particle:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle

    def calcEstLidarMeasurements(self, oldLidarMeasurements: np.ndarray):
        """
        Calculate the expected lidar measurements based on the particle's position and angle.
        This method should transform the lidar measurements according to the particle's pose.
        Lidar measurements are a Numpy array of distance and angle pairs.
        Assuming oldLidarMeasurements is a Nx2 array where each row is [distance, angle]
        Transform lidar measurements based on the particle's position (x, y) and angle. The new
        measurements will be in the same format as oldLidarMeasurements.
        """
        if not isinstance(oldLidarMeasurements, np.ndarray) or oldLidarMeasurements.ndim != 2 or oldLidarMeasurements.shape[1] != 2:
            raise ValueError("oldLidarMeasurements must be a Nx2 numpy array.")
        
        transformed_measurements = np.zeros_like(oldLidarMeasurements)
        for i in range(oldLidarMeasurements.shape[0]):
            distance, angle = oldLidarMeasurements[i]
            newAngle = angle + self.angle
            newDistance = 

        return transformed_measurements

    def calcCost(self, newLidarMeasurements: np.ndarray):
        """
        Calculate the cost of the particle given the lidar measurements.
        This method should compare the expected lidar measurements with the actual ones.
        """
        expected_measurements = self.calcEstLidarMeasurements(newLidarMeasurements)
        # Compare expected_measurements with actual newLidarMeasurements
        # Calculate and return the cost
        pass