import numpy as np
import math, copy

class Particle:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.cost = float('inf')  # Initialize cost to infinity

    def calcEstLidarMeasurements(self, oldLidarScan: np.ndarray):
        """
        Calculate the expected lidar measurements based on the particle's position and angle.
        This method should transform the lidar measurements according to the particle's pose.
        Lidar measurements are a Numpy array of distance and angle pairs.
        Assuming oldLidarScan is a Nx3 array where each row is [quality, angle, distance]
        Transform lidar measurements based on the particle's position (x, y) and angle. The new
        measurements will be in the same format as oldLidarScan.
        """
        if not isinstance(oldLidarScan, np.ndarray) or oldLidarScan.ndim != 2 or oldLidarScan.shape[1] != 3:
            raise ValueError("oldLidarScan must be a Nx3 numpy array.")

        # Vector triangle calculations with domain and division by zero checks
        b = math.sqrt(self.x**2 + self.y**2)
        if self.x == 0 or b == 0:
            raise ValueError("Particle x and (x, y) position must not be zero to avoid division by zero.")
        cos_deltaY = (self.x**2 + b**2 - self.y**2) / (2 * self.x * b)
        cos_deltaY = min(max(cos_deltaY, -1.0), 1.0)
        deltaY = math.degrees(math.acos(cos_deltaY))

        # Vectorized implementation for efficiency
        qualities = oldLidarScan[:, 0]
        angles = oldLidarScan[:, 1]
        distances = oldLidarScan[:, 2]
        valid_mask = qualities >= 5
        qualities = qualities[valid_mask]
        angles = angles[valid_mask]
        distances = distances[valid_mask]

        # Calculate newDistance
        angle_diff_rad = np.radians(angles - deltaY)
        cos_angle_diff = np.cos(angle_diff_rad)
        newDistances = np.sqrt(distances**2 + b**2 - 2 * distances * b * cos_angle_diff)

        # Avoid division by zero in angleOppDistance
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = 2 * b * newDistances
            denom[denom == 0] = np.nan  # Avoid division by zero
            cos_angleOpp = (b**2 + newDistances**2 - distances**2) / denom
            cos_angleOpp = np.clip(cos_angleOpp, -1.0, 1.0)
            angleOppDistances = np.degrees(np.arccos(cos_angleOpp))

        angleDs = 180 - (self.angle - deltaY) - angleOppDistances
        angleDs = np.mod(angleDs, 360)  # Normalize to [0, 360)

        # Stack results
        transformedScan = np.column_stack((qualities, angleDs, newDistances))
        return transformedScan

    def calcCost(self, oldLidarScan: np.ndarray, newLidarScan: np.ndarray, sections: int = 16):
        """
        Calculate the cost of the particle given the lidar measurements.
        This method should compare the expected lidar measurements with the actual ones.
        """

        if not isinstance(newLidarScan, np.ndarray) or newLidarScan.ndim != 2 or newLidarScan.shape[1] != 3:
            raise ValueError("newLidarScan must be a Nx3 numpy array.")
        
        if sections <= 0:
            raise ValueError("Sections must be a positive integer.")

        expectedScan = self.calcEstLidarMeasurements(oldLidarScan)
        
        # Optimized: Use numpy digitize to bin angles into sections for efficiency
        bin_edges = np.linspace(0, 360, sections + 1)
        expected_bins = np.digitize(expectedScan[:, 1], bin_edges) - 1
        new_bins = np.digitize(newLidarScan[:, 1], bin_edges) - 1

        segmentCosts = np.zeros(sections)
        for i in range(sections):
            expectedSegment = expectedScan[expected_bins == i]
            newSegment = newLidarScan[new_bins == i]
            if expectedSegment.size == 0 or newSegment.size == 0: # No data in this segment
                continue
            # For efficiency, compare mean distances in each segment
            distances = np.abs(expectedSegment[:, 2].mean() - newSegment[:, 2].mean())
            segmentCosts[i] = distances

        totalCost = np.sum(segmentCosts)
        # Normalize the cost by the number of segments
        if sections > 0:
            totalCost /= sections

        # Set and return the cost
        self.cost = totalCost
        return self.cost

    def updatePose(self, x, y, angle):
        """
        Update the particle's pose.
        """
        self.x = copy.deepcopy(x)
        self.y = copy.deepcopy(y)
        self.angle = copy.deepcopy(angle)