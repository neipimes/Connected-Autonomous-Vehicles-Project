import numpy as np
import math, copy

class Particle:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.cost = float('inf')  # Initialize cost to infinity

        self.xVelocity = np.random.uniform(-1, 1)
        self.yVelocity = np.random.uniform(-1, 1)
        self.angleVelocity = np.random.uniform(-0.1, 0.1)

        self.personalBest = (x, y, angle)
    
    # Below are copilot generated functions that transform lidar scans to the particle frame using an explicit cartesian approach.
    def calcEstLidarMeasurements(self, angles: np.ndarray, distances: np.ndarray):
        """
        Explicitly transform lidar scan points from the robot frame to the particle frame.
        lidar_scan: Nx3 array [quality, angle (deg), distance]
        Returns: Nx3 array [quality, angle (deg, in particle frame), distance (in particle frame)]
        """

        # Convert polar to Cartesian in robot frame
        x_robot = distances * np.sin(np.radians(angles)) # Sine and cosine are swapped here to match the polar lidar coordinate system.
        y_robot = distances * np.cos(np.radians(angles))

        # Transform to particle frame: translate by -self.x, -self.y
        x_trans = x_robot - self.x
        y_trans = y_robot - self.y

        # Rotate by -self.angle (to align with particle orientation)
        theta = np.radians(-self.angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_part = cos_theta * x_trans - sin_theta * y_trans
        y_part = sin_theta * x_trans + cos_theta * y_trans

        # Convert back to polar in particle frame
        distances_part = np.sqrt(x_part**2 + y_part**2)
        angles_part = np.degrees(np.arctan2(y_part, x_part))
        angles_part = np.mod(angles_part, 360)

        return np.column_stack((angles_part, distances_part))

    def sector_mean_bincount(self, distances, bin_idx, sections: int):
        # Function used to calculate mean distances in each sector using numpy bincount. Written by yanqing.liu@curtin.edu.au
        sum_dist = np.bincount(bin_idx, weights=distances, minlength=sections) # Get the sum of distances in each bin
        count = np.bincount(bin_idx, minlength=sections) # Get the count of distances in each bin
        with np.errstate(invalid='ignore'):
            mean_dist = np.where(count == 0, 0, sum_dist / count) # Calculate mean, set to 0 where count is 0
        return mean_dist

    def calcCost(self, newLidarScan: np.ndarray, angles: np.ndarray, distances: np.ndarray, sections: int = 16):
        """
        Calculate the cost of the particle given the lidar measurements.
        This method should compare the expected lidar measurements with the actual ones.
        """

        if not isinstance(newLidarScan, np.ndarray) or newLidarScan.ndim != 2 or newLidarScan.shape[1] != 3:
            raise ValueError("newLidarScan must be a Nx3 numpy array.")
        
        if sections <= 0:
            raise ValueError("Sections must be a positive integer.")

        expectedScan = self.calcEstLidarMeasurements(angles, distances)
        
        # Optimized: Use numpy digitize to bin angles into sections for efficiency
        bin_edges = np.linspace(0, 360, sections + 1)
        expected_bins = np.digitize(expectedScan[:, 0], bin_edges) - 1
        new_bins = np.digitize(newLidarScan[:, 0], bin_edges) - 1

        # Calculate mean distances for each bin using bincount courtesy of yanqing.liu@curtin.edu.au
        expected_means = self.sector_mean_bincount(expectedScan[:, 1], expected_bins, sections)
        new_means = self.sector_mean_bincount(newLidarScan[:, 1], new_bins, sections)

        # Compute segment costs using absolute differences
        segmentCosts = np.abs(expected_means - new_means)

        # Calculate total cost and normalize by the number of segments
        totalCost = np.sum(segmentCosts) / sections if sections > 0 else np.sum(segmentCosts)

        # Update personal best if current cost is lower
        if totalCost < self.cost:
            self.cost = totalCost
            self.personalBest = (self.x, self.y, self.angle)

        return totalCost

    def updateVelocity(self, best_particle, w_xy, c1_xy, c2_xy, w_angle, c1_angle, c2_angle):
        """
        Update the particle's velocity based on its own best position and the global best position.
        This method should implement the PSO velocity update formula.
        """
        if not isinstance(best_particle, Particle):
            raise ValueError("best_particle must be an instance of Particle.")

        # Random coefficients for exploration (independent for each component)
        r1_x, r2_x = np.random.rand(), np.random.rand()
        r1_y, r2_y = np.random.rand(), np.random.rand()
        r1_angle, r2_angle = np.random.rand(), np.random.rand()

        # Update velocity components using separate PSO parameters for XY and angle
        self.xVelocity = w_xy * self.xVelocity + c1_xy * r1_x * (self.personalBest[0] - self.x) + c2_xy * r2_x * (best_particle.x - self.x)
        self.yVelocity = w_xy * self.yVelocity + c1_xy * r1_y * (self.personalBest[1] - self.y) + c2_xy * r2_y * (best_particle.y - self.y)
        self.angleVelocity = w_angle * self.angleVelocity + c1_angle * r1_angle * (self.personalBest[2] - self.angle) + c2_angle * r2_angle * (best_particle.angle - self.angle)

    def updatePosition(self):
       """
       Update the particle's position based on its velocity.
       """
       self.x += self.xVelocity
       self.y += self.yVelocity
       self.angle += self.angleVelocity
       # Normalize angle to [0, 360)
       self.angle = np.mod(self.angle, 360)
