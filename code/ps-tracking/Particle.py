import numpy as np
import math, copy

class Particle:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.cost = float('inf')  # Initialize cost to infinity

        self.xVelocity = np.random.uniform(-0.1, 0.1)
        self.yVelocity = np.random.uniform(-0.1, 0.1)
        self.angleVelocity = np.random.uniform(-0.1, 0.1)

    '''def calcEstLidarMeasurements(self, oldLidarScan: np.ndarray):
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
        return transformedScan'''
    
    # Below are copilot generated functions that transform lidar scans to the particle frame using an explicit cartesian approach.
    def calcEstLidarMeasurements(self, lidar_scan: np.ndarray):
        """
        Explicitly transform lidar scan points from the robot frame to the particle frame.
        lidar_scan: Nx3 array [quality, angle (deg), distance]
        Returns: Nx3 array [quality, angle (deg, in particle frame), distance (in particle frame)]
        """
        if not isinstance(lidar_scan, np.ndarray) or lidar_scan.ndim != 2 or lidar_scan.shape[1] != 3:
            raise ValueError("lidar_scan must be a Nx3 numpy array.")

        # Filter out low quality
        qualities = lidar_scan[:, 0]
        angles = lidar_scan[:, 1]
        distances = lidar_scan[:, 2]
        valid_mask = qualities >= 5
        qualities = qualities[valid_mask]
        angles = angles[valid_mask]
        distances = distances[valid_mask]

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

        return np.column_stack((qualities, angles_part, distances_part))

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

    def updatePose(self, x, y, angle): # TODO: Might not be needed.
        """
        Update the particle's pose.
        """
        self.x = copy.deepcopy(x)
        self.y = copy.deepcopy(y)
        self.angle = copy.deepcopy(angle)

    def updateVelocity(self, best_particle, w, c1, c2):
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

        # Update velocity components using PSO formula (independent randomness)
        self.xVelocity = w * self.xVelocity + c1 * r1_x * (best_particle.x - self.x) + c2 * r2_x * (best_particle.x - self.x)
        self.yVelocity = w * self.yVelocity + c1 * r1_y * (best_particle.y - self.y) + c2 * r2_y * (best_particle.y - self.y)
        self.angleVelocity = w * self.angleVelocity + c1 * r1_angle * (best_particle.angle - self.angle) + c2 * r2_angle * (best_particle.angle - self.angle)

    def updatePosition(self):
       """
       Update the particle's position based on its velocity.
       """
       self.x += self.xVelocity
       self.y += self.yVelocity
       self.angle += self.angleVelocity
       # Normalize angle to [0, 360)
       self.angle = np.mod(self.angle, 360)

    

    '''def transform_lidar_to_particle_frame_batch(self, lidar_scans: np.ndarray, particles: np.ndarray):
        """
        Batch version: lidar_scans is Nx3, particles is Mx3 (x, y, angle)
        Returns: list of M arrays, each Nx3 [quality, angle, distance] in that particle's frame
        """
        results = []
        for px, py, pa in particles:
            results.append(self.transform_lidar_to_particle_frame(lidar_scans, px, py, pa))
        return results'''
