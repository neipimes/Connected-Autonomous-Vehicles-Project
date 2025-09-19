from Particle import Particle
import numpy as np
import copy, time
from concurrent.futures import ProcessPoolExecutor, as_completed

class PSO:
    def __init__(self, swarmSize: int, w_xy: float, c1_xy: float, c2_xy: float, w_angle: float, c1_angle: float, c2_angle: float,
                 oldLidarScan: np.ndarray, newLidarScan: np.ndarray, sections: int = 16, imuXReading: float = 0.0,
                 imuYReading: float = 0.0, imuAngleReading: float = 0.0, xNoise: float = 0.1, yNoise: float = 0.1, 
                 angleNoise: float = 0.005, targetTime: float = 0.1):
        
        self.trueStartTime = time.time()
        
        # Initialize parameters
        self.swarmSize = swarmSize
        self.w_xy = w_xy
        self.c1_xy = c1_xy
        self.c2_xy = c2_xy
        self.w_angle = w_angle
        self.c1_angle = c1_angle
        self.c2_angle = c2_angle

        self.sections = sections

        # Filter the lidar scans to remove low quality points
        filteredOldTuple = self._filter_scan(oldLidarScan)
        self.qualitiesOld = filteredOldTuple[0]
        self.anglesOld = filteredOldTuple[1]
        self.distancesOld = filteredOldTuple[2]

        filteredNewTuple = self._filter_scan(newLidarScan)
        self.qualitiesNew = filteredNewTuple[0]
        self.anglesNew = filteredNewTuple[1]
        self.distancesNew = filteredNewTuple[2]
        self.newLidarScan = np.column_stack((self.qualitiesNew, self.anglesNew, self.distancesNew))

        self.particles = []
        self.best_particle = None

        # Parallelize particle initialization
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self._initialize_particle, imuXReading, imuYReading, imuAngleReading, xNoise, yNoise, angleNoise)
                for _ in range(self.swarmSize)
            ]
            for future in futures:
                particle, cost = future.result()
                self.particles.append(particle)
                if self.best_particle is None or cost < self.best_particle.cost:
                    # Create a new copy of the best particle and set its cost.
                    self.best_particle = Particle(copy.deepcopy(particle.x), copy.deepcopy(particle.y), copy.deepcopy(particle.angle))
                    self.best_particle.cost = cost

        self.initTime = time.time() - self.trueStartTime

        self.remainingTime = targetTime - self.initTime 
        # How long we have left to run the PSO algorithm due to Lidar constraints.
        # Remaining time should be a conservative estimate to ensure we don't exceed the target time.

    def _initialize_particle(self, imuXReading, imuYReading, imuAngleReading, xNoise, yNoise, angleNoise):
        """Helper function to initialize a single particle."""
        x = np.random.normal(imuXReading, abs(imuXReading) * abs(xNoise) / 2)
        y = np.random.normal(imuYReading, abs(imuYReading) * abs(yNoise) / 2)
        angle = np.random.normal(imuAngleReading, abs(imuAngleReading) * abs(angleNoise) / 2)
        particle = Particle(x, y, angle)
        cost = particle.calcCost(self.newLidarScan, self.anglesOld, self.distancesOld, self.sections)
        return particle, cost

    def _update_particle(self, particle):
        """Helper function to update a single particle's velocity, position, and cost."""
        particle.updateVelocity(
            self.best_particle,
            self.w_xy, self.c1_xy, self.c2_xy,
            self.w_angle, self.c1_angle, self.c2_angle
        )
        particle.updatePosition()
        cost = particle.calcCost(self.newLidarScan, self.anglesOld, self.distancesOld, self.sections)
        return particle, cost

    def _filter_scan(self, scan: np.ndarray):

        if not isinstance(scan, np.ndarray) or scan.ndim != 2 or scan.shape[1] != 3:
            raise ValueError("Lidar scan must be a Nx3 numpy array.")

        # Filter out low quality points from the lidar scan
        qualities = scan[:, 0]
        angles = scan[:, 1]
        distances = scan[:, 2]
        valid_mask = qualities >= 5
        qualities = qualities[valid_mask]
        angles = angles[valid_mask]
        distances = distances[valid_mask]

        return (qualities, angles, distances)

    def run(self, analytics: bool = False):
        startTime = time.time()
        lastIterTime = 0.0
        iterCount = 0
        while time.time() - startTime < self.remainingTime - lastIterTime:
            # Run until the remaining time is up and a new scan will be available.
            # Don't start a new iteration if the last one took long enough that it would exceed the remaining time.
            iterationRunTime = time.time()
            iterCount += 1

            # Parallelize particle updates and cost calculations
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(self._update_particle, particle): particle for particle in self.particles}
                for future in as_completed(futures):
                    particle, cost = future.result()
                    if self.best_particle is None or cost < self.best_particle.cost:
                        # Update the best particle in a thread-safe manner
                        self.best_particle = Particle(copy.deepcopy(particle.x), copy.deepcopy(particle.y), copy.deepcopy(particle.angle))
                        self.best_particle.cost = cost

            lastIterTime = time.time() - iterationRunTime

        totalTime = time.time() - startTime
        trueTotalTime = time.time() - self.trueStartTime

        # Return the best particle's position, angle, cost, total time taken, and iteration count.
        return ({'x': self.best_particle.x, 'y': self.best_particle.y, 'angle': self.best_particle.angle,
                'cost': self.best_particle.cost, 'totalTime': totalTime, 'iterCount': iterCount,
                'trueTotalTime': trueTotalTime, 'initTime': self.initTime})