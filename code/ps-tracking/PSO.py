from Particle import Particle
import numpy as np
import copy, time
from concurrent.futures import ProcessPoolExecutor, as_completed

class PSO:
    def __init__(self, swarmSize: int, w: float, c1: float, c2: float, 
                 oldLidarScan: np.ndarray, newLidarScan: np.ndarray, sections: int = 16, imuXReading: float = 0.0,
                 imuYReading: float = 0.0, imuAngleReading: float = 0.0, xNoise: float = 0.1, yNoise: float = 0.1, 
                 angleNoise: float = 0.05, targetTime: float = 0.1):
        
        self.trueStartTime = time.time()
        
        # Initialize parameters
        self.swarmSize = swarmSize
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.oldLidarScan = oldLidarScan
        self.newLidarScan = newLidarScan
        self.sections = sections
        
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
        cost = particle.calcCost(self.oldLidarScan, self.newLidarScan, self.sections)
        return particle, cost

    def _update_particle(self, particle):
        """Helper function to update a single particle's velocity, position, and cost."""
        particle.updateVelocity(self.best_particle, self.w, self.c1, self.c2)
        particle.updatePosition()
        cost = particle.calcCost(self.oldLidarScan, self.newLidarScan, self.sections)
        return particle, cost

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