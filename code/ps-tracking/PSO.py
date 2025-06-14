from Particle import Particle
import numpy as np
import copy, time

class PSO:
    def __init__(self, swarmSize: int, w: float, c1: float, c2: float, 
                 oldLidarScan: np.ndarray, newLidarScan: np.ndarray, sections: int = 16, imuXReading: float = 0.0,
                 imuYReading: float = 0.0, imuAngleReading: float = 0.0, xNoise: float = 0.1, yNoise: float = 0.1, 
                 angleNoise: float = 0.1, targetTime: float = 0.1):
        
        startTime = time.time()
        
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

        # Initialize particles
        for _ in range(self.swarmSize):
            # Creates particles with random positions and angles normally distributed around the IMU readings. 
            # Noise values are representative of the percentage error in the middle 95% (2 standard deviations) of bias values.
            # Reading (mean) is multiplied by the noise value then divided by two to get the standard deviation.
            x = np.random.normal(imuXReading, imuXReading * xNoise / 2)
            y = np.random.normal(imuYReading, imuYReading * yNoise / 2)
            angle = np.random.normal(imuAngleReading, imuAngleReading * angleNoise / 2)
            particle = Particle(x, y, angle)
            self.particles.append(particle)

        # Calculate costs for each particle
        for particle in self.particles:
            cost = particle.calcCost(self.oldLidarScan, self.newLidarScan, self.sections)
            if self.best_particle is None or cost < self.best_particle.cost:
                # Create a new copy of the best particle and set its cost.
                self.best_particle = Particle(copy.deepcopy(particle.x), copy.deepcopy(particle.y), copy.deepcopy(particle.angle))
                self.best_particle.cost = cost
        
        initTime = time.time() - startTime

        self.remainingTime = targetTime - initTime 
        # How long we have left to run the PSO algorithm due to Lidar constraints.
        # Remaining time should be a conservative estimate to ensure we don't exceed the target time.

    def run(self):
        startTime = time.time()
        lastIterTime = 0.0
        iterCount = 0
        while time.time() - startTime < self.remainingTime - lastIterTime: 
            # Run until the remaining time is up and a new scan will be available.
            # Don't start a new iteration if the last one took long enough that it would exceed the remaining time.
            iterationRunTime = time.time()
            iterCount += 1
            for particle in self.particles:
                # Update particle velocity and position
                particle.updateVelocity(self.best_particle, self.w, self.c1, self.c2) # TODO: These particle functions
                particle.updatePosition()
                # Calculate cost for the new position
                cost = particle.calcCost(self.oldLidarScan, self.newLidarScan, self.sections)
                if self.best_particle is None or cost < self.best_particle.cost:
                    self.best_particle = Particle(copy.deepcopy(particle.x), copy.deepcopy(particle.y), copy.deepcopy(particle.angle))
                    self.best_particle.cost = cost
            lastIterTime = time.time() - iterationRunTime
        totalTime = time.time() - startTime

        # Return the best particle's position, angle, cost, total time taken and iteration count.
        return ({'x': self.best_particle.x, 'y': self.best_particle.y, 'angle': self.best_particle.angle,
                'cost': self.best_particle.cost, 'totalTime': totalTime, 'iterCount': iterCount})