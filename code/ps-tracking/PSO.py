from Particle import Particle
import numpy as np
import copy, time

class PSO:
    def __init__(self, oldLidarScan, newLidarScan, sections=16, particles_count=16, imuXReading=0.0, 
                 imuYReading=0.0, imuAngleReading=0.0, xNoise=0.1, yNoise=0.1, angleNoise=0.1, targetTime=0.1):
        self.oldLidarScan = oldLidarScan
        self.newLidarScan = newLidarScan
        self.sections = sections
        self.particles = []
        self.best_particle = None

        # Initialize particles
        for _ in range(particles_count):
            # Creates particles with random positions and angles normally distributed around the IMU readings. 
            # Noise values are representative of the percentage error in the middle 95% (2 standard deviations) of bias values.
            # Reading (mean) is multiplied by the noise value then divided by two to get the standard deviation.
            x = np.random.normal(imuXReading, imuXReading * xNoise / 2)
            y = np.random.normal(imuYReading, imuYReading * yNoise / 2)
            angle = np.random.normal(imuAngleReading, imuAngleReading * angleNoise / 2)
            particle = Particle(x, y, angle)
            self.particles.append(particle)

        # Calculate costs for each particle
        startTime = time.time()
        for particle in self.particles:
            cost = particle.calcCost(self.oldLidarScan, self.newLidarScan, self.sections)
            if self.best_particle is None or cost < self.best_particle.cost:
                # Create a new copy of the best particle and set its cost.
                self.best_particle = Particle(copy.deepcopy(particle.x), copy.deepcopy(particle.y), copy.deepcopy(particle.angle))
                self.best_particle.cost = cost
        costCalcTime = time.time() - startTime
