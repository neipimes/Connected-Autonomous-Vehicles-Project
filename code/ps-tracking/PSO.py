from Particle import Particle
import numpy as np
import copy, time
import multiprocessing

class PSO:
    def __init__(self, swarmSize: int, w: float, c1: float, c2: float, 
                 oldLidarScan: np.ndarray, newLidarScan: np.ndarray, sections: int = 16, imuXReading: float = 0.0,
                 imuYReading: float = 0.0, imuAngleReading: float = 0.0, xNoise: float = 0.1, yNoise: float = 0.1, 
                 angleNoise: float = 0.05, targetTime: float = 0.1):
        
        startTime = time.time()
        
        # Initialize parameters
        self.swarmSize = swarmSize
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.oldLidarScan = oldLidarScan
        self.newLidarScan = newLidarScan
        self.sections = sections

        self.targetTime = targetTime
        
        self.particles = []
        self.best_particle = None

        # Initialize particles
        for _ in range(self.swarmSize):
            # Creates particles with random positions and angles normally distributed around the IMU readings. 
            # Noise values are representative of the percentage error in the middle 95% (2 standard deviations) of bias values.
            # Reading (mean) is multiplied by the noise value then divided by two to get the standard deviation.
            x = np.random.normal(imuXReading, abs(imuXReading) * abs(xNoise) / 2)
            y = np.random.normal(imuYReading, abs(imuYReading) * abs(yNoise) / 2)
            angle = np.random.normal(imuAngleReading, abs(imuAngleReading) * abs(angleNoise) / 2)
            particle = Particle(x, y, angle)
            self.particles.append(particle)

        # Calculate costs for each particle in parallel
        def calculate_costs(particles_chunk, best_particle, lock):
            for particle in particles_chunk:
                cost = particle.calcCost(self.oldLidarScan, self.newLidarScan, self.sections)
                with lock:
                    if best_particle[0] is None or cost < best_particle[0].cost:
                        best_particle[0] = Particle(copy.deepcopy(particle.x), copy.deepcopy(particle.y), copy.deepcopy(particle.angle))
                        best_particle[0].cost = cost

        # Define shared objects and variables for parallel processing
        manager = multiprocessing.Manager()
        best_particle = manager.list([None])
        lock = manager.Lock()
        num_processors = 4
        chunk_size = len(self.particles) // num_processors

        processes = []
        for i in range(num_processors):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_processors - 1 else len(self.particles)
            chunk = self.particles[start_idx:end_idx]
            p = multiprocessing.Process(target=calculate_costs, args=(chunk, best_particle, lock))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        self.best_particle = best_particle[0]

        initTime = time.time() - startTime

        self.remainingTime = targetTime - initTime 
        # How long we have left to run the PSO algorithm due to Lidar constraints.
        # Remaining time should be a conservative estimate to ensure we don't exceed the target time.

    def run(self, analytics: bool = False):
        startTime = time.time()
        lastIterTime = 0.0
        iterCount = 0

        # Initialize shared objects and variables for parallel processing
        manager = multiprocessing.Manager()
        best_particle = manager.list([None])
        lock = manager.Lock()

        # Define the particle update function
        def update_particle(particle):
            particle.updateVelocity(best_particle[0], self.w, self.c1, self.c2)
            particle.updatePosition()
            cost = particle.calcCost(self.oldLidarScan, self.newLidarScan, self.sections)
            with lock:
                if best_particle[0] is None or cost < best_particle[0].cost:
                    best_particle[0] = Particle(copy.deepcopy(particle.x), copy.deepcopy(particle.y), copy.deepcopy(particle.angle))
                    best_particle[0].cost = cost

        initTime = time.time() - startTime
        self.remainingTime = self.remainingTime - initTime

        startTime = time.time()

        while time.time() - startTime < self.remainingTime - lastIterTime:
            iterationRunTime = time.time()
            iterCount += 1

            # Update particles in parallel using Pool
            with multiprocessing.Pool(processes=4) as pool:
                pool.map(update_particle, self.particles)

            self.best_particle = best_particle[0]
            lastIterTime = time.time() - iterationRunTime

        totalTime = time.time() - startTime

        # Return the best particle's position, angle, cost, total time taken and iteration count.
        return ({'x': self.best_particle.x, 'y': self.best_particle.y, 'angle': self.best_particle.angle,
                'cost': self.best_particle.cost, 'totalTime': totalTime, 'iterCount': iterCount})