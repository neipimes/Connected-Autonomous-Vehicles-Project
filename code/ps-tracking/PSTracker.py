from imucontrol.imus import CAV_imus as imus
from PSO import PSO

import numpy as np
from rplidar import RPLidar
import multiprocessing as mp
import time, copy


class PSTracker:
    def __init__(self, swarmSize: int, w: float, c1: float, c2: float, sections: int = 16, targetTime: float = 1/15):
        """
        Initialize the PSTracker to grab IMU readings and run the PSO algorithm to track the particle swarm.
        :param swarmSize: Number of particles in the swarm.
        :param w: Inertia weight for PSO.
        :param c1: Cognitive coefficient for PSO.
        :param c2: Social coefficient for PSO.
        :param sections: Number of sections for lidar scan comparison.
        :param targetTime: Target time for the tracking loop.
        """

        self.swarmSize = swarmSize
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.sections = sections
        self.targetTime = targetTime

        self.xLocation = 0.0
        self.yLocation = 0.0
        self.angle = 0.0

        self.mutex = mp.Lock()  # Mutex for thread-safe access to IMU readings

        # Initialize IMU and Lidar
        imus.start()
        self.lidar = RPLidar('/dev/ttyUSB0') #TODO: Add adjustability for lidar port. Config file?

    def runIMUReadings(self):
        """
        Continuously read IMU data and return the latest readings.
        This method runs in a separate process to avoid blocking the main thread.
        Y location: FB (forward-backward) axis
        X location: LR (left-right) axis
        Angle: Yaw (rotation around the vertical axis)
        """

        GRAVITY = 9.80665 # m/s^2, standard gravity

        xVelocity = 0.0
        yVelocity = 0.0
        
        xDisplacement = 0.0
        yDisplacement = 0.0
        
        angleValue = 0.0

        startTime = time.time()
        currentRunningTime = 0.0
        while True:
            data = imus.getAvgData()
            priorRunningTime = copy.copy(currentRunningTime)
            currentRunningTime = time.time() - startTime
            timestep = currentRunningTime - priorRunningTime

            data[0] = data[0] * GRAVITY  # Convert acceleration to m/s^2
            data[1] = data[1] * GRAVITY  # Convert acceleration to m/s^2

            # Do calculations for accelerometer and gyroscope data
            xDisplacement = xVelocity * timestep + 0.5 * data[1] * timestep**2
            yDisplacement = yVelocity * timestep + 0.5 * data[0] * timestep**2
            xVelocity = xVelocity + data[1] * timestep  # m/s
            yVelocity = yVelocity + data[0] * timestep

            angleValue = angleValue + data[2] * timestep  # degrees
            # Normalize angle to [0, 360)
            angleValue = np.mod(angleValue, 360)

            # Update imu readings in class
            with self.mutex:
                self.xLocation = xDisplacement
                self.yLocation = yDisplacement
                self.angle = angleValue


    def start(self):
        """ 
        Start the PSTracker to continuously track the particle swarm.
        """
        # Start IMU readings in a separate process
        imu_process = mp.Process(target=self.runIMUReadings)
        imu_process.start()

        originScan = None
        priorScan = None
        # Continuously read lidar scans and run PSO
        for scan in self.lidar.iter_scans():
            # Convert scan to numpy array
            lidar_scan = np.array(scan)

            # Check if this is the first scan
            if priorScan is None:
                # Store the first scan as the initial scan
                priorScan = lidar_scan
                originScan = lidar_scan
                continue

            # Grab most up-to-date IMU readings using a mutex to ensure thread safety.
            with self.mutex:
                imuXReading = self.xLocation
                imuYReading = self.yLocation
                imuAngleReading = self.angle

            # Create a PSO instance with the current lidar scan and IMU readings
            pso = PSO(
                swarmSize=self.swarmSize,
                w=self.w,
                c1=self.c1,
                c2=self.c2,
                oldLidarScan=priorScan,  # Use the current scan as the old scan
                newLidarScan=lidar_scan,  # For simplicity, use the same scan as new
                sections=self.sections,
                imuXReading=imuXReading,
                imuYReading=imuYReading,
                imuAngleReading=imuAngleReading,
                targetTime=self.targetTime
            )

            # Run the PSO algorithm
            results = pso.run()
            priorScan = lidar_scan

            # Update x, y and angle based on the best particle's position
            with self.mutex:
                self.xLocation = results["x"]
                self.yLocation = results["y"]
                self.angle = results["angle"]