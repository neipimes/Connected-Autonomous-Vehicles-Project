from imucontrol.imus import CAV_imus as imus
from PSO import PSO

import numpy as np
from rplidar import RPLidar
import multiprocessing as mp
import time, copy, logging, os


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

        # Start logging using the logging directory in home directory.
        logging.basicConfig(filename=os.path.expanduser("~/logs/pstracker.log"), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        if self.lidar is None:
            logging.error("Failed to connect to LiDAR. Please check the connection.")
            raise ConnectionError("LiDAR connection failed.")
        # Lidar starts on initialization, so we don't need to call start() here.
        
        logging.info("LiDAR and IMUs initialised successfully.")
        logging.info(f"PSTracker initialized with swarmSize={swarmSize}, w={w}, c1={c1}, c2={c2}, sections={sections}, targetTime={targetTime}.")
        

    def runIMUReadings(self):
        """
        Continuously read IMU data and return the latest readings.
        This method runs in a separate process to avoid blocking the main thread.
        Y location: FB (forward-backward) axis
        X location: LR (left-right) axis
        Angle: Yaw (rotation around the vertical axis)
        """

        """
        TODO: The LR axis could possibly be used as a complementary filter to particle adjustments 
        to 90 degrees either side of the displacement vector.
        LR axis could also possibly be used as a complimentary filter to angle readings as well, as there is no doubt a relationship
        betweeen the magnitude of an angle change and the magnitude of a reading on the LR axis due to centripital force.
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
            data = imus.getAvgData() # Blocking call to get IMU data
            priorRunningTime = copy.copy(currentRunningTime)
            currentRunningTime = time.time() - startTime
            timestep = currentRunningTime - priorRunningTime

            # Forward/Back: data[0], Left/Right: data[1], Angle change: data[2]
            data[0] = data[0] * GRAVITY  # Convert acceleration to m/s^2
            data[1] = data[1] * GRAVITY  # Convert acceleration to m/s^2
            
            # Adjust angle by the angle change and normalize angle to (0, 360)
            angleValue = angleValue + data[2] * timestep  # degrees
            angleValue = np.mod(angleValue, 360)

            # Use FB measurement and angle to get approximate location change. 
            # TODO: LR measurement could also be used for a complementary filter for angle measurement.
            xDelta = data[0] * np.sin(np.radians(angleValue))
            yDelta = data[0] * np.cos(np.radians(angleValue))

            # Do calculations for accelerometer and gyroscope data
            xDisplacement = xVelocity * timestep + 0.5 * xDelta * timestep**2
            yDisplacement = yVelocity * timestep + 0.5 * yDelta * timestep**2
            xVelocity = xVelocity + xDelta * timestep  # m/s
            yVelocity = yVelocity + yDelta * timestep

            # Update imu readings in class
            with self.mutex:
                self.xLocation = xDisplacement
                self.yLocation = yDisplacement
                self.angle = angleValue


    def start(self, useOriginScan: bool = False, debug: bool = False):
        """ 
        Start the PSTracker to continuously track the particle swarm.
        """
        logging.info("Starting PSTracker loop...")

        # Start IMU readings in a separate process
        imu_process = mp.Process(target=self.runIMUReadings)
        imu_process.start()
        logging.info("IMU readings process started.")

        originScan = None
        priorScan = None
        # Continuously read lidar scans and run PSO
        for scan in self.lidar.iter_scans():
            # Convert scan to numpy array
            lidar_scan = np.array(scan)

            # Check if this is the first scan
            if priorScan is None:
                # Store the first scan as the initial scan and continue to next iteration
                priorScan = lidar_scan
                originScan = lidar_scan
                continue

            # Grab most up-to-date IMU readings using a mutex to ensure thread safety.
            with self.mutex:
                imuXReading = self.xLocation
                imuYReading = self.yLocation
                imuAngleReading = self.angle

            # Create a PSO instance with the current lidar scan and IMU readings
            if useOriginScan:
                # Use the original scan as the prior scan
                priorScan = originScan

            pso = PSO(
                swarmSize=self.swarmSize,
                w=self.w,
                c1=self.c1,
                c2=self.c2,
                oldLidarScan=priorScan,
                newLidarScan=lidar_scan,
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

            # Debugging output
            if debug:
                print(f"PSO Results: X={self.xLocation:.2f}, Y={self.yLocation:.2f}, Angle={self.angle:.2f}")

    def close(self):
        """
        Close the PSTracker and stop the IMU readings.
        """
        imus.stop()
        self.lidar.stop()
        self.lidar.disconnect()
        logging.info("PSTracker closed and resources released.")



# TODO: Make a main function for testing purposes.
def main():
    try:
        tracker = PSTracker(swarmSize=10, w=0.5, c1=1.5, c2=1.5, sections=16, targetTime=1/15)
        tracker.start(useOriginScan = False, debug=True)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        tracker.close()
        logging.info("PSTracker has been closed successfully.")

if __name__ == "__main__":
    main()