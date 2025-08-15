from imucontrol.imus import CAV_imus as imus
from PSO import PSO

import numpy as np
from rplidar import RPLidar
import multiprocessing as mp
import time, copy, logging, os, sys
import argparse

# Start logging using the logging directory in home directory.
logging.basicConfig(filename=os.path.expanduser("~/logs/pstracker.log"), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PSTracker:
    def __init__(self, 
                 swarmSize: int, 
                 w: float, c1: float, c2: float, 
                 sections: int = 16,
                 xNoise: float = 0.1,
                 yNoise: float = 0.1,
                 angleNoise: float = 0.1,
                 targetTime: float = 1/15, 
                 motorPWM: int = 660, 
                 qualityCutoff: int = 0):
        """
        Initialize the PSTracker to grab IMU readings and run the PSO algorithm to track the particle swarm.
        :param swarmSize: Number of particles in the swarm.
        :param w: Inertia weight for PSO.
        :param c1: Cognitive coefficient for PSO.
        :param c2: Social coefficient for PSO.
        :param sections: Number of sections for lidar scan comparison.
        :param targetTime: Target time for the tracking loop.
        :param motorPWM: Motor PWM value for the LiDAR.
        :param qualityCutoff: Quality cutoff for LiDAR readings.
        """

        self._logger = logging.getLogger("PSTrackerLogger")
        if not self._logger.hasHandlers():
            log_path = os.path.expanduser("~/logs/pstracker.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            handler = logging.FileHandler(log_path)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
        self._logger.info("PSTracker instance created.")

        self.globalStop = False  # Flag to stop the tracking loop
        self.swarmSize = swarmSize
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.sections = sections
        self.targetTime = targetTime
        self.qualityCutoff = qualityCutoff
        self.xNoise = xNoise
        self.yNoise = yNoise
        self.angleNoise = angleNoise

        # Initialize IMU and Lidar
        imus.start()
        self.lidar = RPLidar('/dev/ttyUSB0', baudrate=256000) #TODO: Add adjustability for lidar port. Config file?
        if self.lidar is None:
            self._logger.error("Failed to connect to LiDAR. Please check the connection.")
            raise ConnectionError("LiDAR connection failed.")
        # Lidar starts on initialization, so we don't need to call start() here.
        self.lidar.motor_speed = motorPWM # Set motor speed in PWM value (0-1023)
        self.lidar.start_motor()
        time.sleep(5) # Allow some time for the LiDAR to start up and stabilize.
        #self.lidar.reset()  # Clear any initial data from the LiDAR.
        
        self._logger.info("LiDAR and IMUs initialised successfully.")
        self._logger.info(f"PSTracker initialized with swarmSize={swarmSize}, w={w}, c1={c1}, c2={c2}, sections={sections}, targetTime={targetTime}.")

    @staticmethod
    def runIMUReadings(xLocation, yLocation, angle, psoUpdate, mutex, debug=True):
        """
        Continuously read IMU data and return the latest readings.
        This method runs in a separate process to avoid blocking the main thread.
        Y location: FB (forward-backward) axis
        X location: LR (left-right) axis
        Angle: Yaw (rotation around the vertical axis)
        FB and LR measurements are initially in m/s^2 (to be converted to mm/s^2), angle is in degrees.
        """

        """
        TODO: The LR axis could possibly be used as a complementary filter to particle adjustments 
        to 90 degrees either side of the displacement vector.
        LR axis could also possibly be used as a complimentary filter to angle readings as well, as there is no doubt a relationship
        betweeen the magnitude of an angle change and the magnitude of a reading on the LR axis due to centripital force.
        """

        GRAVITY = 9.80665 # m/s^2, standard gravity

        # Initialize local state from shared values
        xDisplacement = 0.0
        yDisplacement = 0.0
        angleValue = 0.0
        xVelocity = 0.0
        yVelocity = 0.0
        timestep = 0.1  # Default timestep in seconds

        startTime = time.time()
        currentRunningTime = 0.0

        while True:
            # Check for PSO update flag and reinitialize local state if set
            with mutex: # Only hold mutex at the start of the loop to avoid blocking other processes.
                if psoUpdate.value == 1:  # 1 means True (update needed)
                    # If there has been an update during the last iteration,
                    # throw out the old IMU reading and update the local state with the latest values.
                    if debug:
                        print("PSO update detected, reinitializing local state.")
                        sys.stdout.flush()
                    xDisplacement = copy.deepcopy(xLocation.value)
                    yDisplacement = copy.deepcopy(yLocation.value)
                    angleValue = copy.deepcopy(angle.value)
                    psoUpdate.value = 0  # 0 means False (no update needed)
                else:
                    xLocation.value = copy.deepcopy(xDisplacement)
                    yLocation.value = copy.deepcopy(yDisplacement)
                    angle.value = copy.deepcopy(angleValue)

                    if debug:
                        print(
                            f"IMU Results: "
                            f"X={xLocation.value:.2f}, "
                            f"Y={yLocation.value:.2f}, "
                            f"Angle={angle.value:.2f}, "
                            f"xVelocity={xVelocity:.2f}, "
                            f"yVelocity={yVelocity:.2f}, "
                            f"timestep={timestep:.4f}"
                        )
                        sys.stdout.flush()  # Ensure the output is printed immediately


            data = imus.getAvgData() # Blocking call to get IMU data
            
            priorRunningTime = copy.copy(currentRunningTime)
            currentRunningTime = time.time() - startTime
            timestep = currentRunningTime - priorRunningTime

            # Forward/Back: data[0], Left/Right: data[1], Angle change: data[2]
            fbData = data[0] * GRAVITY  # Convert acceleration to m/s^2
            lrData = data[1] * GRAVITY  # Convert acceleration to m/s^2

            # Convert this data to mm/s^2 for consistency with the PSO algorithm.
            fbData *= 1000  # Convert m/s^2 to mm/s^2
            lrData *= 1000  # Convert m/s^2 to mm/s^2

            # Adjust angle by the angle change and normalize angle to (0, 360)
            angleValue = angleValue + data[2] * timestep  # degrees
            angleValue = np.mod(angleValue, 360)

            # Use FB measurement and angle to get approximate location change. 
            # TODO: LR measurement could also be used for a complementary filter for angle measurement.
            xDelta = fbData * np.sin(np.radians(angleValue))
            yDelta = fbData * np.cos(np.radians(angleValue))

            # Do calculations for accelerometer and gyroscope data
            xDisplacement = xDisplacement + xVelocity * timestep + 0.5 * xDelta * timestep**2
            yDisplacement = yDisplacement + yVelocity * timestep + 0.5 * yDelta * timestep**2
            xVelocity = xVelocity + xDelta * timestep  # mm/s
            yVelocity = yVelocity + yDelta * timestep                
            

    def start(self, useOriginScan: bool = False, debug: bool = False, testing: bool = False, noLidar: bool = False, duration: float = None):
        """ 
        Start the PSTracker to continuously track the particle swarm.
        """
        self._logger.info("Starting PSTracker loop...")

        # Clear lidar input to ensure no stale data
        #self.lidar.reset()

        try:
            # Create shared variables for IMU and PSO readings.
            xLocation = mp.Value('d', 0.0)  # double precision float
            yLocation = mp.Value('d', 0.0)
            angle = mp.Value('d', 0.0)
            psoUpdate = mp.Value('i', 0)  # Integer for PSO update flag (1=True, 0=False)
            mutex = mp.Lock()  # Mutex for thread-safe access to IMU readings

            # Testing variables
            avgIterations = 0
            avgCost = 0.0
            runCount = 0

            # Start IMU readings in a separate process
            imu_process = mp.Process(target=PSTracker.runIMUReadings, args=(xLocation, yLocation, angle, psoUpdate, mutex, debug))
            imu_process.start()
            self._logger.info("IMU readings process started.")

            cycleEndTime = None
            originScan = None
            priorScan = None
            start_time = time.time() if duration else None

            # Continuously read lidar scans and run PSO
            if noLidar == False:
                for scan in self.lidar.iter_scans():
                    psoStartTime = time.time()

                    if cycleEndTime is not None:
                        scanGapTime = psoStartTime - cycleEndTime
                        # This time metric could possibly be used to dynamically adjust the target time value.
                    else:
                        scanGapTime = 0.0

                    if duration and time.time() - start_time >= duration:
                        self._logger.info(f"Duration reached. Terminating PSTracker loop after {time.time() - start_time:.2f} seconds.")
                        break

                    # Convert scan to numpy array
                    lidar_scan = np.array(scan)

                    # Check if this is the first scan
                    if priorScan is None:
                        # Store the first scan as the initial scan and continue to next iteration
                        priorScan = lidar_scan
                        originScan = lidar_scan
                        continue

                    # Grab most up-to-date IMU readings using a mutex to ensure thread safety.
                    with mutex:
                        imuXReading = copy.deepcopy(xLocation.value)
                        imuYReading = copy.deepcopy(yLocation.value)
                        imuAngleReading = copy.deepcopy(angle.value)

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
                        xNoise=self.xNoise,
                        yNoise=self.yNoise,
                        angleNoise=self.angleNoise,
                        imuXReading=imuXReading,
                        imuYReading=imuYReading,
                        imuAngleReading=imuAngleReading,
                        targetTime=self.targetTime
                    )

                    # Run the PSO algorithm
                    
                    results = pso.run()
                    outerRunTime = time.time() - psoStartTime
                    priorScan = lidar_scan

                    # Update x, y and angle based on the best particle's position
                    with mutex:
                        xLocation.value = results["x"]
                        yLocation.value = results["y"]
                        angle.value = results["angle"]
                        psoUpdate.value = 1  # 1 means True (update needed)

                        # Debugging output
                        if debug:
                            print(
                                f"\n--- PSO Results ---\n"
                                f"X: {xLocation.value:.2f}\n"
                                f"Y: {yLocation.value:.2f}\n"
                                f"Angle: {angle.value:.2f}\n"
                                f"Iterations: {results['iterCount']}\n"
                                f"Cost: {results['cost']:.4f}\n"
                                f"True Total Time: {results['trueTotalTime']:.4f} s\n"
                                f"Init Time: {results['initTime']:.4f} s\n"
                                f"Outer Run Time: {outerRunTime:.4f} s\n"
                                f"Time gap between scans: {scanGapTime:.4f} s\n"
                                f"-------------------\n"
                            )

                        if testing:
                            runCount += 1
                            avgIterations += results['iterCount']
                            avgCost += results['cost']

                    if self.globalStop:
                        self._logger.info("Global stop signal received. Terminating PSTracker loop.")
                        imu_process.terminate()
                        imu_process.join()
                        break

                    cycleEndTime = time.time()

        except Exception as e:
            print(e)
            self._logger.error(f"An error occurred in PSTracker: {e}")
            imu_process.terminate()
            imu_process.join()
            self._logger.info("IMU readings process terminated due to error.")

        return (xLocation.value, yLocation.value, angle.value, avgIterations / runCount if testing and runCount > 0 else 0, avgCost / runCount if testing and runCount > 0 else 0)


    def close(self):
        """
        Close the PSTracker and stop the IMU readings.
        """
        self.globalStop = True  # Set the global stop flag to terminate the loop
        self.lidar.stop()
        self.lidar.stop_motor()
        self.lidar.disconnect()
        self._logger.info("PSTracker closed and resources released.")



def main(debug: bool = False, useOriginScan: bool = False, swarmSize: int = 10, 
         w: float = 0.2, c1: float = 0.3, c2: float = 1.5, sections: int = 16, targetTime: float = 1/15,
         noLidar: bool = False, motorPWM: int = 660):
    try:
        calibrateChoice = input("Calibrate IMUs? (y/N): ").strip().lower()
        if calibrateChoice == 'y':
            imus.calibrateAll(50)
            logging.info("IMUs calibrated successfully.")
        elif calibrateChoice == 'n' or calibrateChoice == '':
            logging.info("Skipping IMU calibration.")
        else:
            print("Invalid choice. Please enter 'y' or 'n' or <Enter>.")
            return
        tracker = PSTracker(swarmSize=swarmSize, w=w, c1=c1, c2=c2, sections=sections, targetTime=targetTime, motorPWM=motorPWM)
        tracker.start(useOriginScan=useOriginScan, debug=debug, noLidar=noLidar)
    finally:
        tracker.close()
        logging.info("PSTracker has been closed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSTracker Command Line Options")
    parser.add_argument('--swarmSize', type=int, default=10, help='Number of particles in the swarm (default: 10)')
    parser.add_argument('--w', type=float, default=0.3, help='Inertia weight for PSO')
    parser.add_argument('--c1', type=float, default=0.8, help='Cognitive coefficient for PSO')
    parser.add_argument('--c2', type=float, default=2.5, help='Social coefficient for PSO')
    parser.add_argument('--sections', type=int, default=16, help='Number of sections for lidar scan comparison')
    parser.add_argument('--targetTime', type=float, default=1/15, help='Target time for the tracking loop')
    parser.add_argument('--motorPWM', type=int, default=660, help='Motor PWM value to set speed (0-1023)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--originScan', action='store_true', help='Use origin scan as prior scan')
    parser.add_argument('--noLidar', action='store_true', help='Do not use lidar for tracking (for testing purposes)')
    args = parser.parse_args()
    main(
        debug=args.debug,
        useOriginScan=args.originScan,
        swarmSize=args.swarmSize,
        w=args.w,
        c1=args.c1,
        c2=args.c2,
        sections=args.sections,
        targetTime=args.targetTime,
        motorPWM=args.motorPWM
    )