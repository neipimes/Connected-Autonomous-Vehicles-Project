#from PSO import PSO
from PSO_Single_Core import PSO
from Particle import Particle
import configparser
import numpy as np
import time, random, sys

def importParametersFromConfig(configFile):
    # Load configuration file
    config = configparser.ConfigParser()
    config.read(configFile)

    # Extract parameter values
    xPositionChanges = list(map(int, config['XPositionChanges']['values'].split(',')))
    yPositionChanges = list(map(int, config['YPositionChanges']['values'].split(',')))
    angleChanges = list(map(int, config['AngleChanges']['values'].split(',')))
    swarmSizes = list(map(int, config['SwarmSize']['values'].split(',')))
    maxIterations = list(map(int, config['MaxIterationCount']['values'].split(',')))
    angleWVals = list(map(float, config['AngleWVals']['values'].strip('{}').split(',')))
    angleC1Vals = list(map(float, config['AngleC1Vals']['values'].strip('{}').split(',')))
    angleC2Vals = list(map(float, config['AngleC2Vals']['values'].strip('{}').split(',')))
    xyWVals = list(map(float, config['XYWVals']['values'].strip('{}').split(',')))
    xyC1Vals = list(map(float, config['XYC1Vals']['values'].strip('{}').split(',')))
    xyC2Vals = list(map(float, config['XYC2Vals']['values'].strip('{}').split(',')))
    sectionCounts = list(map(int, config['SectionCount']['values'].split(',')))
    runs = list(map(int, config['Runs']['values'].split(',')))[0]

    # Print loaded parameters for verification
    print("X Position Changes:", xPositionChanges)
    print("Y Position Changes:", yPositionChanges)
    print("Angle Changes:", angleChanges)
    print("Swarm Sizes:", swarmSizes)
    print("Max Iterations:", maxIterations)
    print("Angle W Values:", angleWVals)
    print("Angle C1 Values:", angleC1Vals)
    print("Angle C2 Values:", angleC2Vals)
    print("XY W Values:", xyWVals)
    print("XY C1 Values:", xyC1Vals)
    print("XY C2 Values:", xyC2Vals)
    print("Section Counts:", sectionCounts)
    print("Runs:", runs)

    return xPositionChanges, yPositionChanges, angleChanges, swarmSizes, maxIterations, angleWVals, angleC1Vals, angleC2Vals, xyWVals, xyC1Vals, xyC2Vals, sectionCounts, runs

# Function to simulate LiDAR scans with noise
def simulateLidarScan(baseScan, noiseLevel=0.02):
    noise = np.random.uniform(-noiseLevel, noiseLevel, size=len(baseScan))
    return baseScan + noise

# Function to adjust scan positions
def adjustScanPosition(scan, xChange, yChange, angleChange):
    angles, distances = scan[:, 1], scan[:, 2]  # Extract angles and distances

    # Shift the scan using Particle's static scan shift method
    angles, distances = Particle.shiftLidarScan(angles, distances, xChange, yChange, angleChange)

    return np.column_stack((scan[:, 0], angles, distances))

# Function to down-sample LiDAR scans
def downSampleScan(scan):
    return scan[::2]

### PSO testing framework

# Test structure. Has been modified to allow optional passing of a second scan for comparison.
def testPsoAlgorithm(xPositionChanges, 
                     yPositionChanges, 
                     angleChanges, 
                     swarmSizes, 
                     maxIterations, 
                     angleWVals, 
                     angleC1Vals, 
                     angleC2Vals, 
                     xyWVals, 
                     xyC1Vals, 
                     xyC2Vals, 
                     sectionCounts,
                     numRuns,
                     scan1,
                     scan2 = None):

    for xChange in xPositionChanges:
        for yChange in yPositionChanges:
            for angleChange in angleChanges:
                adjustedScan = adjustScanPosition(scan1, xChange, yChange, angleChange) if scan2 is None else scan2

                for swarmSize in swarmSizes:
                    for maxIter in maxIterations:
                        for aW in angleWVals:
                            for aC1 in angleC1Vals:
                                for aC2 in angleC2Vals:
                                    for xyW in xyWVals:
                                        for xyC1 in xyC1Vals:
                                            for xyC2 in xyC2Vals:
                                                for sectionCount in sectionCounts:
                                                    print(f"Swarm Size: {swarmSize}, Max Iterations: {maxIter}, W: {aW}, C1: {aC1}, C2: {aC2}, XY W: {xyW}, XY C1: {xyC1}, XY C2: {xyC2}, Section Count: {sectionCount}")

                                                    runtimes = []
                                                    accuracies = []
                                                    costs = []

                                                    for run in range(numRuns):
                                                        # Initialize PSO with parameters
                                                        pso = PSO(swarmSize=swarmSize, w_xy=xyW, c1_xy=xyC1, c2_xy=xyC2,
                                                                w_angle=aW, c1_angle=aC1, c2_angle=aC2, oldLidarScan=scan1,
                                                                newLidarScan=adjustedScan, angleOffset=0, imuXReading=random.uniform(-10,10), 
                                                                imuYReading=random.uniform(-10,10), imuAngleReading=random.uniform(-1,1))

                                                        # Run PSO on simulated scans
                                                        startTime = time.time()
                                                        result = pso.runWithIterations(maxIterations=maxIter)
                                                        endTime = time.time()

                                                        # Measure metrics
                                                        runtime = endTime - startTime
                                                        runtimes.append(runtime)
                                                        #accuracy = np.linalg.norm(result - np.array([xChange, yChange, angleChange]))
                                                        xAccuracy = abs(result["x"] - xChange)
                                                        yAccuracy = abs(result["y"] - yChange)
                                                        angleAccuracy = abs(result["angle"] - angleChange)
                                                        accuracies.append((xAccuracy, yAccuracy, angleAccuracy))
                                                        costs.append(result["cost"])

                                                    print(f"Average Runtime: {np.mean(runtimes):.2f}s, Average Accuracy: {np.mean(accuracies):.2f}, Average Cost: {np.mean(costs):.2f}")


def main(configFile='testing/testPSO.conf'):
    # Load parameters and run the testing framework
    xPositionChanges, yPositionChanges, angleChanges, swarmSizes, maxIterations, angleWVals, angleC1Vals, angleC2Vals, xyWVals, xyC1Vals, xyC2Vals, sectionCounts, runs = importParametersFromConfig(configFile=configFile)

    # Import scans from testing/scans.txt
    with open('testing/scans.txt', 'r') as file:
        lines = file.read().split('\n\n')  # Split scans by blank line
        scan1 = np.array([list(map(float, line.strip('()').split(','))) for line in lines[0].splitlines()])
        scan2 = np.array([list(map(float, line.strip('()').split(','))) for line in lines[1].splitlines()])

    # Print loaded scans for verification
    #print("Scan 1:", scan1)
    #print("Scan 2:", scan2)

    # Test 1: Stationary and adjusted position scans
    testPsoAlgorithm(xPositionChanges, yPositionChanges, angleChanges, swarmSizes, maxIterations, angleWVals, angleC1Vals, angleC2Vals, xyWVals, xyC1Vals, xyC2Vals, sectionCounts, numRuns=runs, scan1=scan1)

if __name__ == "__main__":
    if len(sys.argv) > 1: # Specified file to read from
        importFile = sys.argv[1]
        main(configFile=importFile)
    else:
        main()