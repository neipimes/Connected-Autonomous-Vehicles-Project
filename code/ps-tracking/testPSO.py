#from PSO import PSO
from PSO_Single_Core import PSO
from Particle import Particle
import configparser
import numpy as np
import time, random, sys
import matplotlib.pyplot as plt

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

def offsetScanAngle(scan, offset):
    scan[:, 1] = (scan[:, 1] - offset) % 360
    return scan

def buildSwarmSizeGraphs(swarmSizeResults):
    # Using matplotlib to plot swarm size results
    # First graph - Average Inaccuracy vs Swarm Size: All three inaccuracies on one graph
    # Second graph - Average Cost vs Swarm Size
    # Third graph - Average Runtime vs Swarm Size

    swarmSizes = [result[0] for result in swarmSizeResults]
    avgXInaccuracies = [result[1] for result in swarmSizeResults]
    avgYInaccuracies = [result[2] for result in swarmSizeResults]
    avgAngleInaccuracies = [result[3] for result in swarmSizeResults]
    avgCosts = [result[4] for result in swarmSizeResults]
    avgRuntimes = [result[5] for result in swarmSizeResults]

    plt.figure(figsize=(18, 5))

    # First graph
    plt.subplot(1, 3, 1)
    plt.plot(swarmSizes, avgXInaccuracies, label='X Inaccuracy')
    plt.plot(swarmSizes, avgYInaccuracies, label='Y Inaccuracy')
    plt.plot(swarmSizes, avgAngleInaccuracies, label='Angle Inaccuracy')
    plt.title('Average Inaccuracy vs Swarm Size')
    plt.xlabel('Swarm Size')
    plt.ylabel('Average Inaccuracy')
    plt.legend()

    # Second graph
    plt.subplot(1, 3, 2)
    plt.plot(swarmSizes, avgCosts, label='Average Cost', color='orange')
    plt.title('Average Cost vs Swarm Size')
    plt.xlabel('Swarm Size')
    plt.ylabel('Average Cost')

    # Third graph
    plt.subplot(1, 3, 3)
    plt.plot(swarmSizes, avgRuntimes, label='Average Runtime', color='green')
    plt.title('Average Runtime vs Swarm Size')
    plt.xlabel('Swarm Size')
    plt.ylabel('Average Runtime')

    plt.tight_layout()
    plt.show()


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

                for sectionCount in sectionCounts:
                    for maxIter in maxIterations:
                        for aW in angleWVals:
                            for aC1 in angleC1Vals:
                                for aC2 in angleC2Vals:
                                    for xyW in xyWVals:
                                        for xyC1 in xyC1Vals:
                                            for xyC2 in xyC2Vals:
                                                ssInaccuracyResults = [] # List of tuples (swarmSize, avgXInaccuracy, avgYInaccuracy, avgAngleInaccuracy, avgCost, avgRuntime)
                                                for swarmSize in swarmSizes:
                                                    print(f"Swarm Size: {swarmSize}, Max Iterations: {maxIter}, XChange: {xChange}, YChange: {yChange}, AngleChange: {angleChange}, W: {aW}, C1: {aC1}, C2: {aC2}, XY W: {xyW}, XY C1: {xyC1}, XY C2: {xyC2}, Section Count: {sectionCount}")

                                                    runtimes = []
                                                    inaccuracies = []
                                                    costs = []
                                                    runResults = []

                                                    for run in range(numRuns):
                                                        # Initialize PSO with parameters
                                                        pso = PSO(swarmSize=swarmSize, w_xy=xyW, c1_xy=xyC1, c2_xy=xyC2,
                                                                w_angle=aW, c1_angle=aC1, c2_angle=aC2, oldLidarScan=scan1,
                                                                newLidarScan=adjustedScan, angleOffset=0, imuXReading=random.uniform(xChange - abs(xChange * 0.25),xChange + abs(xChange * 0.25)), 
                                                                imuYReading=random.uniform(yChange - abs(yChange * 0.25),yChange + abs(yChange * 0.25)), imuAngleReading=(random.uniform(angleChange - abs(angleChange * 0.25),angleChange + abs(angleChange * 0.25))%360))

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

                                                        # Calculate angle accuracy considering wrap-around at 360 degrees
                                                        angle_diff = (result["angle"] - angleChange) % 360
                                                        angleAccuracy = min(angle_diff, 360 - angle_diff)

                                                        inaccuracies.append((xAccuracy, yAccuracy, angleAccuracy))
                                                        runResults.append((result["x"], result["y"], result["angle"]))
                                                        costs.append(result["cost"])

                                                    inaccuraciesPercentage = [(abs(acc[0])/max(abs(xChange),1), abs(acc[1])/max(abs(yChange),1), abs(acc[2])/max(abs(angleChange),1)) for acc in inaccuracies]
                                                    print(f"Average Runtime: {np.mean(runtimes):.2f}s,\nAverage X Inaccuracy: {np.mean([acc[0] for acc in inaccuracies]):.2f} ({(np.mean([acc[0] for acc in inaccuraciesPercentage])*100):.2f}%),\nAverage Y Inaccuracy: {np.mean([acc[1] for acc in inaccuracies]):.2f} ( {(np.mean([acc[1] for acc in inaccuraciesPercentage])*100):.2f}%),\nAverage Angle Inaccuracy: {np.mean([acc[2] for acc in inaccuracies]):.2f} ( {(np.mean([acc[2] for acc in inaccuraciesPercentage])*100):.2f}%),\nAverage X: {np.mean([pos[0] for pos in runResults]):.2f}, Average Y: {np.mean([pos[1] for pos in runResults]):.2f},\nAverage Angle: {np.mean([pos[2] for pos in runResults]):.2f},\nAverage Cost: {np.mean(costs):.2f}\n")

                                                    # Gather and save metrics from run
                                                    ssInaccuracyResults.append((swarmSize, np.mean([acc[0] for acc in inaccuraciesPercentage]), np.mean([acc[1] for acc in inaccuraciesPercentage]), np.mean([acc[2] for acc in inaccuraciesPercentage]), np.mean(costs), np.mean(runtimes)))
                                                #TODO: swarmsize comparison for this parameter set.
                                                buildSwarmSizeGraphs(ssInaccuracyResults)


def main(configFile='testing/testPSO.conf'):
    # Load parameters and run the testing framework
    xPositionChanges, yPositionChanges, angleChanges, swarmSizes, maxIterations, angleWVals, angleC1Vals, angleC2Vals, xyWVals, xyC1Vals, xyC2Vals, sectionCounts, runs = importParametersFromConfig(configFile=configFile)

    # Import scans from testing/scans.txt
    with open('testing/scans.txt', 'r') as file:
        lines = file.read().split('\n\n')  # Split scans by blank line
        scan1 = np.array([list(map(float, line.strip('()').split(','))) for line in lines[0].splitlines()])
        scan2 = np.array([list(map(float, line.strip('()').split(','))) for line in lines[1].splitlines()])

    #print("Original Scan 1 before offset:", scan1)

    scan1 = offsetScanAngle(scan1, 307.5)
    scan2 = offsetScanAngle(scan2, 307.5)

    # Print loaded scans for verification
    #print("Scan 1:", scan1)
    # Adjust scan 1 to simulate a small movement
    scan1Adj = adjustScanPosition(scan1, 0, 10, 0)
    #print("Adjusted Scan 1 (10,10,0):", scan1Adj)
    #print("Scan 2:", scan2)

    '''
    # Test particle and associated functions
    testParticle = Particle(0, 10, 0)
    estMeasurements = testParticle.calcEstLidarMeasurements(scan1[:, 1], scan1[:, 2])
    #print("Estimated Measurements from Particle at (0,10,0):", estMeasurements)
    costTestParticle = testParticle.calcCost(scan1Adj, scan1[:, 1], scan1[:, 2], sections=180)
    #print("Cost of Test Particle against Adjusted (0, 10, 0) Scan 1:", costTestParticle)
    '''

    # Test 1: Stationary and adjusted position scans
    testPsoAlgorithm(xPositionChanges, yPositionChanges, angleChanges, swarmSizes, maxIterations, angleWVals, angleC1Vals, angleC2Vals, xyWVals, xyC1Vals, xyC2Vals, sectionCounts, numRuns=runs, scan1=scan1)

if __name__ == "__main__":
    if len(sys.argv) > 1: # Specified file to read from
        importFile = sys.argv[1]
        main(configFile=importFile)
    else:
        main()