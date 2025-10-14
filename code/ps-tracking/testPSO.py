#from PSO import PSO
from PSO_Single_Core import PSO
from Particle import Particle
import configparser
import numpy as np
import time, random, sys
import os, json, csv
from datetime import datetime
import matplotlib.pyplot as plt

def importParametersFromConfig(configFile):
    # Load configuration file
    config = configparser.ConfigParser()
    config.read(configFile)

    # Extract parameter values
    xPositionChanges = list(map(float, config['XPositionChanges']['values'].split(',')))
    yPositionChanges = list(map(float, config['YPositionChanges']['values'].split(',')))
    angleChanges = list(map(float, config['AngleChanges']['values'].split(',')))
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

    # Calculate average runtime per particle
    averageRuntimePerParticle = []
    for i in range(len(avgRuntimes)):
        if swarmSizes[i] > 0:
            averageRuntimePerParticle.append(avgRuntimes[i] / swarmSizes[i])
        else:
            averageRuntimePerParticle.append(0)

    print("Average Runtime per Particle for each Swarm Size:")
    for i in range(len(swarmSizes)):
        print(f"Swarm Size: {swarmSizes[i]}, Average Runtime per Particle: {averageRuntimePerParticle[i]:.6f} seconds")

    print("\nOverall Average Runtime per Particle across all Swarm Sizes:")
    print(f"{np.mean(averageRuntimePerParticle):.6f} seconds")

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

def buildIterationGraphs(inaccuracyResults):
    # Using matplotlib to plot swarm size results
    # First graph - Average Inaccuracy vs Max Iter: All three inaccuracies on one graph
    # Second graph - Average Cost vs Max Iter
    # Third graph - Average Runtime vs Max Iter

    maxIters = [result[6] for result in inaccuracyResults]
    avgXInaccuracies = [result[1] for result in inaccuracyResults]
    avgYInaccuracies = [result[2] for result in inaccuracyResults]
    avgAngleInaccuracies = [result[3] for result in inaccuracyResults]
    avgCosts = [result[4] for result in inaccuracyResults]
    avgRuntimes = [result[5] for result in inaccuracyResults]

    # Calculate average runtime per iteration
    averageRuntimePerIteration = []
    for i in range(len(avgRuntimes)):
        if maxIters[i] > 0:
            averageRuntimePerIteration.append(avgRuntimes[i] / maxIters[i])
        else:
            averageRuntimePerIteration.append(0)

    print("Average Runtime per iteration for each Max Iteration:")
    for i in range(len(maxIters)):
        print(f"Max Iteration: {maxIters[i]}, Average Runtime per iteration: {averageRuntimePerIteration[i]:.6f} seconds")

    print("\nOverall Average Runtime per iteration across all Max Iterations:")
    print(f"{np.mean(averageRuntimePerIteration):.6f} seconds")

    plt.figure(figsize=(18, 5))

    # First graph
    plt.subplot(1, 3, 1)
    plt.plot(maxIters, avgXInaccuracies, label='X Inaccuracy')
    plt.plot(maxIters, avgYInaccuracies, label='Y Inaccuracy')
    plt.plot(maxIters, avgAngleInaccuracies, label='Angle Inaccuracy')
    plt.title('Average Inaccuracy vs Max Iter')
    plt.xlabel('Max Iter')
    plt.ylabel('Average Inaccuracy')
    plt.legend()

    # Second graph
    plt.subplot(1, 3, 2)
    plt.plot(maxIters, avgCosts, label='Average Cost', color='orange')
    plt.title('Average Cost vs Max Iter')
    plt.xlabel('Max Iter')
    plt.ylabel('Average Cost')

    # Third graph
    plt.subplot(1, 3, 3)
    plt.plot(maxIters, avgRuntimes, label='Average Runtime', color='green')
    plt.title('Average Runtime vs Max Iter')
    plt.xlabel('Max Iter')
    plt.ylabel('Average Runtime')

    plt.tight_layout()
    plt.show()


### PSO testing framework
def testSingleRun(xChange, 
                     yChange, 
                     angleChange, 
                     swarmSize, 
                     maxIter, 
                     aW, 
                     aC1, 
                     aC2, 
                     xyW, 
                     xyC1, 
                     xyC2, 
                     sectionCount,
                     numRuns,
                     scan1,
                     scan2 = None):
    xChange = xChange[0]
    yChange = yChange[0]
    angleChange = angleChange[0]

    if scan2 is None:
        adjustedScan = adjustScanPosition(scan1, xChange, yChange, angleChange)
    else:
        adjustedScan = scan2    

    # Initialize PSO with parameters
    pso = PSO(swarmSize=swarmSize[0], w_xy=xyW[0], c1_xy=xyC1[0], c2_xy=xyC2[0],
            w_angle=aW[0], c1_angle=aC1[0], c2_angle=aC2[0], oldLidarScan=scan1,
            newLidarScan=adjustedScan, angleOffset=0, sections=sectionCount[0], imuXReading=random.uniform(xChange - abs(xChange * 0.25),xChange + abs(xChange * 0.25)), 
            imuYReading=random.uniform(yChange - abs(yChange * 0.25),yChange + abs(yChange * 0.25)), imuAngleReading=(random.uniform(angleChange - abs(angleChange * 0.25),angleChange + abs(angleChange * 0.25))%360))

    # Run PSO on simulated scans
    startTime = time.time()
    result = pso.runWithIterations(maxIterations=maxIter[0])
    endTime = time.time()

    # Measure metrics
    runtime = endTime - startTime
    #accuracy = np.linalg.norm(result - np.array([xChange, yChange, angleChange]))
    xAccuracy = abs(result["x"] - xChange)
    yAccuracy = abs(result["y"] - yChange)

    # Calculate angle accuracy considering wrap-around at 360 degrees
    angle_diff = (result["angle"] - angleChange) % 360
    angleAccuracy = min(angle_diff, 360 - angle_diff)

    xAccuracyPercent = abs(xAccuracy)/max(abs(xChange),1)
    yAccuracyPercent = abs(yAccuracy)/max(abs(yChange),1)
    angleAccuracyPercent = abs(angleAccuracy)/max(abs(angleChange),1)

    print(f'''Single Run Results - Swarm Size: {swarmSize}, Max Iterations: {maxIter}, 
          XChange: {xChange}, YChange: {yChange}, AngleChange: {angleChange}, 
          W: {aW}, C1: {aC1}, C2: {aC2}, XY W: {xyW}, XY C1: {xyC1}, XY C2: {xyC2}, Section Count: {sectionCount}
          Average Runtime: {runtime:.2f}s,
          X Inaccuracy: {xAccuracy:.2f} ({(xAccuracyPercent*100):.2f}%), Y Inaccuracy: {yAccuracy:.2f} ({(yAccuracyPercent*100):.2f}%), Angle Inaccuracy: {angleAccuracy:.2f} ({(angleAccuracyPercent*100):.2f}%),
          X: {result['x']:.2f}, Y: {result['y']:.2f}, Angle: {result['angle']:.2f},
          Cost: {result['cost']:.2f}\n''')

    bestCosts = result['best_costs']
    avgCosts = result['avg_costs']

    # Graphs showing best costs and average costs over iterations
    plt.figure(figsize=(10, 5))
    plt.plot(bestCosts, label='Best Costs')
    plt.plot(avgCosts, label='Average Costs')
    plt.title('Costs over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()

def write_grouped_data_to_files(grouped_param_set_data, out_dir='testing/grouped_results'):
    os.makedirs(out_dir, exist_ok=True)

    # Get current date and time string for filenames
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a JSON summary of all groups (keys as string)
    serializable = {}
    for key, entries in grouped_param_set_data.items():
        key_str = f"x{key[0]}_y{key[1]}_a{key[2]}"
        serializable[key_str] = []
        for entry in entries:
            params, avgX, avgY, avgAngle, avgCost, avgRuntime = entry
            serializable[key_str].append({
                "params": list(params),
                "avg_x_inacc": avgX,
                "avg_y_inacc": avgY,
                "avg_angle_inacc": avgAngle,
                "avg_cost": avgCost,
                "avg_runtime": avgRuntime
            })

    # Find a unique JSON filename
    json_base = f"grouped_results_{dt_str}.json"
    json_path = os.path.join(out_dir, json_base)
    counter = 1
    while os.path.exists(json_path):
        json_path = os.path.join(out_dir, f"grouped_results_{dt_str}_{counter}.json")
        counter += 1

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(serializable, jf, indent=2)

    # Write a CSV file per group for easy viewing
    for key, entries in grouped_param_set_data.items():
        filename_base = f"group_x{key[0]}_y{key[1]}_a{key[2]}_{dt_str}.csv"
        path = os.path.join(out_dir, filename_base)
        counter = 1
        while os.path.exists(path):
            path = os.path.join(out_dir, f"group_x{key[0]}_y{key[1]}_a{key[2]}_{dt_str}_{counter}.csv")
            counter += 1
        with open(path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(["params", "avg_x_inacc", "avg_y_inacc", "avg_angle_inacc", "avg_cost", "avg_runtime"])
            for entry in entries:
                params, avgX, avgY, avgAngle, avgCost, avgRuntime = entry
                writer.writerow([";".join(map(str, params)), avgX, avgY, avgAngle, avgCost, avgRuntime])

    print(f"Grouped data written to: {out_dir}")

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
                     scan2 = None,
                     originInitialization = False,
                     swarmSizeComparison = False,
                     iterationComparison = False):
    
    paramSetData = [] # List of tuples (params, avgXInaccuracy, avgYInaccuracy, avgAngleInaccuracy, avgCost, avgRuntime)
    testStartTime = time.time()
    
    for xChange in xPositionChanges:
        for yChange in yPositionChanges:
            for angleChange in angleChanges:
                adjustedScan = adjustScanPosition(scan1, xChange, yChange, angleChange) if scan2 is None else scan2

                for sectionCount in sectionCounts:
                    iterationData = [] # List of tuples (swarmSize, avgXInaccuracy, avgYInaccuracy, avgAngleInaccuracy, avgCost, avgRuntime, maxIter)
                    for maxIter in maxIterations:
                        for aW in angleWVals:
                            for aC1 in angleC1Vals:
                                for aC2 in angleC2Vals:
                                    for xyW in xyWVals:
                                        for xyC1 in xyC1Vals:
                                            for xyC2 in xyC2Vals:
                                                inaccuracyResults = [] # List of tuples (swarmSize, avgXInaccuracy, avgYInaccuracy, avgAngleInaccuracy, avgCost, avgRuntime, maxIter)
                                                for swarmSize in swarmSizes:
                                                    print(f"Swarm Size: {swarmSize}, Max Iterations: {maxIter}, XChange: {xChange}, YChange: {yChange}, AngleChange: {angleChange}, W: {aW}, C1: {aC1}, C2: {aC2}, XY W: {xyW}, XY C1: {xyC1}, XY C2: {xyC2}, Section Count: {sectionCount}")

                                                    runtimes = []
                                                    inaccuracies = []
                                                    costs = []
                                                    runResults = []

                                                    for run in range(numRuns):
                                                        # Initialize PSO with parameters

                                                        if originInitialization:
                                                            # Initialize particles around origin with small noise
                                                            pso = PSO(swarmSize=swarmSize, w_xy=xyW, c1_xy=xyC1, c2_xy=xyC2,
                                                                w_angle=aW, c1_angle=aC1, c2_angle=aC2, oldLidarScan=scan1,
                                                                newLidarScan=adjustedScan, angleOffset=0, sections=sectionCount, imuXReading=random.uniform(-0.5, 0.5), 
                                                                imuYReading=random.uniform(-0.5, 0.5), imuAngleReading=random.uniform(-1, 1))
                                                        else:
                                                            # Initialize particles around IMU reading with some noise (within 25% of actual change)
                                                            pso = PSO(swarmSize=swarmSize, w_xy=xyW, c1_xy=xyC1, c2_xy=xyC2,
                                                                w_angle=aW, c1_angle=aC1, c2_angle=aC2, oldLidarScan=scan1,
                                                                newLidarScan=adjustedScan, angleOffset=0, sections=sectionCount, imuXReading=random.uniform(xChange - abs(xChange * 0.25),xChange + abs(xChange * 0.25)), 
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
                                                    inaccuracyResults.append((swarmSize, np.mean([acc[0] for acc in inaccuraciesPercentage]), np.mean([acc[1] for acc in inaccuraciesPercentage]), np.mean([acc[2] for acc in inaccuraciesPercentage]), np.mean(costs), np.mean(runtimes), maxIter))
                                                    iterationData.append((swarmSize, np.mean([acc[0] for acc in inaccuraciesPercentage]), np.mean([acc[1] for acc in inaccuraciesPercentage]), np.mean([acc[2] for acc in inaccuraciesPercentage]), np.mean(costs), np.mean(runtimes), maxIter))
                                                    paramSetData.append(((xChange, yChange, angleChange, maxIter, swarmSize, aW, aC1, aC2, xyW, xyC1, xyC2, sectionCount), np.mean([acc[0] for acc in inaccuraciesPercentage]), np.mean([acc[1] for acc in inaccuraciesPercentage]), np.mean([acc[2] for acc in inaccuraciesPercentage]), np.mean(costs), np.mean(runtimes)))

                                                buildSwarmSizeGraphs(inaccuracyResults) if swarmSizeComparison else None
                    buildIterationGraphs(iterationData) if iterationComparison else None
    
    # Split paramSetData into groups based on xChange, yChange and angleChange combinations. For example, all entries with (10,10,1) together.
    groupedParamSetData = {}
    for entry in paramSetData:
        params = entry[0]
        key = (params[0], params[1], params[2])  # (xChange, yChange, angleChange)
        if key not in groupedParamSetData:
            groupedParamSetData[key] = []
        groupedParamSetData[key].append(entry)

    # Write grouped data to file for possible future analysis and appendix data.
    write_grouped_data_to_files(groupedParamSetData)

    # For each group, print top 5 metrics
    for key, group in groupedParamSetData.items():
        print(f"\n------------------------------------------------------------------------\nResults for XChange: {key[0]}, YChange: {key[1]}, AngleChange: {key[2]}")
        top5Metrics(group)

    testEndTime = time.time()
    print(f"\nTotal Testing Time: {(testEndTime - testStartTime)/60:.2f} minutes")
# end def

def top5Metrics(paramSetData):
    # Process paramSetData to find best parameter set based on various metrics
    # 1. Sort by lowest average total inaccuracy (sum of x, y, angle inaccuracies)
    totalInaccuracySorted = sorted(paramSetData, key=lambda x: x[1] + x[2] + x[3])
    print("\nTop 5 Parameter Sets by Lowest Total Inaccuracy (X + Y + Angle):")
    for entry in totalInaccuracySorted[:5]: # Top 5
        params, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime = entry
        print(f'''Params: {params}
            Avg X Inacc: {avgXInacc:.4f}, Avg Y Inacc: {avgYInacc:.4f}, Avg Angle Inacc: {avgAngleInacc:.4f}, 
            Avg Total Inacc: {avgXInacc + avgYInacc + avgAngleInacc:.4f}, Avg Cost: {avgCost:.2f}, Avg Runtime: {avgRuntime:.2f}s''')
        
    # 2. Sort by lowest average cost
    costSorted = sorted(paramSetData, key=lambda x: x[4])
    print("\nTop 5 Parameter Sets by Lowest Average Cost:")
    for entry in costSorted[:5]: # Top 5
        params, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime = entry
        print(f'''Params: {params}
            Avg X Inacc: {avgXInacc:.4f}, Avg Y Inacc: {avgYInacc:.4f}, Avg Angle Inacc: {avgAngleInacc:.4f}, 
            Avg Total Inacc: {avgXInacc + avgYInacc + avgAngleInacc:.4f}, Avg Cost: {avgCost:.2f}, Avg Runtime: {avgRuntime:.2f}s''')
        
    # 3. Sort by lowest average runtime
    runtimeSorted = sorted(paramSetData, key=lambda x: x[5])
    print("\nTop 5 Parameter Sets by Lowest Average Runtime:")
    for entry in runtimeSorted[:5]: # Top 5
        params, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime = entry
        print(f'''Params: {params}
            Avg X Inacc: {avgXInacc:.4f}, Avg Y Inacc: {avgYInacc:.4f}, Avg Angle Inacc: {avgAngleInacc:.4f}, 
            Avg Total Inacc: {avgXInacc + avgYInacc + avgAngleInacc:.4f}, Avg Cost: {avgCost:.2f}, Avg Runtime: {avgRuntime:.2f}s''')
        
    # 4. Sort by best balance of accuracy and runtime (lowest total inaccuracy * avg runtime)
    balanceSorted = sorted(paramSetData, key=lambda x: (x[1] + x[2] + x[3]) * x[5])
    print("\nTop 5 Parameter Sets by Best Balance of Accuracy and Runtime (Total Inaccuracy * Avg Runtime):")
    for entry in balanceSorted[:5]: # Top 5
        params, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime = entry
        print(f'''Params: {params}
            Avg X Inacc: {avgXInacc:.4f}, Avg Y Inacc: {avgYInacc:.4f}, Avg Angle Inacc: {avgAngleInacc:.4f}, 
            Avg Total Inacc: {avgXInacc + avgYInacc + avgAngleInacc:.4f}, Avg Cost: {avgCost:.2f}, Avg Runtime: {avgRuntime:.2f}s''')

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

    runChoice = input("Run a single run test (y/N)? ")
    if runChoice.lower() == 'y':
        # Prompt for file containing parameters
        paramFile = input("Enter the parameter file path (default ./testing/singlePSO.conf): ") or './testing/singlePSO.conf'
        singleParams = importParametersFromConfig(configFile=paramFile)
        testSingleRun(*singleParams, scan1=scan1)

    
    # Test 3: Swarm size and iteration count comparisons
    runChoice = input("Run swarm size comparison tests (y/N)? ")
    if runChoice.lower() == 'y':
        paramFile = input("Enter the parameter file path (default ./testing/swarmSizeTest.conf): ") or './testing/swarmSizeTest.conf'
        swarmSizeParams = importParametersFromConfig(configFile=paramFile)
        testPsoAlgorithm(*swarmSizeParams, scan1=scan1, swarmSizeComparison=True)

    runChoice = input("Run max iteration count comparison tests (y/N)? ")
    if runChoice.lower() == 'y':
        paramFile = input("Enter the parameter file path (default ./testing/maxIterTest.conf): ") or './testing/maxIterTest.conf'
        maxIterParams = importParametersFromConfig(configFile=paramFile)
        testPsoAlgorithm(*maxIterParams, scan1=scan1, iterationComparison=True)

    # Test 1: Stationary and adjusted position scans
    testPsoAlgorithm(xPositionChanges, yPositionChanges, angleChanges, swarmSizes, maxIterations, angleWVals, angleC1Vals, angleC2Vals, xyWVals, xyC1Vals, xyC2Vals, sectionCounts, numRuns=runs, scan1=scan1)

if __name__ == "__main__":
    if len(sys.argv) > 1: # Specified file to read from
        importFile = sys.argv[1]
        main(configFile=importFile)
    else:
        main()