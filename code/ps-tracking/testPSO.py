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

def buildAccuracyComparisonGraphs(accuracyResults):
    """
    Build graphs showing Total Average Inaccuracy (TAI) vs Swarm Size and Average Cost vs Swarm Size
    with multiple lines for different parameter combinations, plus a focused view on lowest section count.
    Displays as three separate figures.
    
    accuracyResults: List of tuples (swarmSize, angleW, xyW, sectionCount, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime)
    """
    from collections import defaultdict
    
    # Group results by parameter combinations (angleW, xyW, sectionCount)
    groupedResults = defaultdict(list)
    
    for result in accuracyResults:
        swarmSize, angleW, xyW, sectionCount, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime = result
        tai = avgXInacc + avgYInacc + avgAngleInacc  # Total Average Inaccuracy
        
        key = (angleW, xyW, sectionCount)
        groupedResults[key].append((swarmSize, tai, avgCost))
    
    # Sort each group by swarm size for proper line plotting
    for key in groupedResults:
        groupedResults[key].sort(key=lambda x: x[0])
    
    # Find the lowest section count
    all_section_counts = [key[2] for key in groupedResults.keys()]
    lowest_section_count = min(all_section_counts)
    
    # Define colors and line styles for different combinations
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black', 'lime', 'indigo', 'coral', 'gold', 'silver', 'teal', 'navy', 'maroon', 'khaki', 'plum', 'salmon', 'tan']
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
    
    # Figure 1: TAI vs Swarm Size (All Data)
    plt.figure(figsize=(11, 8))
    color_idx = 0
    for key, data in groupedResults.items():
        angleW, xyW, sectionCount = key
        swarmSizes = [point[0] for point in data]
        tais = [point[1] for point in data]
        
        label = f'Angle w={angleW}, XY w={xyW}, Sections={sectionCount}'
        line_style = line_styles[color_idx % len(line_styles)]
        color = colors[color_idx % len(colors)]
        marker = markers[color_idx % len(markers)]
        
        plt.plot(swarmSizes, tais, label=label, color=color, linestyle=line_style, marker=marker, markersize=6, linewidth=2)
        color_idx += 1
    
    plt.title('Total Average Inaccuracy (TAI) vs Swarm Size\nfor All Parameter Combinations', fontsize=16)
    plt.xlabel('Swarm Size', fontsize=14)
    plt.ylabel('Total Average Inaccuracy (TAI)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Average Cost vs Swarm Size (All Data)
    plt.figure(figsize=(11, 8))
    color_idx = 0
    for key, data in groupedResults.items():
        angleW, xyW, sectionCount = key
        swarmSizes = [point[0] for point in data]
        costs = [point[2] for point in data]
        
        label = f'Angle w={angleW}, XY w={xyW}, Sections={sectionCount}'
        line_style = line_styles[color_idx % len(line_styles)]
        color = colors[color_idx % len(colors)]
        marker = markers[color_idx % len(markers)]
        
        plt.plot(swarmSizes, costs, label=label, color=color, linestyle=line_style, marker=marker, markersize=6, linewidth=2)
        color_idx += 1
    
    plt.title('Average Cost vs Swarm Size\nfor All Parameter Combinations', fontsize=16)
    plt.xlabel('Swarm Size', fontsize=14)
    plt.ylabel('Average Cost', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Figure 3: TAI vs Swarm Size (Lowest Section Count Only)
    plt.figure(figsize=(11, 8))
    color_idx = 0
    for key, data in groupedResults.items():
        angleW, xyW, sectionCount = key
        
        if sectionCount == lowest_section_count:
            swarmSizes = [point[0] for point in data]
            tais = [point[1] for point in data]
            
            label_focused = f'Angle w={angleW}, XY w={xyW}'
            line_style = line_styles[color_idx % len(line_styles)]
            color = colors[color_idx % len(colors)]
            marker = markers[color_idx % len(markers)]
            
            plt.plot(swarmSizes, tais, label=label_focused, color=color, linestyle=line_style, marker=marker, markersize=6, linewidth=2)
            color_idx += 1
    
    plt.title(f'Total Average Inaccuracy (TAI) vs Swarm Size\nfor Lowest Section Count ({lowest_section_count} sections)', fontsize=16)
    plt.xlabel('Swarm Size', fontsize=14)
    plt.ylabel('Total Average Inaccuracy (TAI)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
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

def write_accuracy_comparison_data_to_files(accuracyResults, out_dir='testing/grouped_results'):
    """
    Write accuracy comparison data to files in the same format as existing grouped results.
    
    accuracyResults: List of tuples (swarmSize, angleW, xyW, sectionCount, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Get current date and time string for filenames
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Group by parameter combinations for JSON format
    grouped_accuracy_data = {}
    for result in accuracyResults:
        swarmSize, angleW, xyW, sectionCount, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime = result
        
        # Create a key similar to existing format but for accuracy comparison
        key_str = f"AccComp_aW{angleW}_xyW{xyW}_sec{sectionCount}"
        
        if key_str not in grouped_accuracy_data:
            grouped_accuracy_data[key_str] = []
        
        # Format similar to existing grouped results
        entry = {
            "params": [swarmSize, angleW, xyW, sectionCount],
            "avg_x_inacc": avgXInacc,
            "avg_y_inacc": avgYInacc,
            "avg_angle_inacc": avgAngleInacc,
            "avg_cost": avgCost,
            "avg_runtime": avgRuntime,
            "tai": avgXInacc + avgYInacc + avgAngleInacc
        }
        grouped_accuracy_data[key_str].append(entry)

    # Find a unique JSON filename
    json_base = f"accuracy_comparison_results_{dt_str}.json"
    json_path = os.path.join(out_dir, json_base)
    counter = 1
    while os.path.exists(json_path):
        json_path = os.path.join(out_dir, f"accuracy_comparison_results_{dt_str}_{counter}.json")
        counter += 1

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(grouped_accuracy_data, jf, indent=2)

    # Write a single consolidated CSV file with all results
    csv_base = f"accuracy_comparison_all_results_{dt_str}.csv"
    csv_path = os.path.join(out_dir, csv_base)
    counter = 1
    while os.path.exists(csv_path):
        csv_path = os.path.join(out_dir, f"accuracy_comparison_all_results_{dt_str}_{counter}.csv")
        counter += 1
    
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["swarm_size", "angle_w", "xy_w", "section_count", "avg_x_inacc", "avg_y_inacc", "avg_angle_inacc", "avg_cost", "avg_runtime", "tai"])
        
        # Write all results sorted by parameter combination and then by swarm size
        for result in sorted(accuracyResults, key=lambda x: (x[1], x[2], x[3], x[0])):  # Sort by angleW, xyW, sectionCount, swarmSize
            swarmSize, angleW, xyW, sectionCount, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime = result
            tai = avgXInacc + avgYInacc + avgAngleInacc
            writer.writerow([swarmSize, angleW, xyW, sectionCount, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime, tai])

    print(f"Accuracy comparison data written to: {out_dir}")
    print(f"JSON summary: {json_path}")
    print(f"Consolidated CSV file: {csv_path}")

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

def testAccuracyComparison(xPositionChanges, 
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
                          scan2=None):
    """
    Test accuracy comparison across different parameter combinations.
    Focus on TAI (Total Average Inaccuracy) vs Swarm Size relationships.
    """
    
    accuracyResults = []  # List of tuples (swarmSize, angleW, xyW, sectionCount, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime)
    testStartTime = time.time()
    
    # Use first values for position changes and angle changes (fixed for this test)
    xChange = xPositionChanges[0]
    yChange = yPositionChanges[0] 
    angleChange = angleChanges[0]
    maxIter = maxIterations[0]
    
    print(f"Running accuracy comparison test with fixed displacement: X={xChange}, Y={yChange}, Angle={angleChange}")
    print(f"Max iterations: {maxIter}, Runs per configuration: {numRuns}")
    print("Testing parameter combinations for TAI analysis...")
    
    # Create adjusted scan once for all tests
    adjustedScan = adjustScanPosition(scan1, xChange, yChange, angleChange) if scan2 is None else scan2
    
    total_combinations = len(swarmSizes) * len(angleWVals) * len(xyWVals) * len(sectionCounts)
    current_combination = 0
    
    for angleW in angleWVals:
        for xyW in xyWVals:
            for sectionCount in sectionCounts:
                for swarmSize in swarmSizes:
                    current_combination += 1
                    print(f"Testing combination {current_combination}/{total_combinations}: Swarm Size={swarmSize}, Angle w={angleW}, XY w={xyW}, Sections={sectionCount}")
                    
                    runtimes = []
                    inaccuracies = []
                    costs = []
                    
                    for run in range(numRuns):
                        # Initialize PSO with current parameter combination
                        pso = PSO(swarmSize=swarmSize, w_xy=xyW, c1_xy=xyC1Vals[0], c2_xy=xyC2Vals[0],
                                w_angle=angleW, c1_angle=angleC1Vals[0], c2_angle=angleC2Vals[0], 
                                oldLidarScan=scan1, newLidarScan=adjustedScan, angleOffset=0, 
                                sections=sectionCount, 
                                imuXReading=random.uniform(xChange - abs(xChange * 0.25), xChange + abs(xChange * 0.25)), 
                                imuYReading=random.uniform(yChange - abs(yChange * 0.25), yChange + abs(yChange * 0.25)), 
                                imuAngleReading=(random.uniform(angleChange - abs(angleChange * 0.25), angleChange + abs(angleChange * 0.25)) % 360))

                        # Run PSO
                        startTime = time.time()
                        result = pso.runWithIterations(maxIterations=maxIter)
                        endTime = time.time()

                        # Calculate metrics
                        runtime = endTime - startTime
                        runtimes.append(runtime)
                        
                        xAccuracy = abs(result["x"] - xChange)
                        yAccuracy = abs(result["y"] - yChange)
                        
                        # Calculate angle accuracy considering wrap-around at 360 degrees
                        angle_diff = (result["angle"] - angleChange) % 360
                        angleAccuracy = min(angle_diff, 360 - angle_diff)
                        
                        # Convert to percentage inaccuracies
                        xAccuracyPercent = abs(xAccuracy) / max(abs(xChange), 1)
                        yAccuracyPercent = abs(yAccuracy) / max(abs(yChange), 1)
                        angleAccuracyPercent = abs(angleAccuracy) / max(abs(angleChange), 1)
                        
                        inaccuracies.append((xAccuracyPercent, yAccuracyPercent, angleAccuracyPercent))
                        costs.append(result["cost"])
                    
                    # Calculate averages for this parameter combination
                    avgXInacc = np.mean([acc[0] for acc in inaccuracies])
                    avgYInacc = np.mean([acc[1] for acc in inaccuracies])
                    avgAngleInacc = np.mean([acc[2] for acc in inaccuracies])
                    avgCost = np.mean(costs)
                    avgRuntime = np.mean(runtimes)
                    
                    # Store results
                    accuracyResults.append((swarmSize, angleW, xyW, sectionCount, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime))
                    
                    # Print summary for this combination
                    tai = avgXInacc + avgYInacc + avgAngleInacc
                    print(f"  TAI: {tai:.4f}, Avg Runtime: {avgRuntime:.2f}s, Avg Cost: {avgCost:.2f}")
    
    testEndTime = time.time()
    print(f"\nAccuracy comparison test completed in {(testEndTime - testStartTime)/60:.2f} minutes")
    
    # Save results data in the same format as existing grouped results
    write_accuracy_comparison_data_to_files(accuracyResults)
    
    # Generate graphs
    buildAccuracyComparisonGraphs(accuracyResults)
    
    # Print summary of best configurations by TAI
    print("\n" + "="*80)
    print("ACCURACY COMPARISON RESULTS SUMMARY")
    print("="*80)
    
    # Sort by TAI (Total Average Inaccuracy)
    sortedByTAI = sorted(accuracyResults, key=lambda x: x[4] + x[5] + x[6])
    
    print("\nTop 10 Parameter Combinations by Lowest Total Average Inaccuracy (TAI):")
    print("-" * 80)
    for i, result in enumerate(sortedByTAI[:10]):
        swarmSize, angleW, xyW, sectionCount, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime = result
        tai = avgXInacc + avgYInacc + avgAngleInacc
        print(f"{i+1:2d}. Swarm={swarmSize:3d}, AngleW={angleW}, XYW={xyW}, Sections={sectionCount:3d} | TAI={tai:.4f}, Runtime={avgRuntime:.2f}s")
    
    return accuracyResults

def buildPositionalShiftGraphs(positionalShiftResults):
    """
    Build graphs showing Total Average Inaccuracy (TAI) vs Positional Shifts
    for both origin initialization and IMU-based initialization.
    Displays as two separate figures.
    
    positionalShiftResults: List of tuples (shiftIndex, xChange, yChange, angleChange, angleW, xyW, initType, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime)
    """
    from collections import defaultdict
    
    # Group results by initialization type and parameter combinations
    originResults = defaultdict(list)
    imuResults = defaultdict(list)
    
    # Create position shift labels for x-axis
    shiftLabels = []
    shiftPositions = []
    
    for result in positionalShiftResults:
        shiftIndex, xChange, yChange, angleChange, angleW, xyW, initType, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime = result
        tai = avgXInacc + avgYInacc + avgAngleInacc
        
        # Create shift label
        if shiftIndex not in [item[0] for item in shiftPositions]:
            shiftPositions.append((shiftIndex, f"({xChange}, {yChange}, {angleChange})"))
            shiftLabels.append(f"({xChange}, {yChange}, {angleChange})")
        
        key = (angleW, xyW)
        
        if initType == "origin":
            originResults[key].append((shiftIndex, tai, avgCost, avgRuntime, xChange, yChange, angleChange))
        else:  # imu initialization
            imuResults[key].append((shiftIndex, tai, avgCost, avgRuntime, xChange, yChange, angleChange))
    
    # Sort each group by shift index for proper line plotting
    for key in originResults:
        originResults[key].sort(key=lambda x: x[0])
    for key in imuResults:
        imuResults[key].sort(key=lambda x: x[0])
    
    # Sort shift labels by shift index
    shiftPositions.sort(key=lambda x: x[0])
    shiftLabels = [item[1] for item in shiftPositions]
    
    # Define colors and markers for different w combinations
    colors = ['red', 'blue', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    
    # Figure 1: Origin Initialization Results
    plt.figure(figsize=(12, 8))
    color_idx = 0
    
    for key, data in originResults.items():
        angleW, xyW = key
        shiftIndices = [point[0] for point in data]
        tais = [point[1] for point in data]
        
        label = f'Angle w={angleW}, XY w={xyW}'
        color = colors[color_idx % len(colors)]
        marker = markers[color_idx % len(markers)]
        
        plt.plot(shiftIndices, tais, label=label, color=color, marker=marker, markersize=8, linewidth=2, linestyle='-')
        color_idx += 1
    
    plt.title('Total Average Inaccuracy (TAI) vs Positional Shifts\n(Origin Initialization)', fontsize=16)
    plt.xlabel('Positional Shift (ΔX, ΔY, Δθ)', fontsize=14)
    plt.ylabel('Total Average Inaccuracy (TAI)', fontsize=14)
    plt.xticks(range(len(shiftLabels)), shiftLabels, rotation=45, ha='right')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Figure 2: IMU Initialization Results
    plt.figure(figsize=(12, 8))
    color_idx = 0
    
    for key, data in imuResults.items():
        angleW, xyW = key
        shiftIndices = [point[0] for point in data]
        tais = [point[1] for point in data]
        
        label = f'Angle w={angleW}, XY w={xyW}'
        color = colors[color_idx % len(colors)]
        marker = markers[color_idx % len(markers)]
        
        plt.plot(shiftIndices, tais, label=label, color=color, marker=marker, markersize=8, linewidth=2, linestyle='-')
        color_idx += 1
    
    plt.title('Total Average Inaccuracy (TAI) vs Positional Shifts\n(IMU-based Initialization)', fontsize=16)
    plt.xlabel('Positional Shift (ΔX, ΔY, Δθ)', fontsize=14)
    plt.ylabel('Total Average Inaccuracy (TAI)', fontsize=14)
    plt.xticks(range(len(shiftLabels)), shiftLabels, rotation=45, ha='right')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Figure 3: Comparison of both initialization methods
    plt.figure(figsize=(14, 8))
    
    # Combine data for comparison
    for key in originResults.keys():
        if key in imuResults:
            angleW, xyW = key
            
            # Origin data
            origin_data = originResults[key]
            origin_indices = [point[0] for point in origin_data]
            origin_tais = [point[1] for point in origin_data]
            
            # IMU data
            imu_data = imuResults[key]
            imu_indices = [point[0] for point in imu_data]
            imu_tais = [point[1] for point in imu_data]
            
            color = colors[list(originResults.keys()).index(key) % len(colors)]
            
            plt.plot(origin_indices, origin_tais, label=f'Origin Init - Angle w={angleW}, XY w={xyW}', 
                    color=color, marker='o', markersize=8, linewidth=2, linestyle='-')
            plt.plot(imu_indices, imu_tais, label=f'IMU Init - Angle w={angleW}, XY w={xyW}', 
                    color=color, marker='s', markersize=8, linewidth=2, linestyle='--')
    
    plt.title('Comparison of Initialization Methods\nTotal Average Inaccuracy (TAI) vs Positional Shifts', fontsize=16)
    plt.xlabel('Positional Shift (ΔX, ΔY, Δθ)', fontsize=14)
    plt.ylabel('Total Average Inaccuracy (TAI)', fontsize=14)
    plt.xticks(range(len(shiftLabels)), shiftLabels, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def write_positional_shift_data_to_files(positionalShiftResults, out_dir='testing/grouped_results'):
    """
    Write positional shift data to files in the same format as existing grouped results.
    
    positionalShiftResults: List of tuples (shiftIndex, xChange, yChange, angleChange, angleW, xyW, initType, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Get current date and time string for filenames
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Group by initialization type for JSON format
    grouped_shift_data = {"origin_initialization": {}, "imu_initialization": {}}
    
    for result in positionalShiftResults:
        shiftIndex, xChange, yChange, angleChange, angleW, xyW, initType, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime = result
        
        # Create a key for the parameter combination
        key_str = f"Shift{shiftIndex}_aW{angleW}_xyW{xyW}"
        
        init_key = "origin_initialization" if initType == "origin" else "imu_initialization"
        
        if key_str not in grouped_shift_data[init_key]:
            grouped_shift_data[init_key][key_str] = []
        
        # Format similar to existing grouped results
        entry = {
            "shift_index": shiftIndex,
            "position_shift": [xChange, yChange, angleChange],
            "params": [angleW, xyW],
            "avg_x_inacc": avgXInacc,
            "avg_y_inacc": avgYInacc,
            "avg_angle_inacc": avgAngleInacc,
            "avg_cost": avgCost,
            "avg_runtime": avgRuntime,
            "tai": avgXInacc + avgYInacc + avgAngleInacc
        }
        grouped_shift_data[init_key][key_str].append(entry)

    # Find a unique JSON filename
    json_base = f"positional_shift_results_{dt_str}.json"
    json_path = os.path.join(out_dir, json_base)
    counter = 1
    while os.path.exists(json_path):
        json_path = os.path.join(out_dir, f"positional_shift_results_{dt_str}_{counter}.json")
        counter += 1

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(grouped_shift_data, jf, indent=2)

    # Write a single consolidated CSV file with all results
    csv_base = f"positional_shift_all_results_{dt_str}.csv"
    csv_path = os.path.join(out_dir, csv_base)
    counter = 1
    while os.path.exists(csv_path):
        csv_path = os.path.join(out_dir, f"positional_shift_all_results_{dt_str}_{counter}.csv")
        counter += 1
    
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["shift_index", "x_change", "y_change", "angle_change", "angle_w", "xy_w", "init_type", "avg_x_inacc", "avg_y_inacc", "avg_angle_inacc", "avg_cost", "avg_runtime", "tai"])
        
        # Write all results sorted by shift index, then initialization type, then parameters
        for result in sorted(positionalShiftResults, key=lambda x: (x[0], x[6], x[4], x[5])):
            shiftIndex, xChange, yChange, angleChange, angleW, xyW, initType, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime = result
            tai = avgXInacc + avgYInacc + avgAngleInacc
            writer.writerow([shiftIndex, xChange, yChange, angleChange, angleW, xyW, initType, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime, tai])

    print(f"Positional shift data written to: {out_dir}")
    print(f"JSON summary: {json_path}")
    print(f"Consolidated CSV file: {csv_path}")

def testPositionalShiftComparison(xPositionChanges, 
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
                                 scan2=None):
    """
    Test positional shift accuracy comparison across different position shifts and inertia values.
    Tests both origin initialization and IMU-based initialization (25% of actual shift).
    Position changes are tested by index pairing (first x with first y with first angle, etc.).
    """
    
    positionalShiftResults = []  # List of tuples (shiftIndex, xChange, yChange, angleChange, angleW, xyW, initType, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime)
    testStartTime = time.time()
    
    # Use first values for fixed parameters
    swarmSize = swarmSizes[0]
    maxIter = maxIterations[0]
    sectionCount = sectionCounts[0]
    aC1 = angleC1Vals[0]
    aC2 = angleC2Vals[0]
    xyC1 = xyC1Vals[0]
    xyC2 = xyC2Vals[0]
    
    # Ensure we have the same number of position changes
    min_length = min(len(xPositionChanges), len(yPositionChanges), len(angleChanges))
    if min_length < len(xPositionChanges) or min_length < len(yPositionChanges) or min_length < len(angleChanges):
        print(f"Warning: Position change arrays have different lengths. Using first {min_length} values from each.")
    
    # Create position shift pairs by index
    positionShifts = []
    for i in range(min_length):
        positionShifts.append((i, xPositionChanges[i], yPositionChanges[i], angleChanges[i]))
    
    print(f"Running positional shift comparison test with {len(positionShifts)} position shift combinations:")
    for i, (idx, x, y, a) in enumerate(positionShifts):
        print(f"  Shift {i+1}: (ΔX={x}, ΔY={y}, Δθ={a})")
    
    print(f"Swarm size: {swarmSize}, Max iterations: {maxIter}, Sections: {sectionCount}")
    print(f"Runs per configuration: {numRuns}")
    print("Testing both origin initialization and IMU-based initialization (25% of shift)...")
    
    total_combinations = len(positionShifts) * len(angleWVals) * len(xyWVals) * 2  # 2 for origin + IMU init
    current_combination = 0
    
    for shiftIndex, xChange, yChange, angleChange in positionShifts:
        print(f"\n{'='*80}")
        print(f"TESTING POSITION SHIFT {shiftIndex + 1}: (ΔX={xChange}, ΔY={yChange}, Δθ={angleChange})")
        print(f"{'='*80}")
        
        # Create adjusted scan for this position shift
        adjustedScan = adjustScanPosition(scan1, xChange, yChange, angleChange) if scan2 is None else scan2
        
        for angleW in angleWVals:
            for xyW in xyWVals:
                # Test both initialization methods
                for initType, initDescription in [("origin", "Origin"), ("imu", "IMU-based (25% of shift)")]:
                    current_combination += 1
                    print(f"\nTesting combination {current_combination}/{total_combinations}: Angle w={angleW}, XY w={xyW}, Init={initDescription}")
                    
                    runtimes = []
                    inaccuracies = []
                    costs = []
                    
                    for run in range(numRuns):
                        # Initialize PSO with current parameter combination and initialization type
                        if initType == "origin":
                            # Initialize particles around origin with small noise (-0.5 to 0.5 for X,Y and -1 to 1 for angle)
                            imuX = random.uniform(-0.5, 0.5)
                            imuY = random.uniform(-0.5, 0.5)
                            imuAngle = random.uniform(-1, 1)
                        else:  # IMU-based initialization (25% of actual shift)
                            imuX = random.uniform(xChange - abs(xChange * 0.25), xChange + abs(xChange * 0.25))
                            imuY = random.uniform(yChange - abs(yChange * 0.25), yChange + abs(yChange * 0.25))
                            imuAngle = (random.uniform(angleChange - abs(angleChange * 0.25), angleChange + abs(angleChange * 0.25)) % 360)
                        
                        pso = PSO(swarmSize=swarmSize, w_xy=xyW, c1_xy=xyC1, c2_xy=xyC2,
                                w_angle=angleW, c1_angle=aC1, c2_angle=aC2, 
                                oldLidarScan=scan1, newLidarScan=adjustedScan, angleOffset=0, 
                                sections=sectionCount, 
                                imuXReading=imuX, imuYReading=imuY, imuAngleReading=imuAngle)

                        # Run PSO
                        startTime = time.time()
                        result = pso.runWithIterations(maxIterations=maxIter)
                        endTime = time.time()

                        # Calculate metrics
                        runtime = endTime - startTime
                        runtimes.append(runtime)
                        
                        xAccuracy = abs(result["x"] - xChange)
                        yAccuracy = abs(result["y"] - yChange)
                        
                        # Calculate angle accuracy considering wrap-around at 360 degrees
                        angle_diff = (result["angle"] - angleChange) % 360
                        angleAccuracy = min(angle_diff, 360 - angle_diff)
                        
                        # Convert to percentage inaccuracies
                        xAccuracyPercent = abs(xAccuracy) / max(abs(xChange), 1)
                        yAccuracyPercent = abs(yAccuracy) / max(abs(yChange), 1)
                        angleAccuracyPercent = abs(angleAccuracy) / max(abs(angleChange), 1)
                        
                        inaccuracies.append((xAccuracyPercent, yAccuracyPercent, angleAccuracyPercent))
                        costs.append(result["cost"])
                    
                    # Calculate averages for this parameter combination
                    avgXInacc = np.mean([acc[0] for acc in inaccuracies])
                    avgYInacc = np.mean([acc[1] for acc in inaccuracies])
                    avgAngleInacc = np.mean([acc[2] for acc in inaccuracies])
                    avgCost = np.mean(costs)
                    avgRuntime = np.mean(runtimes)
                    
                    # Store results
                    positionalShiftResults.append((shiftIndex, xChange, yChange, angleChange, angleW, xyW, initType, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime))
                    
                    # Print summary for this combination
                    tai = avgXInacc + avgYInacc + avgAngleInacc
                    print(f"  TAI: {tai:.4f}, Avg Runtime: {avgRuntime:.2f}s, Avg Cost: {avgCost:.2f}")
                    print(f"  Individual inaccuracies - X: {avgXInacc:.4f}, Y: {avgYInacc:.4f}, Angle: {avgAngleInacc:.4f}")
    
    testEndTime = time.time()
    print(f"\nPositional shift comparison test completed in {(testEndTime - testStartTime)/60:.2f} minutes")
    
    # Save results data
    write_positional_shift_data_to_files(positionalShiftResults)
    
    # Generate graphs
    buildPositionalShiftGraphs(positionalShiftResults)
    
    # Print comprehensive summary
    print("\n" + "="*100)
    print("POSITIONAL SHIFT COMPARISON RESULTS SUMMARY")
    print("="*100)
    
    # Group results by initialization type for comparison
    originResults = [r for r in positionalShiftResults if r[6] == "origin"]
    imuResults = [r for r in positionalShiftResults if r[6] == "imu"]
    
    print("\nOrigin Initialization Results:")
    print("-" * 60)
    print("Rank | Shift | AngleW | XYW | TAI      | Runtime | Cost")
    print("-" * 60)
    sortedOrigin = sorted(originResults, key=lambda x: x[7] + x[8] + x[9])  # Sort by TAI
    for i, result in enumerate(sortedOrigin[:10]):
        shiftIndex, xChange, yChange, angleChange, angleW, xyW, initType, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime = result
        tai = avgXInacc + avgYInacc + avgAngleInacc
        print(f"{i+1:4d} | ({xChange:4.0f},{yChange:3.0f},{angleChange:2.0f}) | {angleW:6.1f} | {xyW:3.1f} | {tai:8.4f} | {avgRuntime:7.2f} | {avgCost:6.2f}")
    
    print("\nIMU-based Initialization Results:")
    print("-" * 60)
    print("Rank | Shift | AngleW | XYW | TAI      | Runtime | Cost")
    print("-" * 60)
    sortedIMU = sorted(imuResults, key=lambda x: x[7] + x[8] + x[9])  # Sort by TAI
    for i, result in enumerate(sortedIMU[:10]):
        shiftIndex, xChange, yChange, angleChange, angleW, xyW, initType, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime = result
        tai = avgXInacc + avgYInacc + avgAngleInacc
        print(f"{i+1:4d} | ({xChange:4.0f},{yChange:3.0f},{angleChange:2.0f}) | {angleW:6.1f} | {xyW:3.1f} | {tai:8.4f} | {avgRuntime:7.2f} | {avgCost:6.2f}")
    
    # Compare initialization methods for each position shift
    print("\nComparison Between Initialization Methods (Lower TAI is better):")
    print("-" * 80)
    print("Position Shift | AngleW | XYW | Origin TAI | IMU TAI | Improvement")
    print("-" * 80)
    
    # Group by position shift and parameters for comparison
    comparison_data = {}
    for result in positionalShiftResults:
        shiftIndex, xChange, yChange, angleChange, angleW, xyW, initType, avgXInacc, avgYInacc, avgAngleInacc, avgCost, avgRuntime = result
        key = (shiftIndex, xChange, yChange, angleChange, angleW, xyW)
        tai = avgXInacc + avgYInacc + avgAngleInacc
        
        if key not in comparison_data:
            comparison_data[key] = {}
        comparison_data[key][initType] = tai
    
    for key, init_data in comparison_data.items():
        if "origin" in init_data and "imu" in init_data:
            shiftIndex, xChange, yChange, angleChange, angleW, xyW = key
            origin_tai = init_data["origin"]
            imu_tai = init_data["imu"]
            improvement = ((origin_tai - imu_tai) / origin_tai) * 100 if origin_tai > 0 else 0
            improvement_str = f"{improvement:+6.1f}%" if improvement != 0 else "  0.0%"
            
            print(f"({xChange:4.0f},{yChange:3.0f},{angleChange:2.0f}) | {angleW:6.1f} | {xyW:3.1f} | {origin_tai:10.4f} | {imu_tai:7.4f} | {improvement_str}")
    
    return positionalShiftResults

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

    # Test 4: Accuracy comparison test (TAI vs Swarm Size)
    runChoice = input("Run accuracy comparison test (TAI vs Swarm Size) (y/N)? ")
    if runChoice.lower() == 'y':
        paramFile = input("Enter the parameter file path (default ./testing/accuracyComparison.conf): ") or './testing/accuracyComparison.conf'
        accuracyParams = importParametersFromConfig(configFile=paramFile)
        testAccuracyComparison(*accuracyParams, scan1=scan1)

    # Test 5: Positional shift comparison test (TAI vs Position Shifts with different initialization methods)
    runChoice = input("Run positional shift comparison test (TAI vs Position Shifts) (y/N)? ")
    if runChoice.lower() == 'y':
        paramFile = input("Enter the parameter file path (default ./testing/positionalShiftTest.conf): ") or './testing/positionalShiftTest.conf'
        positionalParams = importParametersFromConfig(configFile=paramFile)
        print("\nNote: This test will run with BOTH initialization methods:")
        print("  1. Origin initialization: Particles initialized around (0,0,0) with small noise")
        print("  2. IMU-based initialization: Particles initialized within 25% of actual position shift")
        testPositionalShiftComparison(*positionalParams, scan1=scan1)

    runChoice = input("Run full parameter sweep tests (y/N)? ")
    if runChoice.lower() != 'y':
        print("Exiting.")
        return
    # Test 1: Stationary and adjusted position scans
    initChoice = input("Initialize particles around origin (y/N)? ")
    originInit = True if initChoice.lower() == 'y' else False
    testPsoAlgorithm(xPositionChanges, yPositionChanges, angleChanges, swarmSizes, maxIterations, angleWVals, angleC1Vals, angleC2Vals, xyWVals, xyC1Vals, xyC2Vals, sectionCounts, numRuns=runs, scan1=scan1, originInitialization=originInit)

if __name__ == "__main__":
    if len(sys.argv) > 1: # Specified file to read from
        importFile = sys.argv[1]
        main(configFile=importFile)
    else:
        main()