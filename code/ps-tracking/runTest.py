import pandas as pd
import itertools
import time
import math
import csv
import json
from consts import LIDAR_PWM_TO_TIME

# Load parameters from CSV file
parameters_file = "parameters.csv"
parameters_df = pd.read_csv(parameters_file)

# Validate parameters file structure
if not all(["Parameter" in parameters_df.columns, "ValueList" in parameters_df.columns]):
    raise ValueError("Invalid structure in parameters.csv")

# Convert parameters to a dictionary of lists
parameters = {row["Parameter"]: list(map(float, row["ValueList"].split(','))) for _, row in parameters_df.iterrows()}

# Generate all combinations of parameters
parameter_combinations = list(itertools.product(*parameters.values()))

# Map combinations to dictionaries
parameter_dicts = [dict(zip(parameters.keys(), combination)) for combination in parameter_combinations]

target_time_cache = {}

def run_tracker(params):
    from PSTracker import PSTracker
    import math
    from consts import LIDAR_PWM_TO_TIME

    if params['motorPWM'] not in LIDAR_PWM_TO_TIME:
        raise ValueError(f"Invalid motorPWM value: {params['motorPWM']}")

    # Cache calculated targetTime
    if params['motorPWM'] not in target_time_cache:
        freq = LIDAR_PWM_TO_TIME[params['motorPWM']]
        target_time_cache[params['motorPWM']] = 1 / (freq * 1.15)

    # Use cached targetTime
    target_time = target_time_cache[params['motorPWM']]

    tracker = PSTracker(
        swarmSize=params['swarmSize'],
        w=params['w'],
        c1=params['c1'],
        c2=params['c2'],
        xNoise=params['xNoise'],
        yNoise=params['yNoise'],
        angleNoise=params['angleNoise'],
        sections=params['segments'],
        motorPWM=params['motorPWM'],
        targetTime=target_time
    )

    x, y, angle, avg_iterations, avg_cost = tracker.start(testing=True, duration=30, useOriginScan=True)
    distance = math.sqrt(x**2 + y**2)

    return {
        'parameters': params,
        'distance': distance,
        'avg_iterations': avg_iterations,
        'avg_cost': avg_cost
    }

# Sequential testing process
results = []
for params in parameter_dicts:
    results.append(run_tracker(params))

# Save results to JSON file
results_file = "results.json"
with open(results_file, mode='w') as file:
    json.dump(results, file, indent=4)

# Find optimal parameters
optimal_result = min(results, key=lambda x: x['distance'])

print("Optimal Parameters:")
print(optimal_result['parameters'])
print(f"Distance: {optimal_result['distance']:.2f}")
print(f"Average Iterations: {optimal_result['avg_iterations']:.2f}")
print(f"Average Cost: {optimal_result['avg_cost']:.2f}")

# Log optimal parameters to a file
optimal_result_file = "optimal_parameters.json"
with open(optimal_result_file, mode='w') as file:
    json.dump(optimal_result, file, indent=4)

