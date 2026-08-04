import os
import sys
import threading
import subprocess
import configparser
import numpy as np
from astropy.io import fits
from pathlib import Path
from collections import defaultdict
import re

def digits_after_T(path):
    match = re.search(r"(\d+)T(\d+)", path.stem)  # Look for T followed by digits
    return list(match.group(1,2)) if match else -1


configfile = 'Spectral_fitting.config'
config = configparser.ConfigParser()
config.read(configfile)
# data_dir = "../NEID_data_set3"
# data_dir = "../NEID_data_onecycle"
data_dir = Path(config['data_dir']['NEID_DIR'])

data_list = data_dir.glob("*.fits")
grouped_data = defaultdict(list)
for i in data_list:
    digits = digits_after_T(i)
    grouped_data[digits[0]].append(digits[1])

# print(len(list(grouped_data.keys())))

selected_epoches = []
for date, time in grouped_data.items():
    # print(date, len(time))
    if len(time) > 1:
        for t in time: # [::10]:  The selection was there when we were running on the data which was not averaged for pmode oscillation.
            regexp = "*" + date + "T" + t + "*"
            # print(regexp)
            for i in data_dir.glob(regexp):
                # print(i)
                selected_epoches.append(i.name)
    else:
        regexp = "*" + date + "T" + time[0] + "*"
        for i in data_dir.glob(regexp):
            selected_epoches.append(i.name)

# print(grouped_data)
# print('selected_epoches', selected_epoches)
# bjd_list = []
# for epoch in selected_epoches:
#     fname =  data_dir / epoch
#     bjd = fits.getheader(fname, ext=12)['CCFJDMOD']
#     bjd_list.append(bjd)
execute_list = []
vels_array = np.array([
    # -0.00020,
    # -0.00015,
    # -0.00010,
    # -0.00005,
    # 0.00000,
    # 0.00005,
    # 0.00010,
    # 0.00015,
    # 0.00020
    ])

vels_array = np.array([
    # -0.1,
    # 0.1,
    # 0.01,
    # 0.001,
    # 0.0001,
    0
    # -0.0005,
    # 0.0005,
    # -0.0002,
    # 0.0002,
    # -0.0001,
    # 0.0001,
    # 0
])
#     -0.00040,
#     -0.00035,
#     -0.00030,
#     -0.00025,
#     0.00025,
#     0.00030,
#     0.00035,
#     0.00040
# vels_array = np.arange(-0.1, 0.11, 0.01)
inits_vals = np.ones(12).reshape(3, 4)
inits_vals[0] *= -1
inits_vals[1] *= 0
P = 2 # days
K = 0.01 # km/s
inits = inits_vals[1]
periods = [
    3,
    5,
    7,
    10,
    15,
    20,
    30
]
amplitudes = [
    0.01,
    0.001,
    0.0001,
    0.0005,
    0.005
]
for P in periods:
    for K in amplitudes:
        for data in selected_epoches:
            # vels = 0 # -0.20002

            for vels in vels_array:
                    print(data, vels)
                    # for inits in inits_vals:
                      #  print(inits)
                    init_str = " ".join(map(str, inits))
                    command = "taskset -c 10-79 python Velocity_fitting_function.py --fname {} --dead_vel={} --planet {} {}".format(data, round(vels, 8), P, K)
                    execute_list.append(command)
# sys.exit()
# Running for each order
# for data in selected_epoches[:10]:
#     # vels = 0 # -0.20002

#     for vels in vels_array:
#             print(data, vels)
#             # for inits in inits_vals:
#               #  print(inits)
#             init_str = " ".join(map(str, inits))
#             for order in range(71, 162):
#                 command = "taskset -c 10-79 python Velocity_fitting_function.py --fname {} --init {} --dead_vel={} --orders {}".format(data, init_str, round(vels, 8), order)
#                 execute_list.append(command)

def run_sequentially(cmd_list, label):
    for cmd in cmd_list:
        print(f"[{label}] Running: {cmd}")
        subprocess.run(cmd, shell=True)
        print(f"[{label}] Finished: {cmd}")

n_threads = 70
n_batches = 1 + len(execute_list) // n_threads
batches = [execute_list[i:i + n_batches] for i in range(0, len(execute_list), n_batches)]
print("Batches", len(batches))
print(batches)
# print(batches)

count = sum(len(sublist) for sublist in batches)
print(count)


threads = []
for i, batch in enumerate(batches):
    thread = threading.Thread(target=run_sequentially, args=(batch, f"Seq {i+1}"))
    threads.append(thread)

# Start all threads in parallel
for thread in threads:
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

print("All commands finished.")
