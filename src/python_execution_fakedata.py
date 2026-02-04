import numpy as np
import os
import threading
import subprocess

# data_dir = "../NEID_data_set3"
data_dir = "../NEID_data_onecycle"

# data_list = # os.listdir(data_dir)

# print(data_list)
# dead_vels = np.arange(-1., 1., 0.1)
# dead_vels = np.arange(0.15, 0.25, 0.01)
dead_vels = np.arange(-0.005, 0.005, 0.0001)
snrs_list = [10, 50, 100, 300, 500][-2:]
execute_list = []
for snr in snrs_list:
    for vel in dead_vels:
        command = "taskset -c 0-39 python Velocity_fitting_function_fakedata.py Korg --SNR {} --dead_vel {}".format(snr, round(vel, 6))
        execute_list.append(command)


def run_sequentially(cmd_list, label):
    for cmd in cmd_list:
        print(f"[{label}] Running: {cmd}")
        subprocess.run(cmd, shell=True)
        print(f"[{label}] Finished: {cmd}")

n_cores = 5
print(n_cores, execute_list)
n_batches = len(execute_list) // n_cores # 4

batches = [execute_list[i:i + n_batches] for i in range(0, len(execute_list), n_batches)]
print(len(batches))
print(batches)

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


