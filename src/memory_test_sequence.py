import os
import threading
import subprocess
import configparser
import numpy as np
from pathlib import Path
from collections import defaultdict
import re


for i in range(70, 75, 5):
    execute_list = []
    for j in range(i):
        command = "python memory_time_test.py {}".format(i)
        execute_list.append(command)

    def run_cmd(cmd, label):
        print(f"[{label}] Running: {cmd}")
        subprocess.run(cmd, shell=True)
        print(f"[{label}] Finished: {cmd}")

    threads = []
    for i, cmd in enumerate(execute_list):
        thread = threading.Thread(target=run_cmd, args=(cmd, f"T {i+1}"))
        threads.append(thread)

    # Start all threads in parallel
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All commands finished.")


