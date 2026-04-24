import time
import numpy as np
import multiprocessing as mp
import sys
import psutil  # pip install psutil if needed
from spectral_synthesis_functions import generate_with_korg
from pathlib import Path
import pickle

# Your generate_with_korg function here (with mocks or real Korg)
# ... [paste your exact function] ...

def one_process():
    """One process: 3x generate_with_korg + memory tracking."""
    velocity = np.zeros(56)  # Your input
    proc = psutil.Process()
    
    # Memory before
    # print(f"Process {idx}: Memory before: {mem_before_mb:.1f} MB")
    
    st = time.time()
    call_times = []
    print("Starting")
    t = 0
    for i in range(3):
        call_st = time.time()
        # print("Calling korg")
        _ = generate_with_korg(velocity)
        call_times.append(time.time() - call_st)
        t += 1
        print(t,'th', time.time() - st, 's')
    total_time = time.time() - st
    
    return call_times[1:]


args = sys.argv[1:]
call_times = one_process()
# fname = Path("nprocess_time_251GB.pkl")
# if fname.exists():
#     with open(fname, 'rb') as f:
#         data = pickle.load(f)
# else:
#     data = {}
# data[args[0]] = call_times
# with open(fname, 'wb') as f:
#     pickle.dump(data, f)
        
