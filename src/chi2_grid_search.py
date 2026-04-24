import sys
import argparse
import logging
import configparser

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import time
import os
from collections import defaultdict
import shutil

from utils import calling_linelist
# from utils import call_neid_data, call_neiddata_full
from utils import call_neiddata_full
from utils import generate_fakedata
from utils import save_dict_to_pickle
from utils import save_synt_data
from utils import inject_vel_neidata
from pathlib import Path
from astropy.table import Table

from functools import partial

import pickle

# from model_functions import residue_scale_factor
# from model_functions import residue_velocity_profile
from model_functions import generate_with_korg
from model_functions import generate_profile
from model_functions import calculate_parameter_errors
from model_functions import generate_full_korgspectra
from model_functions import residue_profile
from model_functions import residue_ccf
from model_functions import reconstruct_params
from scipy.interpolate import CubicSpline
from juliacall import Main as jl
from multiprocessing import Pool
jl.seval("using Korg")
Korg = jl.Korg


def velprofile_fit_function(params, neid_filename, configfile,
                            save_generated_syntspectra=False, dead_velocity=0,
                            snr=500, purpose='both', logfilesave='T',
                            ip_data='epoch',
                            fitting='spectra',
                            save_ip=False):
    config = configparser.ConfigParser()
    config.read(configfile)
    if neid_filename == 'all':
        basename = 'all'
    elif neid_filename == 'Korg':
        basename = 'Korg'
    else:
        basename = os.path.splitext(neid_filename)[0]

    script_path = os.path.abspath(__file__)
    srcdir = os.path.dirname(os.path.dirname(script_path))

    master_resdir = config['output_dir']['OP_MAIN']
    master_subdir = config['output_dir']['OP_SUB']

    if purpose == 'debug':
        master_subdir += '_debug'

    resultsubdir_prefix = config['output_dir']['RESDIR_PREFIX']
    wlwinds = config['inputs']['WL_WINT'].strip().split(', ')

    wlwinds = (int(wlwinds[0]), int(wlwinds[1]))
    if basename == 'Korg':
        resultsubdir_prefix += "_{}_snr{}_".format(round(dead_velocity, 7),
                                                   snr)
    if abs(dead_velocity) != 0:
        master_subdir += "_{}".format(round(dead_velocity, 7))
    resultdict = os.path.join(srcdir, master_resdir, master_subdir,
                              resultsubdir_prefix + basename + "_{}-{}".format(
                                  wlwinds[0], wlwinds[1]))
    # print(fullpath)
    if not os.path.exists(resultdict):
        os.makedirs(resultdict)
        print(resultdict, "Directory created successfully!")
    else:
        print(resultdict, "Directory already exists!")

    logging.basicConfig(filename=resultdict+"/least_squares.log",
                        level=logging.INFO, format="%(asctime)s - %(message)s")

    class StreamToLogger:
        """Redirects stdout/stderr to logging"""
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level
            self.buffer = ""

        def write(self, message):
            if message.strip():  # Avoid empty lines
                self.logger.log(self.level, message.strip())

        def flush(self):
            pass  # No need to flush for logging

    # Redirect stdout and stderr
    if logfilesave == 'T':
        sys.stdout = StreamToLogger(logging.getLogger(), logging.INFO)
        sys.stderr = StreamToLogger(logging.getLogger(), logging.ERROR)

    result_filename = os.path.join(resultdict, "fitted_params.pkl")

    maindir = config['data_dir']['NEID_DIR']
    korg_data_name = os.path.join("data", config['inputs']['REF_SPEC'])
    result_files_list = ['fitted_params.pkl', 'Data_CCF.fits', 'Synt_CCF.fits']
    truth_list = np.array([os.path.isfile(os.path.join(resultdict, i)) for i in result_files_list])
    # print("Truth_list", truth_list)
    if np.sum(truth_list) == len(result_files_list):
        #print("All required result files exist already")
        sys.exit()
    if os.path.isfile(korg_data_name):
        print("The reference Korg data already exists. Calling that one")
        with open(korg_data_name, 'rb') as korgspec:
            korg_data_ref = pickle.load(korgspec)
    else:
        print("The reference Korg data does not exist. Generating one.")
        korg_data_ref = generate_full_korgspectra(velocity=False)
        save_dict_to_pickle(korg_data_ref, korg_data_name)

    print("Resultdict", resultdict)
    shutil.copy("utils.py", resultdict)
    shutil.copy("model_functions.py", resultdict)
    shutil.copy(configfile, resultdict)
    print("Model functions copied sucessfully")
    shutil.copy("Velocity_fitting_function.py", resultdict)
    print("This code copied sucessfully")

    # km/s. This is the additional velocity adding to neid spectra.
    neid_data_dict = defaultdict(list)
    if neid_filename == 'all':
        files_list = [f for f in os.listdir(maindir) if f.endswith(".fits")]
        # files_list = os.listdir(maindir)
    else:
        files_list = [neid_filename]
    for onefile in files_list:
        if onefile == "Korg":
            reference_file = "neidL2_20220514T172101.fits"
            fullpath = os.path.join(maindir, reference_file)
            neid_data_dict = generate_fakedata([dead_velocity], snr, fullpath,
                                               resultdict)
        else:
            fullpath = os.path.join(maindir, onefile)
            # print("Dead velocity {}".format(dead_velocity))
            # print(fullpath)
            shutil.copy(fullpath, resultdict)
            print(fullpath, "copied to", resultdict)
            fullpath = os.path.join(resultdict, neid_filename)
            inject_vel_neidata(fullpath, dead_velocity)
            dead_velocity = 0
            print("Using the file", fullpath)
            neid_data_dict = call_neiddata_full(fullpath, resultdict,
                                                refspec=korg_data_ref,
                                                save_interactive_plots=save_ip,
                                                ref_velocity=dead_velocity)

    # Calling lines
    lines_file_path = '/home/varghese/Desktop/Stellar_activity_mitigation'
    solar_lines_fname = 'FullNeidRange.csv'
    # line_dict = calling_linelist(os.path.join(lines_file_path,
    # solar_lines_fname))
    shutil.copy(os.path.join(lines_file_path, solar_lines_fname), resultdict)
    # korg_data = generate_with_korg(np.zeros(56))

    # plot_fname = config['outputs']['SPEC_PRIFIX']

    # Initial conditions and bounds
    # init_params3 = [0.649, -1.287, -1.16, 0.03]
    # init_params3 = np.array([-9.92452535e-01,  1.82029738e+00,  8.31773945e-01,  6.64431737e-05])
    lower_bounds = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]# -5]#, -1]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf] #1]#]

    # param_pos = np.array(['r', 'r', 'r', 'v'])
    # param_pos = np.array(['p', 'p', 'p', 'p', 'v'])
    param_pos = np.array(['p', 'p', 'p', 'v'])
    pca = False
    residue_vel = partial(residue_profile, neid_data=neid_data_dict,
                          synt_spectra=None,
                          area_fact=0.5,
                          scale_fact=-1,
                          pca_comp=pca,
                          param_pos=param_pos,
                          ip_data=ip_data,
                          required='spectra')
    result = residue_vel(params)
    residue = result['Res']
    residue_wl = result['Wl']
    wl_orders = neid_data_dict['Wave']
    residue_list = []
    for order, wl_ar in wl_orders.items():
        # print(residue_wl)
        # print(wl_ar)
        # order_residue = CubicSpline(residue_wl, residue)(wl_ar)
        order_wl_mask = np.isin(residue_wl, wl_ar)
        residue_order = residue[order_wl_mask]
        chi2_ord = np.sum(residue_order**2)
        residue_list.append(chi2_ord)
    print("====================== The End ===============================")
    return np.asarray(residue_list, dtype=float), wl_orders.keys()
# print("Arguements:", sys.argv)

def worker_wrapper(params,
                   neid_filename,
                   configfile,
                   dead_velocity,
                   snr,
                   purpose,
                   logfilesave,
                   fitting,
                   save_ip,
                   nparray_name):
    if nparray_name.exists():
        result_array = np.load(nparray_name)
        params_vals = result_array[:, :4]
        mask = params_vals == params
        if np.sum(mask) == 4:
            print("This one already done")
            return
    res, _ = velprofile_fit_function(
        params,
        neid_filename,
        configfile,
        dead_velocity,
        snr,
        purpose,
        logfilesave,
        fitting,
        save_ip,
    )
    config = configparser.ConfigParser()
    config.read(configfile)
    master_resdir = config['output_dir']['OP_MAIN']
    master_subdir = config['output_dir']['OP_SUB']
    if neid_filename == 'all':
        basename = 'all'
    elif neid_filename == 'Korg':
        basename = 'Korg'
    else:
        basename = os.path.splitext(neid_filename)[0]

    if abs(dead_velocity) != 0:
        master_subdir += "_{}".format(round(dead_velocity, 7))
    resultdict = os.path.join(srcdir, master_resdir, master_subdir,
                              resultsubdir_prefix + basename + "_{}-{}".format(
                                  wlwinds[0], wlwinds[1]))

    res = np.concatenate((params, res))
    if nparray_name.exists():
        res_array = np.load(nparray_name)
        res_array = np.vstack((res_array, res))
    else:
        res_array = np.array([res])
    nparray_name = os.path.join(resultdict, nparray_name)
    np.save(nparray_name, res_array)
    # return params, res
parser = argparse.ArgumentParser(description="Run velprofile fitting.")
ref_fname = "neidL2_20220402T173047.fits"
# Required positional argument
parser.add_argument('--fname', type=str,
                    default=ref_fname, help='Input file name')

# Config file
parser.add_argument('--config', type=str,
                    default='Spectral_fitting_chi2.config', help='Config file name')

# OUTPUT_FILE = "residue_grid.txt"
OUTPUT_FILE = "residue_grid.npy"

def init_output_file(n_orders):
    if not os.path.exists(OUTPUT_FILE):

        header_params = ["p2", "p1", "p0", "dC"][:N_PARAMS]
        header_chi = [f"ch{i+1}" for i in n_orders]

        header = header_params + header_chi

        with open(OUTPUT_FILE, "w") as f:
            f.write(" ".join(header) + "\n")
def append_result(params, residue_array):
    with open(OUTPUT_FILE, "a") as f:
        line = " ".join(map(str, np.atleast_1d(params)))
        line += " " + " ".join(map(str, residue_array))
        f.write(line + "\n")

def load_completed_params(filename):
    completed = set()

    if not Path(filename).exists():
        return completed

    with open(filename) as f:
        next(f)  # ← skip header line

        for line in f:
            parts = line.strip().split()
            if len(parts) < N_PARAMS:
                continue

            param_tuple = tuple(map(float, parts[:N_PARAMS]))
            completed.add(param_tuple)

    return completed
# --WL takes 2 values


# --SNR is optional here, but we will enforce it manually later
# if fname=="Korg"
parser.add_argument('--SNR', type=float,
                    help='Signal-to-noise ratio (only needed if fname is Korg)')

parser.add_argument('--LOG', type=str,
                    help='F to display the outputs.', default='T')

# --dead_vel
parser.add_argument('--dead_vel', type=float,
                    default=0.0, help='Dead velocity')

parser.add_argument('--purpose', type=str,
                    default='minimize', help='Purpose. (minimize, chi2, both)')
parser.add_argument('--fitting', type=str,
                    default='spectra',
                    help="Fitting for: (spectra, ccf)")
parser.add_argument('--ip_data', type=str,
                    default='epoch', help='epoch, combined')
parser.add_argument('--save_ip', type=str,
                    default='F', help='Save interactive plot (T, F)')
# Parse args
args = parser.parse_args()

# ================= CONFIG =================
OUTPUT_FILE = "residue_grid.npy"
N_PARAMS = 4
FLUSH_EVERY = 20
N_CORES = 70
CHUNKSIZE = 2
# =========================================


# ---------- completed set loader (NUMPY version) ----------
def load_completed_params_npy(filename):
    completed = set()

    if not os.path.exists(filename):
        return completed

    arr = np.load(filename)

    if arr.size == 0:
        return completed

    params_part = arr[:, :N_PARAMS]

    for row in params_part:
        completed.add(tuple(np.round(row.astype(float), 10)))

    return completed


# ---------- chunk saver ----------
def save_chunk(buffer):
    new_block = np.vstack(buffer).astype(np.float64)

    if os.path.exists(OUTPUT_FILE):
        old = np.load(OUTPUT_FILE)
        combined = np.vstack([old, new_block])
    else:
        combined = new_block

    np.save(OUTPUT_FILE, combined)


# ---------- filter ----------
def not_done(p):
    return tuple(np.round(np.atleast_1d(p), 10)) not in completed


# ================= PREP =================

completed = load_completed_params_npy(OUTPUT_FILE)

params_grid = np.array([
    [-9.92452535e-01, 1.82029738e+00, 8.31773945e-01, 6.64431737e-05]
])

param_grid_filtered = filter(not_done, params_grid)
snr_value = 100
save_ip = False
worker = partial(
    worker_wrapper,
    neid_filename=args.fname,
    configfile=args.config,
    dead_velocity=args.dead_vel,
    snr=snr_value,
    purpose=args.purpose,
    logfilesave=args.LOG,
    fitting=args.fitting,
    save_ip=save_ip,
    nparray_name=Path("chi2_vals.npy")
)
params = np.array([0, 0, 0, 0])
worker(params)
# This is to do the grid search
# for p2 in np.linspace(-9.92452534e-01, -9.92452536e-01, 20):
#     for p1 in np.linspace(1.82029737e+00, 1.82029739e+00, 20):
#         for p0 in np.linspace(8.31773944e-01, 8.31773946e-01, 20):
#             for dC in np.linspace(6.64431736e-05, 6.64431738e-05, 20):
#                 params = np.array([p2, p1, p0, dC])
#                 worker(params)
