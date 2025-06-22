import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import time
import os
from collections import defaultdict
import shutil
import csv

from utils import calling_linelist
# from utils import call_neid_data, call_neiddata_full
from utils import call_neiddata_full
from utils import generate_fakedata
from utils import save_dict_to_pickle

from functools import *

import multiprocessing
import pickle

# from model_functions import residue_scale_factor
# from model_functions import residue_velocity_profile
from model_functions import generate_with_korg
from model_functions import generate_profile
from model_functions import shifting_korg_flux
from model_functions import calculate_parameter_errors
from model_functions import generate_full_korgspectra
from model_functions import residue_profile

from scipy.interpolate import CubicSpline
import juliapkg
from juliacall import Main as jl
jl.seval("using Korg"); Korg=jl.Korg

import sys
import argparse
import logging
import configparser




def velprofile_fit_function(neid_filename, configfile, save_generated_syntspectra=False, dead_velocity=0,
                            snr=500, purpose='both', logfilesave='T'):
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
    
    if purpose=='debug':
        master_subdir += '_debug'

    resultsubdir_prefix = config['output_dir']['RESDIR_PREFIX'] # 'Result_dir_actualway_'
    wlwinds = config['inputs']['WL_WINT'].strip().split(', ')
        
    wlwinds = (int(wlwinds[0]), int(wlwinds[1]))
    if basename == 'Korg':
        resultsubdir_prefix += "_{}_snr{}_".format(round(dead_velocity, 7), snr)
    resultdict = os.path.join(srcdir, master_resdir, master_subdir, resultsubdir_prefix + basename + "_{}-{}".format(wlwinds[0], wlwinds[1]))
        # print(fullpath)
    if not os.path.exists(resultdict):
        os.makedirs(resultdict)
        print(resultdict, "Directory created successfully!")
    else:
        print(resultdict, "Directory already exists!")


    logging.basicConfig(filename=resultdict+"/least_squares.log", level=logging.INFO, format="%(asctime)s - %(message)s")


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
    if logfilesave=='T':
        sys.stdout = StreamToLogger(logging.getLogger(), logging.INFO)
        sys.stderr = StreamToLogger(logging.getLogger(), logging.ERROR)

    
    result_filename = os.path.join(resultdict, "fitted_params.pkl")
    # if os.path.isfile(result_filename):
    #     print("The results already exists. If you want to make new result, change the name of the directory")
    #     return
    maindir = config['data_dir']['NEID_DIR'] 
    korg_data_name = os.path.join("data", config['inputs']['REF_SPEC'])
    if os.path.isfile(korg_data_name):
        print("The reference Korg data already exists. Calling that one")
        with open(korg_data_name, 'rb') as korgspec:
            korg_data_ref = pickle.load(korgspec)
    else:
        print("The reference Korg data does not exist. Generating one.")
        korg_data_ref =generate_full_korgspectra(velocity=False)
        save_dict_to_pickle(korg_data_ref, korg_data_name)
        
    print("Resultdict", resultdict)
    shutil.copy("utils.py", resultdict)
    shutil.copy("model_functions.py", resultdict)
    shutil.copy(configfile, resultdict)
    print("Model functions copied sucessfully")
    shutil.copy("Velocity_fitting_function_fakedata.py", resultdict)
    print("This code copied sucessfully")
    


    # cntm_filename = "data/Ratio_3800_9000.pkl"
    # if os.path.exists(cntm_filename):
    #     print("The continuums already exists.")
    #     with open(cntm_filename, 'rb') as korgfile:
    #         ratio_pkl = pickle.load(korgfile)
    # else:
    #     print("Pre-generated data does not exist. Generating now")

    #     cntm1_data = generate_with_korg(np.zeros(56), (3800, 9000), continuum=True)
    #     cntm1 = cntm1_data['cont']
    #     cntm2_data = generate_with_korg(np.zeros(56), (3800, 9000), temp=5570, continuum=True)
    #     cntm2 = cntm2_data['cont']
    #     wl = cntm2_data['Wl']
    #     print('Type', type(cntm1))
    #     ratio = cntm1/cntm2
    #     ratio_pkl = {'ratio':ratio, 'Wl':wl}

    #     with open(cntm_filename, 'wb') as korgfile:
    #         pickle.dump(ratio_pkl, korgfile)

    

    neid_orders = 115
    #km/s. This is the additional velocity adding to neid spectra.
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
            neid_data_dict = generate_fakedata([dead_velocity], snr, fullpath, resultdict)
        else:
            fullpath = os.path.join(maindir, onefile)
            neid_data_dict = call_neiddata_full(fullpath, resultdict, refspec=korg_data_ref)
    # print(neid_data_dict)


    # Calling lines
    lines_file_path = '/home/varghese/Desktop/Stellar_activity_mitigation'
    solar_lines_fname ='FullNeidRange.csv' # 'Solar_lines_gray.csv' #'sol_line_window_modified.csv' #  # 'sol_line_window_modified.csv'
    line_dict = calling_linelist(os.path.join(lines_file_path, solar_lines_fname))
    shutil.copy(os.path.join(lines_file_path, solar_lines_fname), resultdict)
    # korg_data = generate_with_korg(np.zeros(56))

    
    # print("simply calling Korg")
    # generate_with_korg(velocity=None)

    # residue_scale_factor([0,0,0], neid_data_dict, line_dict, korg_data)
    starttime=time.time()

    v_raise0 = -0.6109816 # fitted_params[0]
    scale_factor = 1

    plot_fname = config['outputs']['SPEC_PRIFIX'] 
    # To do for single lane one parameter
    # init_params3 = [0, 0, -0.2] # float(config['fit_init']['INIT_PARAMS'])
    # lower_bounds = [-np.inf, -np.inf, -5]
    # upper_bounds = [np.inf, np.inf, 5]
    init_params3 = [-0.2, 0.1]
    lower_bounds = [-5, 0]
    upper_bounds = [0, 5]
    # param_pos = np.array(['r','r','r']) # Mask for one lane model
    param_pos = np.array(['r','f'])#,'a'])
    residue_vel = partial(residue_profile, neid_data=neid_data_dict, synt_spectra=None,
                          scale_fact=0.47,
                          param_pos=param_pos) # korg_data_ref)
    if (purpose=='minimize') or (purpose=='both'):
        if  not os.path.isfile(result_filename):
            print("Start fitting")
            result = least_squares(residue_vel, init_params3,
                                   bounds=(lower_bounds, upper_bounds),
                                   x_scale='jac',
                                   jac='3-point',
                                   verbose=2,
                                   ftol=None,
                                   method='trf',
                                   loss='soft_l1',
                                   max_nfev=1000)
            print(result)
            
            fitted_params3 = result.x
            # fitted_params3 = [4.93100795e-05, -7.32842107e-05, -1.13824891e+00, 1.32546308e+00, 1.68787459e+00]
            print("Fitted_params for third order", result.x)
            # residue_vel3(params=fitted_params3, plot_lines=True, save_interactive_plot=False)
            
            errorvals, cov_matrix = calculate_parameter_errors(result)
            resultdictionary = {'params':fitted_params3,
                                'params_err': errorvals,
                                'cov_matr': cov_matrix}
            
            save_dict_to_pickle(resultdictionary, result_filename)
            save_dict_to_pickle(result, os.path.join(resultdict, "least_squares_op.pkl"))

        else:
            print("The result already exist. Calling it to plot")
            with open(result_filename, 'rb') as opfile:
                fitted_params3 = pickle.load(opfile)['params']
        korg_spectra = residue_vel(fitted_params3, required='spectra')

        from utils import plotting_spectra
        plotting_spectra(neid_data_dict, korg_spectra, resultdict+"/Fitted_spectra.pdf")
        if purpose == 'both':
            vel_array = np.linspace(fitted_params3-2*errorvals, fitted_params3+2*errorvals, 200)
            chi2_array = np.array([])
            for vel in vel_array:
                residue_array = residue_vel([vel[0]])
                chi2_val = np.sum(residue_array**2)
                red_chi2_val = np.array([chi2_val/(np.size(residue_array)-1)])
                chi2_array = np.concatenate((chi2_array, red_chi2_val))
            plt.figure(figsize=(16, 9))
            plt.plot(vel_array, chi2_array)
            plt.xlabel("Velocities (km/s)")
            plt.ylabel("Reduced chi2")
            plt.savefig(os.path.join(resultdict, "Chi2_plot.pdf"))

        if save_generated_syntspectra:
            x = np.linspace(0, 1, 56)
            vel_raise = generate_profile(fitted_params3, x)

            print(vel_raise)
            raising_spectra = generate_with_korg(vel_raise)

            generated_spectra = {"Raise":raising_spectra}

            result_specname = os.path.join(resultdict, "Generated_spectra.pkl")
            save_dict_to_pickle(generated_spectra, result_specname)
            
        print("Results saved in", resultdict)

    elif purpose=='chi2':
        vel_array = np.linspace(dead_velocity-0.002, dead_velocity+0.002, 200)
        chi2_array = np.array([])
        for vel in vel_array:
            residue_array = residue_vel([vel])
            chi2_val = np.sum(residue_array**2)
            red_chi2_val = np.array([chi2_val / (np.size(residue_array)-1)])
            chi2_array = np.concatenate((chi2_array, red_chi2_val))
        plt.figure(figsize=(16, 9))
        plt.plot(vel_array, chi2_array)
        plt.xlabel("Velocities (km/s)")
        plt.ylabel("Reduced chi2")
        plt.savefig(os.path.join(resultdict, "Chi2_plot.pdf"))

# print("Arguements:", sys.argv)


parser = argparse.ArgumentParser(description="Run velprofile fitting.")

# Required positional argument
parser.add_argument('fname', type=str, help='Input file name')

# Config file
parser.add_argument('--config', type=str, default='Spectral_fitting.config', help='Config file name')
# --WL takes 2 values
# parser.add_argument('--WL', type=float, nargs=2, metavar=('wl_wind1', 'wl_wind2'), default=[3700, 9000], help='Wavelength window (start end)')

# --SNR is optional here, but we will enforce it manually later if fname=="Korg"
parser.add_argument('--SNR', type=float, help='Signal-to-noise ratio (only needed if fname is Korg)')

parser.add_argument('--LOG', type=str, help='F to display the outputs.', default='T')

# --dead_vel
parser.add_argument('--dead_vel', type=float, default=0.0, help='Dead velocity')

parser.add_argument('--purpose', type=str, default='minimize', help='Purpose. (minimize, chi2, both)')

# Parse args
args = parser.parse_args()
print(args)
# Unpack WL window
# wl_wind1, wl_wind2 = args.WL

# print(config['data_dir']['NEID_DIR'])


# Check if SNR is needed
if args.fname == "Korg" and args.SNR is None:
    print("Error: --SNR must be provided when fname is 'Korg'.")
    sys.exit(1)

# Either use given SNR or a dummy value if not provided (only for non-Korg cases)
snr_value = args.SNR if args.SNR is not None else 500  # You can also choose not to pass it at all if not needed

# Print info
print("Doing for:", args.fname)
# print(f"Wavelength window: {wl_wind1} - {wl_wind2}")
print(f"Dead velocity: {args.dead_vel}")
if args.fname == "Korg":
    print(f"SNR: {args.SNR}")
print(args.LOG)
# Call your function
velprofile_fit_function(
    args.fname,
    args.config,
    dead_velocity=args.dead_vel,
    snr=snr_value,  # Only needed for Korg, but passing anyway
    logfilesave=args.LOG,
    purpose=args.purpose
)


'''
if len(sys.argv) > 1:
    args = sys.argv[1:]
    fname = args[0] # sys.argv[1]
    if len(args) > 1:
        wl_wind1 = args[1]
        wl_wind2 = args[2]
        snr = float(args[3])
        dead_vel = float(args[4])
    else:
        wl_wind1 = 4000
        wl_wind2 = 9000
        snr = 500
        dead_vel = -0.1
    # print(fname)
    print("Doing for:", fname)
    velprofile_fit_function(fname, wlwinds=(wl_wind1, wl_wind2), dead_velocity=dead_vel, snr=snr)
    # for i, arg in enumerate(sys.argv[1:], start=1):
    #     print(f'Argument {i}: {arg}')
else:
    print("No arguments were provided.")
'''
print('\a')




