import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import time
import os
from collections import defaultdict
import shutil
import csv

from utils import calling_linelist
from utils import call_neid_data
from utils import save_dict_to_pickle

from functools import *

import multiprocessing
import pickle

# from model_functions import residue_scale_factor
from model_functions import residue_velocity_profile
from model_functions import generate_with_korg
from model_functions import generate_profile
from model_functions import shifting_korg_flux
from model_functions import calculate_parameter_errors

from scipy.interpolate import CubicSpline
import juliapkg
from juliacall import Main as jl
jl.seval("using Korg"); Korg=jl.Korg

import sys
import logging





def velprofile_fit_function(neid_filename, wlwinds=(4000, 9000)):
    if neid_filename == 'all':
        basename = 'all'
    else:
        basename = os.path.splitext(neid_filename)[0]
    dead_velocity = -2
    # resultdict = "Result_allgrayline_korgcont_constinterp/Result_directory_witherrcov_" + basename
    # resultdict = os.path.join("Result_algorithm_30epoches", "Result_directory_witherrcov_" + basename)
    resultdict = os.path.join("Result_single_lane_zeroth", "Result_dir_" + basename + "_{}-{}".format(wlwinds[0], wlwinds[1]))
    resultdict = os.path.join("Result_zeroth_withdeadvel4_chi2", "Result_dir_{}_".format(dead_velocity) + basename + "_{}-{}".format(wlwinds[0], wlwinds[1]))
    # resultdict = os.path.join("Result_fakedata", "Result_dir_" + basename + "_{}-{}".format(wlwinds[0], wlwinds[1]))
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
    sys.stdout = StreamToLogger(logging.getLogger(), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger(), logging.ERROR)

    
    result_filename = os.path.join(resultdict, "fitted_params.pkl")
    if os.path.isfile(result_filename):
        print("The results already exists. If you want to make new result, change the name of the directory")
        return
    print("This trial is to generate a fake data")
    maindir = '/home/varghese/Desktop/Stellar_activity_mitigation/NEID_data_onecycle'
    

    print("Resultdict", resultdict)
    shutil.copy("utils.py", resultdict)
    shutil.copy("model_functions.py", resultdict)
    print("Model functions copied sucessfully")
    # shutil.copy("Velocity_profile_chi2.py", resultdict)
    shutil.copy(os.path.abspath(__file__), resultdict)
    print("This code copied sucessfully")
    
    korg_filename = "Korg_data_3800_9000.pkl"
    if os.path.exists(korg_filename):
        print("The data already exists.")
        with open(korg_filename, 'rb') as korgfile:
            korg_data = pickle.load(korgfile)
    else:
        print("Pre-generated data does not exist. Generating now")
        
        korg_data = generate_with_korg(np.zeros(56), (3800, 9000), continuum=True)
        with open(korg_filename, 'wb') as korgfile:
            pickle.dump(korg_data, korgfile)

    cntm_filename = "Ratio_3800_9000.pkl"
    if os.path.exists(cntm_filename):
        print("The continuums already exists.")
        with open(cntm_filename, 'rb') as korgfile:
            ratio_pkl = pickle.load(korgfile)
    else:
        print("Pre-generated data does not exist. Generating now")

        cntm1_data = generate_with_korg(np.zeros(56), (3800, 9000), continuum=True)
        cntm1 = cntm1_data['cont']
        cntm2_data = generate_with_korg(np.zeros(56), (3800, 9000), temp=5570, continuum=True)
        cntm2 = cntm2_data['cont']
        wl = cntm2_data['Wl']
        print('Type', type(cntm1))
        ratio = cntm1/cntm2
        ratio_pkl = {'ratio':ratio, 'Wl':wl}

        with open(cntm_filename, 'wb') as korgfile:
            pickle.dump(ratio_pkl, korgfile)

    

    neid_orders = 122

    neid_data_dict = defaultdict(list)
    if neid_filename == 'all':
        files_list = [f for f in os.listdir(maindir) if f.endswith(".fits")]
        # files_list = os.listdir(maindir)
    else:
        files_list = [neid_filename]
    for order in range(neid_orders):
        flux_order = []
        flux_err = []
        for onefile in files_list:
            # print("Extracting", onefile)
            fullpath = os.path.join(maindir, onefile)
            neid_wl, neid_flux, neid_err = call_neid_data(order, neid_filename=fullpath)
            neid_err = neid_err.astype(np.float64)
            flux_order.append(neid_flux)
            flux_err.append(neid_err)
        # print("Order", order, "Length of flux_order", len(flux_order))
        l = len(flux_order)
        neid_flux = np.sum(flux_order, axis=0)/l
        neid_err = np.sqrt(np.sum(np.array(flux_err)**2, axis=0))/l
        
        # print(order, neid_wl, neid_flux)
        neid_data_dict["Flux"].append(neid_flux)
        neid_data_dict["Wave"].append(neid_wl)
        neid_data_dict["Err"].append(neid_err)

    # print(neid_data_dict)

    # Calling lines
    lines_file_path = '/home/varghese/Desktop/Stellar_activity_mitigation'
    solar_lines_fname = 'Solar_lines_gray.csv' #'sol_line_window_modified.csv' #  # 'sol_line_window_modified.csv'
    line_dict = calling_linelist(os.path.join(lines_file_path, solar_lines_fname))
    shutil.copy(os.path.join(lines_file_path, solar_lines_fname), resultdict)
    # korg_data = generate_with_korg(np.zeros(56))


    # print("simply calling Korg")
    # generate_with_korg(velocity=None)
    print("Start fitting")
    # residue_scale_factor([0,0,0], neid_data_dict, line_dict, korg_data)

    # print(neid_data_dict)
    # neid_filename = '/home/varghese/Desktop/Stellar_activity_mitigation/neidL2_20220110T180951.fits'
    # init_params = [-0.5, 2.5, 0.1]
    # lower_bounds = [-5, 0, 0]
    # upper_bounds = [0, 5, 0.5]
    starttime=time.time()

    v_raise0 = -0.6109816 # fitted_params[0]
    # v_fall0 = 2.31816064 # 2.24984668 # fitted_params[1]
    # scale_factor = 0.1284777954266144 # fitted_params[2]
    scale_factor = 1

    plot_fname = "Fitted_spectra"
    # Fitting for third order
    # init_params3 = np.insert(v_raise2, 0, 0)
    # init_params3 = [0, 0, v_raise0, 0, v_fall0] # Fitting in one step
    init_params3 = np.array([ 0.39390598, -1.05239532, -1.21619457, 1.63296202, 1.67221564])
    err = np.array([0.07054589, 0.11912412, 0.03782152, 0.17998885, 0.04291938])
    init_params3 = init_params3-2*err
    # init_params3 = np.array([7.5005791, -6.85719545, -0.84877234, -0.01646308, 3.04498675,
    #                 0.60185694])[1:]
    
    # lower_bounds3 = np.array([-np.inf, -np.inf, -np.inf, -5, -np.inf, 0])[1:]
    # upper_bounds3 = np.array([np.inf, np.inf, np.inf, 0, np.inf, 5])[1:]

    # To do for single lane one parameter
    lower_bounds = -np.inf
    upper_bounds = np.inf
    init_params3 = init_params3[0]
    # lower_bounds3 = [-np.inf, -np.inf, -np.inf, -5,]
    # upper_bounds3 = [np.inf, np.inf, np.inf, 0]

    residue_vel3 = partial(residue_velocity_profile, neid_data=neid_data_dict,
                           linelist=line_dict,
                           scale_factor=scale_factor,
                           falling_params=0,
                           timer=starttime,
                           result_dict=resultdict,
                           plot_fname=plot_fname,
                           ratio_dict=ratio_pkl,
                           model='onelane',
                           wlwinds=wlwinds)

    ref_value = dead_velocity
    err = 0.2
    trial_vals = np.linspace(ref_value-2*err, ref_value+2*err, 200)
    # trial_vals = np.linspace(-0.7, -0.6, 200)
    # residue_vel3(ref_value) # Trieing only one velocity value. This trial is to generate a fake data.
    chi2_values = []
    for vel in trial_vals:
        residue_array = residue_vel3([vel])
        chi2 = np.sum(residue_array**2)
        chi2_values.append(chi2)
    chi2dict = {"Trail_pts":trial_vals,
                  "Chi2_vals":np.array(chi2_values)}
    save_dict_to_pickle(chi2dict, os.path.join(resultdict, "Chi2_values2.pkl"))
    plt.figure()
    plt.plot(trial_vals, np.array(chi2_values), 'o')
    plt.title("$\chi^2$ profile around CCFRV value")
    plt.xlabel("RV (km/s")
    plt.ylabel("$\chi^2$")
    plt.savefig(os.path.join(resultdict, "Chi2_profile2.pdf"))
    # result3 = least_squares(residue_vel3, init_params3,
    #                         bounds=(lower_bounds, upper_bounds),
    #                         x_scale='jac',
    #                         verbose=2,
    #                         loss='linear')

    # print(result3)
    # fitted_params3 = result3.x
    # # fitted_params3 = [4.93100795e-05, -7.32842107e-05, -1.13824891e+00, 1.32546308e+00, 1.68787459e+00]
    # print("Fitted_params for third order", result3.x)
    # residue_vel3(params=fitted_params3, plot_lines=True)

    # errorvals, cov_matrix = calculate_parameter_errors(result3)
    # resultdictionary = {'params':fitted_params3,
    #                    'params_err': errorvals,
    #                    'cov_matr': cov_matrix}
    
    # save_dict_to_pickle(resultdictionary, result_filename)
    # save_dict_to_pickle(result3, os.path.join(resultdict, "least_squares_op.pkl"))
    # x = np.linspace(0, 1, 56)
    # vel_raise = generate_profile(fitted_params3[:-2], x)
    # vel_fall = generate_profile(fitted_params3[-2:], x)
    # raising_spectra = generate_with_korg(vel_raise)
    # falling_spectra = generate_with_korg(vel_fall)
    # generated_spectra = {"Raise":raising_spectra,
    #                      "Fall":falling_spectra}
    # result_specname = os.path.join(resultdict, "Generated_spectra.pkl")
    # save_dict_to_pickle(generated_spectra, result_specname)

    print("Results saved in", resultdict)


# print("Arguements:", sys.argv)

if len(sys.argv) > 1:
    args = sys.argv[1:]
    fname = args[0] # sys.argv[1]
    if len(args) > 1:
        wl_wind1 = args[1]
        wl_wind2 = args[2]
    else:
        wl_wind1 = 4000
        wl_wind2 = 9000
    # print(fname)
    print("Doing for:", fname)
    velprofile_fit_function(fname, wlwinds=(wl_wind1, wl_wind2))
    # for i, arg in enumerate(sys.argv[1:], start=1):
    #     print(f'Argument {i}: {arg}')
else:
    print("No arguments were provided.")

# neid_list = 'neid_files.csv'
# files_list = []
# with open(neid_list, "r") as file:
#     reader = csv.reader(file)
#     for row in reader:
#         # print(row)  # Each row is a list of values
#         files_list.append((row[1], row[0]))
# files_list = files_list[1:]

# if __name__ == "__main__":
#     with multiprocessing.Pool(processes=4) as pool:
#         results = pool.starmap(velprofile_fit_function, files_list)  # Pass multiple args
#     print(results)  # Output: [3, 7, 11, 15]





