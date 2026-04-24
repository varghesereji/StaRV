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
from concurrent.futures import ProcessPoolExecutor

from utils import calling_linelist
# from utils import call_neid_data, call_neiddata_full
from utils import call_neiddata_full
from utils import generate_fakedata
from utils import save_dict_to_pickle
from utils import save_synt_data
from utils import inject_vel_neidata

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
from scipy.optimize import differential_evolution

from juliacall import Main as jl
jl.seval("using Korg")
Korg = jl.Korg


def make_ref_residue(reference_params, datadir, korg_data_ref, opdir='data/Reference_dir'):
    print("Calling reference data")
    fname = reference_params['fname']
    fullpath = os.path.join(datadir, fname)
    neid_data_dict = call_neiddata_full(fullpath, opdir, refspec=korg_data_ref)
    params = reference_params['params']
    # print(neid_data_dict)
    param_pos = np.array(['r', 'r', 'r', 'v'])
    residue_dict = residue_profile(params, neid_data=neid_data_dict,
                                   synt_spectra=None,
                                   area_fact=0.5,
                                   scale_fact=-1,
                                   pca_comp=False,
                                   param_pos=param_pos,
                                   profile='parabola',
                                   required='spectra')
    print(residue_dict)
    ref_res = {}
    ref_res['residue'] = residue_dict['Res_noerr']
    ref_res['Wl'] = residue_dict['Wl']
    ref_res['err'] = residue_dict['err']
    ref_res['data'] = residue_dict['data']
    ref_res['synt'] = residue_dict['Total']
    print(ref_res)
    with open(os.path.join('data', "Reference_residue.pkl"), 'wb') as res:
        pickle.dump(ref_res, res)
    return ref_res

def velprofile_fit_function(neid_filename, configfile,
                            save_generated_syntspectra=False, dead_velocity=0,
                            snr=500, purpose='both', logfilesave='T',
                            order='F',
                            ip_data='epoch',
                            fitting='spectra',
                            algorithm='ls',
                            save_ip=False,
                            reference_params=None,
                            inits=[0.0, 0.0, 0.0, 0.0]):
    '''
    algorithm: ls for least squares, diff for differential evaluation
    reference params: dict. {fname: Fname of reference file, params: Reference parameters}
    or {residue: reference residue, err: error of reference file, Wl: wavelength of reference residue}
    '''
    if order == 'F':
        order = None
    else:
        order = int(order)
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
    if order is not None:
        resultdict = os.path.join(resultdict, "Order_{}".format(order))
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
    # if np.sum(truth_list) == len(result_files_list):
    #     #print("All required result files exist already")
    #     sys.exit()
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
        # files_list = os.\
        # listdir(maindir)
    else:
        files_list = [neid_filename]

    # Making Reference residue
    ref_res = make_ref_residue(reference_params, maindir, korg_data_ref)
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
                                                ref_velocity=dead_velocity, sel_order=order)
            if neid_data_dict is None:
                print("No data in this order")
                return
            # save_dict_to_pickle(neid_data_dict, os.path.join(resultdict, basename + ".pkl"))
            # sys.exit()
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
    init_params3 = [-1,-1, 0, 0]
    # init_params3 = [-1, 1]
    # init_params3 = [0.649, -1.287, -1.16, 0.03] # inits# [2, 2, 2, 2]
    # lower_bounds = [-np.inf, -np.inf, -np.inf, -np.inf]# -5]#, -1]
    # upper_bounds = [np.inf, np.inf, np.inf, np.inf] #1]#]
    lower_bounds = [-np.inf, -np.inf, -np.inf, -np.inf]# -5]#, -1]
    upper_bounds = [0, 0, np.inf, np.inf] #1]#]

    bounds = np.array([lower_bounds, upper_bounds]).T
    param_pos = np.array(['r', 'r', 'r', 'v'])
    # param_pos = np.array(['p', 'p', 'p', 'p', 'v'])
    # param_pos = np.array(['p', 'p', 'p', 'v'])
    pca = False
    residue_vel = partial(residue_profile, neid_data=neid_data_dict,
                          synt_spectra=None,
                          # ref_params=np.array([ 0.6493511 , -1.28690028, -1.14780551,  0.03885628]),
                          area_fact=0.5,
                          scale_fact=-1,
                          pca_comp=pca,
                          param_pos=param_pos,
                          profile='parabola',
                          ip_data=ip_data,
                          ref_res=None, # ref_res,
                          algorithm=algorithm)
    if (purpose == 'minimize') or (purpose == 'both'):
        if not os.path.isfile(result_filename):
            print("Start fitting")
            if fitting == 'spectra':
                if algorithm == 'ls':
                    result = least_squares(residue_vel, init_params3,
                                           bounds=(lower_bounds, upper_bounds),
                                           x_scale='jac',
                                           jac='3-point',
                                           verbose=2,
                                           # ftol=None,
                                           # ftol=1e-8,
                                           xtol=None,
                                           gtol=None,
                                           method='trf',
                                           loss='linear', # 'soft_l1',
                                           max_nfev=1000)
                elif algorithm == 'diff':
                    bounds = [
                        (-100, 100),
                        (-100, 100),
                        (-100, 100),
                        (-100, 100)
                        ]
                    result = differential_evolution(
                        residue_vel,
                        bounds=bounds,
                        strategy='best1bin',
                        maxiter=1000
                    )
            elif fitting == "ccf":
                residue_ccf_fn = partial(residue_ccf,
                                         neid_fname=fullpath,
                                         opdir=resultdict)
                                         
                result = least_squares(residue_ccf_fn,
                                       init_params3,
                                       bounds=(lower_bounds, upper_bounds),
                                       x_scale='jac',
                                       jac='3-point',
                                       verbose=2,
                                       ftol=None,
                                       method='trf',
                                       loss='soft_l1',
                                       max_nfev=1000)
                
            print(result)

            pca_array_result = result.x
            
            print("Fitted_params for third order", result.x)
        
            additional_constant = pca_array_result[-1]
            if pca:
                pca_comps = pca_array_result[:-1]
                profile_params = reconstruct_params(pca_comps)
                profile_params[-1] += additional_constant
            else:
                profile_params = pca_array_result
            
            fitted_params3 = profile_params
            errorvals, cov_matrix = calculate_parameter_errors(result)
            resultdictionary = {'profile_params': fitted_params3,
                                'pca_comps': pca_array_result,
                                'params_err': errorvals,
                                'cov_matr': cov_matrix}
            save_dict_to_pickle(resultdictionary, result_filename)
            save_dict_to_pickle(result, os.path.join(resultdict,
                                                     "least_squares_op.pkl"))
            
        else:
            print("The result already exist. Calling it to plot")
            with open(result_filename, 'rb') as opfile:
                fitted_params3 = pickle.load(opfile)['profile_params']
        print("profile params", fitted_params3)
        save_synt_data(fitted_params3, fullpath, resultdict, save_interactive_plots=save_ip)
        print('resultdict', resultdict)
        print(os.path.dirname(fullpath))
        if resultdict == os.path.dirname(fullpath):
            print("Removing", fullpath)
            if os.path.exists(fullpath):
                os.remove(fullpath)
            synt_spectra_fname = "Synt_"+neid_filename
            if os.path.exists(os.path.join(resultdict, synt_spectra_fname)):
                os.remove(os.path.join(resultdict, synt_spectra_fname))
        # sys.exit(1)
        korg_spectra = residue_vel(fitted_params3, required='spectra')
        # generating residue for each order
        residue = korg_spectra['Res']
        residue_wl = korg_spectra['Wl']
        wl_orders = neid_data_dict['Wave']
        print(korg_spectra)
        residue_dict = {}
        # for order, wl_ar in wl_orders.items():
        #     order_wl_mask = np.isin(residue_wl, wl_ar)
        #     residue_order = residue[order_wl_mask]
        #     chi2_ord = np.sum(residue_order**2)
        #     residue_dict[order] = chi2_ord
        # # print(residue_dict)
        # save_dict_to_pickle(residue_dict, os.path.join(resultdict, "Order_residue.pkl"))
        from utils import plotting_spectra, plot_lines
        plotting_spectra(neid_data_dict,
                         korg_spectra, resultdict+"/Fitted_spectra.pdf",
                         interactive=save_ip)
        plot_lines(neid_data_dict,
                   korg_spectra, resultdict+"/Fitted_spectra_lines.pdf",
                   mask_filename='data/Sample_lines.csv') # Deep_lines.csv')

        if purpose == 'both':
            vel_array = np.linspace(fitted_params3-2*errorvals,
                                    fitted_params3+2*errorvals,
                                    200)
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

            generated_spectra = {"Raise": raising_spectra}

            result_specname = os.path.join(resultdict, "Generated_spectra.pkl")
            save_dict_to_pickle(generated_spectra, result_specname)

        print("Results saved in", resultdict)

    elif purpose == 'chi2':
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
    print("====================== The End ===============================")

# print("Arguements:", sys.argv)


parser = argparse.ArgumentParser(description="Run velprofile fitting.")
ref_fname = "neidL2_20220402T173047.fits"
# Required positional argument
parser.add_argument('--fname', type=str,
                    default=ref_fname, help='Input file name')

# Config file
parser.add_argument('--config', type=str,
                    default='Spectral_fitting.config', help='Config file name')
# --WL takes 2 values


# --SNR is optional here, but we will enforce it manually later
# if fname=="Korg"
parser.add_argument('--SNR', type=float,
                    help='Signal-to-noise ratio (only needed if fname is Korg)')

parser.add_argument('--LOG', type=str,
                    help='F to display the outputs.', default='T')
parser.add_argument('--init', type=float,
                    default=[0.0, 0.0, 0.0, 0.0],
                    nargs="+",
                    help="Initial conditions")

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
parser.add_argument('--orders', type=str,
                    default='F', help='Do for each order (T, F)')

# Parse args
args = parser.parse_args()
# print(args)
# Unpack WL window
# wl_wind1, wl_wind2 = args.WL

# print(config['data_dir']['NEID_DIR'])


# Check if SNR is needed
if args.fname == "Korg" and args.SNR is None:
    print("Error: --SNR must be provided when fname is 'Korg'.")
    sys.exit(1)

# Either use given SNR or a dummy value if not provided
# (only for non-Korg cases)
snr_value = args.SNR if args.SNR is not None else 500
# You can also choose not to pass it eat all if not needed

ip = args.save_ip
# print("ip", ip)

if ip == "T" or args.fname == ref_fname:
    save_ip = True
else:
    save_ip = False


# Print info
reference_params = {'fname': 'neidL2_20220402T173047.fits',
                    'params': np.array([-1.45847855e+00, -3.05012614e-14, -4.61296508e-02, -6.17010467e-01])  # np.array([ 1.70357454, -1.92482808, -1.07396776,  0.04051153])
                    # 'params': np.array([ 0.6493511 , -1.28690028, -1.14780551,  0.03885628])
                    }
print("Doing for:", args.fname)
print("Order:", args.orders)
# print(f"Wavelength window: {wl_wind1} - {wl_wind2}")
print(f"Dead velocity: {args.dead_vel}")
if args.fname == "Korg":
    print(f"SNR: {args.SNR}")
print(args.LOG)
def process_order(order):
    velprofile_fit_function(
        args.fname,
        args.config,
        dead_velocity=args.dead_vel,
        snr=snr_value,
        logfilesave=args.LOG,
        purpose=args.purpose,
        fitting=args.fitting,
        save_ip=save_ip,
        inits=args.init,
        order=order,
        reference_params=reference_params
    )


# Call your function
try:
    order = int(args.orders)
    if (order > 70) & (order < 162):
        process_order(order)
    else:
        print("Please enter orders between 71 and 161")
        sys.exit()
except ValueError:
    if args.orders == 'F':
        velprofile_fit_function(
            args.fname,
            args.config,
            dead_velocity=args.dead_vel,
            snr=snr_value,  # Only needed for Korg, but passing anyway
            logfilesave=args.LOG,
            purpose=args.purpose,
            fitting=args.fitting,
            save_ip=save_ip,
            inits=args.init,
            reference_params=reference_params
        )
    elif args.orders == 'T':
        # num_cores = 1
        # orders = [173 - i for i in range(121) if 70 <= (173 - i) <= 162]
        # with ProcessPoolExecutor(max_workers=num_cores) as executor:
        #     executor.map(process_order, orders)
        for i in range(121):
            if ((173-i) < 70) or ((173-i) > 162):
                continue
            order = 173-i
            process_order(order)
      #       velprofile_fit_function(
    #             args.fname,
    #             args.config,
    #             dead_velocity=args.dead_vel,
    #             snr=snr_value,  # Only needed for Korg, but passing anyway
    #             logfilesave=args.LOG,
    #             purpose=args.purpose,
    #             fitting=args.fitting,
    #             save_ip=save_ip,
    #             inits=args.init,
    #             order=order
    #         )
    # # 
    # elif isinstance(args.orders, int)


print('\a')

# End of code
