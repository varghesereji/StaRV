import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import time
import os
from collections import defaultdict
import shutil
import pickle

from utils import calling_linelist
from utils import call_neid_data

from functools import *

# from model_functions import residue_scale_factor
from model_functions import residue_velocity_profile
from model_functions import generate_with_korg
from model_functions import shifting_korg_flux
from model_functions import calculate_parameter_errors

from scipy.interpolate import CubicSpline
import juliapkg
from juliacall import Main as jl
jl.seval("using Korg"); Korg=jl.Korg


# resultdict = "Result_directory_Worstline_1"
resultdict = "Result_directory_Debugnan2"

if not os.path.exists(resultdict):
    os.makedirs(resultdict)
    print(resultdict, "Directory created successfully!")
else:
    print(resultdict, "Directory already exists!")

shutil.copy("model_functions.py", resultdict)
shutil.copy("Velocity_profile_fitting.py", resultdict)

korg_filename = "Korg_data_3800_9000_nocontdiv.pkl"
if os.path.exists(korg_filename):
    print("The data already exists.")
    with open(korg_filename, 'rb') as korgfile:
        korg_data = pickle.load(korgfile)
else:
    print("Pre-generated data does not exist. Generating now")

    korg_data = generate_with_korg(np.zeros(56), (3800, 9000), continuum=False)
    with open(korg_filename, 'wb') as korgfile:
        pickle.dump(korg_data, korgfile)

cntm_filename = "Ratio_3800_9000.pkl"
# if os.path.exists(cntm_filename):
# print("The continuums already exists.")
with open(cntm_filename, 'rb') as korgfile:
    ratio_pkl = pickle.load(korgfile)
# else:
#     print("Pre-generated data does not exist. Generating now")

#     cntm1_data = generate_with_korg(np.zeros(56), (3800, 9000), continuum=True)
#     cntm1 = np.array(cntm1_data['cont'])
#     cntm2_data = generate_with_korg(np.zeros(56), (3800, 9000), temp=5570, continuum=True)
#     cntm2 = np.array(cntm2_data['cont'])
#     print(type(cntm1))
#     wl = np.array(cntm2_data['Wl'])
#     ratio = cntm1/cntm2
    
#     ratio_pkl = {'ratio':ratio, 'Wl':wl}

#     with open(cntm_filename, 'wb') as korgfile:
#         pickle.dump(ratio_pkl, korgfile)


neid_orders = 122

neid_data_dict = defaultdict(list)
for order in range(neid_orders):
    neid_wl, neid_flux, neid_err = call_neid_data(order)
    neid_err = neid_err.astype(np.float64)
    # print(order, neid_wl, neid_flux, neid_wl)
    neid_data_dict["Flux"].append(neid_flux)
    neid_data_dict["Wave"].append(neid_wl)
    neid_data_dict["Err"].append(neid_err)

# print(neid_data_dict)

# Calling lines
lines_file_path = '/home/varghese/Desktop/Stellar_activity_mitigation'
# solar_lines_fname = 'sol_line_window_modified.csv' # 'Solar_lines_gray.csv' # 'sol_line_window_modified.csv'
solar_lines_fname = 'Solar_lines_gray.csv'
line_dict = calling_linelist(os.path.join(lines_file_path, solar_lines_fname))
shutil.copy(os.path.join(lines_file_path, solar_lines_fname), resultdict)
# korg_data = generate_with_korg(np.zeros(56))


# print("simply calling Korg")
# generate_with_korg(velocity=None)

# residue_scale_factor([0,0,0], neid_data_dict, line_dict, korg_data)

# print(neid_data_dict)
# neid_filename = '/home/varghese/Desktop/Stellar_activity_mitigation/neidL2_20220110T180951.fits'

result0_name = "fitted_params0.pkl"
result0_path = os.path.join(resultdict, result0_name)
if os.path.exists(os.path.join(result0_path)):
    print("The result0 already exists")
    with open(result0_path, 'rb') as result0:
        result0_vals = pickle.load(result0)
    fitted_params = result0_vals['params']
    v_raise0 = fitted_params[0]
    v_fall0 = fitted_params[1]
    scale_factor = fitted_params[2]
    
else:
    print("Result0 does not exist")
    init_params = [-0.5, 2.5, 0.1]
    lower_bounds = [-5, 0, 0]
    upper_bounds = [0, 5, 0.5]
    starttime=time.time()
    
    residue_vel0 = partial(residue_velocity_profile, neid_data=neid_data_dict,
                           linelist=line_dict,
                           timer=starttime,
                           result_dict=resultdict,
                           korg_data=korg_data,
                           ratio_dict=ratio_pkl,
                           fitting_line=True)
    
    result = least_squares(residue_vel0, init_params,
                           bounds=(lower_bounds, upper_bounds),
                           x_scale='jac',
                           verbose=2,
                           loss='cauchy')
    
    fitted_params = result.x
    residue_vel0(params=fitted_params, plot_lines=True, plot_fname='Vel_profile_invertedratio')
    v_raise0 = fitted_params[0]
    v_fall0 = fitted_params[1]
    scale_factor = fitted_params[2]
    print("Fitted_params for 0th order:", result.x)
    errorvals, cov_matrix = calculate_parameter_errors(result)
    resultdictionary = {'params':fitted_params,
                        'params_err': errorvals,
                        'cov_matr': cov_matrix}
    # result_filename = os.path.join(resultdict, "fitted_params.pkl")
    # save_dict_to_pickle(resultdictionary, result_filename)
    with open(result0_path, 'wb') as resultfile:
        pickle.dump(resultdictionary, resultfile)
    print("Results saved in", resultdict)
    
   # print("Results saved in", resultdict)

plot_fname = "Velocity_profile"
# Fitting for 1st order
'''
korg_flux = korg_data["Flux"]
korg_wl = korg_data["Wl"]
falling_vel = 2.1
shifted_falling_lane = shifting_korg_flux(korg_wl, korg_flux, falling_vel)

raising_vel0 = -0.7
init_params1 = [0, v_raise0, 0, v_fall0]
lower_bounds1 = [-np.inf, -5, 0, 0]
upper_bounds1 = [np.inf, 0, np.inf, 5]

starttime=time.time()
residue_vel1 = partial(residue_velocity_profile, neid_data=neid_data_dict,
                       linelist=line_dict,
                       scale_factor=scale_factor,
                       timer=starttime,
                       result_dict=resultdict)




result1 = least_squares(residue_vel1, init_params1,
                        bounds=(lower_bounds1, upper_bounds1),
                        x_scale='jac',
                        verbose=2,
                        loss='cauchy')
fitted_params1 = result1.x

# scale_factor = 0.3
# fitted_params1 = np.array([-0.12176467, -0.72631159, 0.29697187, 2.47365526])
print("Fitted_params from for first order", fitted_params1)
print("Results saved in", resultdict)
v_raise1 = fitted_params1[:-2]
v_fall1 = fitted_params1[-2:]

# Fitting for second order
init_params2 = np.insert(v_raise1, 0, 0)
lower_bounds2 = [-np.inf, -np.inf, -5,]
upper_bounds2 = [np.inf, np.inf, 0]
starttime=time.time()
residue_vel2 = partial(residue_velocity_profile, neid_data=neid_data_dict,
                       linelist=line_dict,
                       scale_factor=scale_factor,
                       falling_params=v_fall1,
                       timer=starttime,
                       result_dict=resultdict)
result2 = least_squares(residue_vel2, init_params2,
                        bounds=(lower_bounds2, upper_bounds2),
                        x_scale='jac',
                        verbose=2,
                        loss='cauchy')

fitted_params2 = result2.x
print("Fitted_params for second order", fitted_params2)
print("Results saved in", resultdict)
v_raise2 = fitted_params2
'''
# Fitting for third order
# init_params3 = np.insert(v_raise2, 0, 0)
starttime=time.time()
init_params3 = [0, 0, 0, v_raise0, 0, v_fall0] # Fitting in one step
lower_bounds3 = [-np.inf, -np.inf, -np.inf, -5, -np.inf, 0]
upper_bounds3 = [np.inf, np.inf, np.inf, 0, np.inf, 5]

# lower_bounds3 = [-np.inf, -np.inf, -np.inf, -5,]
# upper_bounds3 = [np.inf, np.inf, np.inf, 0]

residue_vel3 = partial(residue_velocity_profile, neid_data=neid_data_dict,
                       linelist=line_dict,
                       scale_factor=scale_factor,
                       # falling_params=v_fall1,
                       timer=starttime,
                       result_dict=resultdict)
result3 = least_squares(residue_vel3, init_params3,
                        bounds=(lower_bounds3, upper_bounds3),
                        x_scale='jac',
                        verbose=2,
                        loss='cauchy')

print(result3)
fitted_params3 = result3.x

print("Fitted_params for third order", result3.x)
residue_vel3(params=fitted_params3, plot_lines=True)
print("Results saved in", resultdict)






