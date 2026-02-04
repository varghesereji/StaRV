import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import time
import csv
import os
from scipy.optimize import least_squares
from scipy.ndimage import median_filter
from itertools import combinations_with_replacement
from astropy.io import fits
from scipy.signal import fftconvolve

from scipy.interpolate import CubicSpline
from scipy.special import eval_legendre
import gc

from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

from spectral_synthesis_functions import generate_with_korg
from spectral_synthesis_functions import generate_full_korgspectra

    

def shifting_korg_flux(wavelength, flux, velocity):
    c = 299792.458
    # print('Shifting for ', velocity)
    shifted_wl = wavelength * (velocity/c + 1)
    shifted_flux = CubicSpline(shifted_wl, flux)(wavelength)
    return shifted_flux
    

def shifted_legendre(n, x):
    """
    Compute the nth-degree Shifted Legendre Polynomial at points x in [0, 1].

    Parameters:
        n (int): Degree of the polynomial.
        x (array_like): Points at which to evaluate the polynomial (in [0, 1]).

    Returns:
        array_like: Values of the shifted Legendre polynomial.
    """
    # Transform x from [0, 1] to [-1, 1] for standard Legendre polynomials
    x_transformed = 2 * x - 1
    return eval_legendre(n, x_transformed)


def generate_profile(params, x, param_errs=None):
    vel_profile = 0
    vel_profile_err = 0
    # print("Generate_profile", params)
    for n, p in enumerate(params[::-1]):
        # print('Deg', n, p)
        poly_deg = x**n # shifted_legendre(n, x)
        vel_profile += poly_deg * p
        if param_errs is not None:
            vel_profile_err += (poly_deg*param_errs[::-1][n]) ** 2
    if param_errs is not None:
        return vel_profile, np.sqrt(vel_profile_err)
    return vel_profile
        


def synt_spectra_for_wavelength(velocity, wl_array, synt_spectra=None, stellar_params=None, cont_divide=True):
    '''
    Main tasks of this function is:
    1. Synthesize the function with the given velocity profile (if no pre-defined spectra is given).
    2. If a pre-defined spectra and a velocity value is given, apply doppler shift on the spectra.
    3. Interpolate the synthetic spectra to the given wavelength.
    velocity: The velocity profile or velocity value. If False, just interpolate the flux to the given wavelength.
    wl_array: The required wavelength array. The synthetic spectra will be interpolated onto this array.
    synt_spectra: If None, the spectra will be generated. Pre-saved synthetic spectra also can be used.
    synt_spectra should be a dictionary in the format {'Flux':..., 'Wl':...}.
    
    '''
    # print("synt_spectra_for_wavelength", velocity)
    # print("synt_spectra", synt_spectra)
    temp_layers = np.load("data/Temp_layers.npy")
    x = (temp_layers - np.min(temp_layers)) / (np.max(temp_layers) - np.min(temp_layers))
    if velocity is None:
        transformed_flux = CubicSpline(synt_spectra['Wl'], synt_spectra['Flux'])(wl_array)
        return transformed_flux
    
    if (np.size(velocity)==1) and velocity is not None:
        velocity = velocity[0]
    if (np.size(velocity) > 1):
        n_layers = 56
        # x = np.linspace(0, 1, n_layers)
        velocity = generate_profile(velocity, x)
    if synt_spectra is None:
        if isinstance(velocity, (int, float)):
            n_layers = 56
            # x = np.linspace(0, 1, n_layers)
            velocity = generate_profile([velocity], x)
        # print("Velocity profile", velocity)
        synt_spectra = generate_with_korg(velocity, stellar_params=stellar_params, cont_divide=cont_divide)
        shifted_flux = synt_spectra['Flux']
    else:
        if isinstance(velocity, np.ndarray):
            print("Only doppler shift is applicable on pre-defined spectra. Either give a single value for the velocity or keep synt_spectra=None to generate new one")
        else:
            shifted_flux = shifting_korg_flux(synt_spectra['Wl'], synt_spectra['Flux'], velocity)
    transformed_flux = CubicSpline(synt_spectra['Wl'], shifted_flux)(wl_array)
    if not cont_divide:
        cont = synt_spectra['cont']
        transformed_cont = CubicSpline(synt_spectra['Wl'], cont)(wl_array)
        return transformed_flux, transformed_cont
    else:
        return transformed_flux


def dict_to_array(dictionary):
    '''
    This function is to extract the values of the input dictionary and return as a signle array.
    '''
    flattened_values = np.array([])
    dict_values = list(dictionary.values())
    dict_keys = list(dictionary.keys())
    # print(dict_keys)
    for order in dict_keys:
        subarray = dictionary[order]
        flattened_values = np.concatenate((flattened_values, subarray))
    return flattened_values


def profile_penalty(params=None, param_pos=None, raise_params=None, fall_params=None):
    '''
    Function to estimate the penalty.
    '''
    if params is not None and param_pos is not None:
        raise_mask = param_pos == 'r'
        fall_mask = param_pos == 'f'
        raise_params = params[raise_mask]
        fall_params = params[fall_mask]
    elif raise_params is not None and fall_params is not None:
        raise_params = raise_params
        fall_params = fall_params

    n_layers = 56
    x = np.linspace(0, 1, n_layers)
    velocity_raise = generate_profile(raise_params, x)
    velocity_fall = generate_profile(fall_params, x)

    # raise_vel_edge_penalty = generate_profile(raise_params, 0)/100# penalty for left edge of Raising lane
    # fall_vel_edge_penalty = generate_profile(fall_params, 0)/100 # penalty for left edge of Falling lane
    # Penalty for the sign of profile elements.

    raise_vel_mask = velocity_raise > 0 # Mask for raising velocity. Positive terms will be masked.
    # fall_vel_mask = velocity_fall < 0 # Mask for falling velocity. Negative terms will be masked.

    raise_prof_penalty = np.sum(np.abs(velocity_raise[raise_vel_mask]))
    # fall_prof_penalty = np.sum(np.abs(velocity_fall[fall_vel_mask]))

    # prof_penalty = np.array([np.abs(raise_vel_edge_penalty),
    #                          np.abs(fall_vel_edge_penalty),
    #                          raise_prof_penalty, fall_prof_penalty]) *10000000
    prof_penalty = np.array([raise_prof_penalty])
    # Penalty for the derivative of profile elements.
    # The negative profile should have negative derivative and
    # Falling should have positive derivative.

    raise_der = np.gradient(velocity_raise)
    fall_der = np.gradient(velocity_fall)

    raise_der_mask = raise_der > 0
    fall_der_mask = fall_der < 0

    raise_der_penalty = np.sum(np.abs(raise_der[raise_der_mask]))
    # fall_der_penalty = np.sum(np.abs(fall_der[fall_der_mask]))

    # der_penalty = np.array([raise_der_penalty, fall_der_penalty]) * 1000000000
    der_penalty = np.array([raise_der_penalty])
    penalty_array = np.concatenate((prof_penalty, der_penalty)) * 1000000000
    print("Penalty: {}".format(penalty_array))
    return penalty_array
    
    

def residue_profile(params, neid_data, synt_spectra=None,
                    param_pos=np.array(['r','f']),
                    area_fact=0.5, # None,
                    scale_fact=None,
                    required='residue',
                    ip_data='epoch'):
    '''
    This function is to return the residue of synthetic and observed spectra.
    params: The model parameters. Parameters for the velocity profiles of raising, falling and scale factor if needed.
    neid_data: Dictionary of the form {"Flux":{#order:...}, "Wave":{#order:...}, "Err":{#order:...}}
    synt_spectra: If a pre-defined synthetic spectra is available, simply do a doppler shift. Use only for one parameter fitting.
    params_pos: array to separate raising and falling profile parameters. 'r': raise, 'f': fall, 'a': scale factor which is the fractional area of granular region.
    scale_fact: The fractional area of granular region on stellar disk.
    required: 'residue' if need the residue. 'spectra' of the generated spectra is needed.
    ip_data: epoch or average. If epoch, the difference from the result of average will be taken.
    '''
    print("Params: {}".format(params))
    avgepoch_result = "/home/varghese/Desktop/Stellar_activity_mitigation/25_Avg/Result_dead_vel/Result_dir_Comb_spectra_3700-9000/fitted_params.pkl"
    if ip_data == 'epoch':
        with open(avgepoch_result, 'rb') as avgres:
            avgepo = pickle.load(avgres)
        params = params + avgepo['params']
    # Parameter mask
    raise_mask = param_pos == 'r'
    fall_mask = param_pos == 'f'
    add_shift_mask = param_pos == 'v' # This is the additional velocity for doppler shift the entire spectra
    scale_mask = param_pos == 'a'

    # Additional velocity
    if np.sum(add_shift_mask) == 0:
        add_vel = 0
    else:
        add_vel = params[add_shift_mask]
    # Extracting data
    neid_wl_array = dict_to_array(neid_data["Wave"])
    neid_flux_array = dict_to_array(neid_data["Flux"])
    neid_err_array = dict_to_array(neid_data["Err"])

    # Generating spectra
    raise_params = params[raise_mask]
    raise_params[-1] += add_vel
    # raise_params
    raise_spectra, raise_cont = synt_spectra_for_wavelength(raise_params, neid_wl_array, synt_spectra=synt_spectra, cont_divide=False)
    fall_spectra = 0
    total_cont = raise_cont
    penalty = np.array([0, 0, 0, 0])
    falling_lane = False

    if np.sum(fall_mask) > 0:
        fall_params = params[fall_mask]
        falling_lane = True
        # penalty = profile_penalty(params, param_pos)
    elif np.sum(fall_mask) == 0:
        if (np.sum(scale_mask) == 1) and scale_fact is None:
            params_scale_fact = params[scale_mask]
        elif (np.sum(scale_mask) == 0) and scale_fact is not None:
            params_scale_fact = scale_fact
    
    
        fall_params = params_scale_fact * params[raise_mask]
        falling_lane = True
    
    if falling_lane:
        stellar_params = {'temp':5321} #5570}
        fall_params[-1] += add_vel
        fall_spectra, fall_cont = synt_spectra_for_wavelength(fall_params, neid_wl_array, synt_spectra=synt_spectra,
                                                              stellar_params=stellar_params, cont_divide=False)
        penalty = profile_penalty(raise_params=params[raise_mask], fall_params=params_scale_fact*params[raise_mask])
        # if area_fact is None:
        #     a = params[scale_mask]
        # else:
        #     a = scale_fact

        total_flux = area_fact * raise_spectra + (1-area_fact) * fall_spectra # raise_spectra + ((1-a)/(1+a)) * fall_spectra
        total_cont = area_fact * raise_cont + (1-area_fact)*fall_cont
        synt_spectra = total_flux / total_cont


    else:
        synt_spectra = raise_spectra / raise_cont
        fall_spectra = np.nan
    residue = (neid_flux_array - synt_spectra) / neid_err_array
    if required == 'residue':
        residue = np.concatenate((residue, penalty))
        return residue
    
    elif required == 'spectra':
        spectra_dict = {"Raise":area_fact*raise_spectra/total_cont,
                        "Fall":(1-area_fact)*fall_spectra/total_cont,
                        "Total":synt_spectra,
                        "Res":residue,
                        "Wl":neid_wl_array}
        return spectra_dict
    
    
    
            
            
    
# def residue_velocity_profile(params, neid_data, linelist, scale_factor=None,falling_params=None,
#                              parameter_set=np.array(['r','r','r','r', 'f','f']),
#                              stellar_params=None, n_layers=56, selected_orders=None, timer=None,
#                              model='bilane',
#                              plot_fname='Vel_profile', result_dict='.',
#                              korg_data=None,
#                              plot_lines=False,
#                              fitting_line=False,
#                              ratio_dict=None,
#                              save_interactive_plot=False,
#                              wlwinds=(3700,9000)):
#     '''
#     Function to find the velocity profile of the raising lane
#     params: The model parameters.
#     neid_data: Dictionary. {"Flux":ndarray, "Wave":ndarray, "Err", ndarray}
#     linelist: list of selected lines.
#     scale_factor: If the model is to find scale factor, keep this to None. Otherwise, pass scale factor.
#     falling_params: If the falling parameters is already calculated, pass it to here. Otherwise, keep None.
#     parameter_set: 'r' means the parameter at this position in the parameter array is for raising lane. 'f' means it is for falling lane.
#     Size of this array should be same as size of params.
#     stellar_params: Stellar parameters. None will use the parameter for sun.
#     n_layer: number of atmospheric layer.
#     selected_orders: NEID orders used for the analysis.
#     timer: Parameter used to calculate the time passed.
#     plot_fname: Filename of plot.
#     result_dict: Directory to save the results.
#     korg_data: Spectra generated with Korg. If not given, it will be generated in this function.
#     fitting_line: Bool. If true, it will fit a gaussian with the line to find the oprtion of the line to select.
#     ratio_dict: Dictionary. Ratio between raising and falling lanes. This is to compensate the difference in temperature between them.
#     wlwinds: Wavelength window.
#     '''
#     print("Params {}".format(params))
#     penalty_fact = 0
#     penalty_fact_derivative = 0
#     if stellar_params is not None:
#         temp, logg = stellar_params
#     if selected_orders is None:
#         selected_orders = list(neid_data["Flux"].keys()) # Getting the orders.
#     raising_params = None
#     if scale_factor is None: # If the scale factor is None, the velocity profile is 0th order by default. Then this function is to calculate the scale_factor
#         scale_factor = params[-1]
#         raising_params = np.array([params[0]])
#         falling_params = np.array([params[1]])
#     if falling_params is None:
#         raise_mask = parameter_set == 'r'
#         fall_mask = parameter_set == 'f'
#         raising_params = params[raise_mask] # np.array([params[0], params[1]])
#         falling_params = params[fall_mask] #np.array([params[2], params[3]])
#     if raising_params is None:
#         raising_params = params
#         falling_params = falling_params

#     # print("Raising params", raising_params)
#     # print("Falling params", falling_params)
#     # print("scale_factor", scale_factor)
#     if korg_data is None: # If pre-defined Korg data is not given, a synthetic data will be generated according to the given velocity profile.
#         x = np.linspace(0, 1, n_layers)
#         vel_raise = generate_profile(raising_params, x) # np.poly1d(raising_params)(x)  # Velocity profile of raising lane
#         if model=='bilane':
#             vel_fall = generate_profile(falling_params, x) # np.poly1d(falling_params)(x) # Velocity profile of falling lane
#             # Calculating the penalty term for the profile
#             vraise_mask = vel_raise > 0
#             vfall_mask = vel_fall < 0
            
#             vraise_fact = np.sum(vel_raise[vraise_mask])
#             vfall_fact = np.sum(vel_fall[vfall_mask])

#             # Derivative of the velocity profile
#             vraise_der = np.gradient(vel_raise) / np.gradient(x)
#             vfall_der = np.gradient(vel_fall) / np.gradient(x)

#             # Calculating the penalty term for the derivative of the profile
#             vraise_der_mask = vraise_der > 0
#             vfall_der_mask = vfall_der < 0
            
#             vraise_der_fact = np.sum(np.abs(vraise_der[vraise_der_mask]))
#             vfall_der_fact = np.sum(np.abs(vfall_der[vfall_der_mask]))
            
#             penalty_fact = abs(vraise_fact) + abs(vfall_fact)
            
#             penalty_fact_derivative = abs(vraise_der_fact) + abs(vfall_der_fact)
#         else:
#             vel_fall = 0

#         if plot_lines:
#             plt.figure()
#             plt.plot(x, vel_raise, 'o-', color='blue')
#             if model=='bilane':
#                 plt.plot(x, vel_fall, 'o-', color='red')
#             plt.xlabel("Layer parameter")
#             plt.ylabel("Velocity (km/s)")
#             plt.savefig(result_dict+"/"+"Velocity_profile_o{}.pdf".format(len(raising_params)-1))
#     else:
#         # If the pre-defined Korg spectra is given. Preferred to apply on one parameter to save time.
#         # This is only to do the fitting for order 0
#         korg_flux = korg_data['Flux']
#         korg_wl = korg_data['Wl']
#         raising_flux = shifting_korg_flux(korg_wl, korg_flux, params[0])
#         # falling_lane = shifting_korg_flux(korg_wl, korg_flux, params[1])

#     # synt_flux = raising_flux * (1-scale_factor) + falling_lane*scale_factor
#     residue_list = np.array([]) # Initilizing the residue array
#     centroids = list(linelist.keys()) # List of centroids of the line
#     # print("Plot_lines", plot_lines)
#     if plot_lines:
#         pdf = PdfPages(result_dict+"/"+plot_fname+"_o{}.pdf".format(len(raising_params)-1))
#         # fig, axs = plt.subplots(5, 3, figsize=(16, 16))
#     # korg_data_raise = generate_full_korgspectra(vel_raise)
#     if model == 'bilane': # If the model is bilane, generate the spectra for falling lane
#         korg_data_fall = generate_full_korgspectra(vel_fall)
    
#     for order in selected_orders:
#         # print("Order", order)
#         if isinstance(neid_data, str):
#             from utils import call_neid_data
#             neid_wls, neid_flux, neid_err = call_neid_data(173 - order, neid_data) # If the NEID data is not given.
#         else:
#             # Extracting the flux, wavelength and error
#             neid_flux = neid_data["Flux"][order] 
#             neid_wls = neid_data["Wave"][order]
#             neid_err = neid_data["Err"][order]
#         # generating Korg spectra

#         for n, line in enumerate(centroids):
#             wl_mask = (neid_wls > linelist[line][0] ) & (neid_wls < linelist[line][1]) & (~np.isnan(neid_wls)) & (~np.isnan(neid_flux)) & (~np.isnan(neid_err)) & (neid_wls >= float(wlwinds[0])) & (neid_wls <= float(wlwinds[1]))
#             # print("Check mask", np.sum(wl_mask), neid_wls, neid_flux)
#             if np.sum(wl_mask) > 0:
#                 line_wl = neid_wls[wl_mask]


#                 line_neid = neid_flux[wl_mask]
#                 line_err = neid_err[wl_mask]

#                 line_level = np.percentile(line_neid, 90)

#                 norm_neid = line_neid # / line_level
#                 norm_err = line_err # / line_level
                
#                 # Generating Korg spectra
#                 if korg_data is None:
#                     diff = 5
#                     wlmin = int(np.nanmin(line_wl))-diff
#                     wlmax = int(np.nanmax(line_wl))+diff
#                     # print("Order", order, "Wl window", wlmin, wlmax)
#                     # print("vraise_fact:", vraise_fact, vfall_fact)
#                     # print(vel_raise)
#                     korg_data_raise = generate_with_korg(vel_raise, wl_wind=(wlmin, wlmax))
#                     raising_flux = korg_data_raise['Flux']
#                     korg_wl = korg_data_raise['Wl']
#                     # print(vel_fall)
#                     if model=='bilane':
#                         # korg_data_fall = generate_with_korg(vel_fall, wl_wind=(wlmin, wlmax))
#                         falling_lane = korg_data_fall['Flux']
#                     else:
#                         falling_lane = 0

#                 # filtering Korg spectra
#                 # scaling_the_flux. Useless for entire spectra which continuum fitting was already done.
#                 if ratio_dict is not None and model=='bilane': # The ratio of continuums of two lanes.
#                     ratio_wl = ratio_dict['Wl']
#                     ratio = ratio_dict['ratio']
#                     ratio_window = CubicSpline(ratio_wl, ratio)(korg_wl)
#                     falling_lane = falling_lane / ratio_window
#                 # print("sizes", np.size(raising_flux), np.size(falling_lane))
#                 synt_flux = raising_flux # + ((1-scale_factor)/(1+scale_factor)) * falling_lane # The synthetic spectra
#                 # print(synt_flux, np.sum(np.isnan(synt_flux)))
#                 line_korg = CubicSpline(korg_wl, synt_flux)(line_wl)
#                 korg_level = np.percentile(line_korg, 90)
#                 norm_line_korg = line_korg # / korg_level
                
#                 # Fitting line with a gaussian to reducethe window
#                 residue = (norm_neid - norm_line_korg)/norm_err
#                 # print("Fitting line", fitting_line)
#                 if fitting_line: # If the fitting is for single line, fit the line with a gaussian to isolate the line from other regions. Not needed for entire spectrum.
#                     initial_guess = [-0.3, line, 0.3, 1]
#                     result = least_squares(residuals_gaussian, initial_guess, args=(line_wl, norm_neid),
#                                            verbose=0)
#                     popt = result.x
#                     A, mu, sigma, C = popt
#                     mult = 2
#                     fwhm = 2.355*np.abs(sigma)
#                     infremum = line - mult*fwhm
#                     supremum = line + mult*fwhm
#                     sel_mask = (line_wl >= infremum) & (line_wl <= supremum)
                    
#                     residue = residue[sel_mask]
#                 # print(line, "line_wl", line_wl)
#                 if plot_lines:
#                     # plt.figure(figsize=(16,8))
#                     fig, axs = plt.subplots(2, sharex=True, figsize=(16, 8))
#                     axs[0].plot(line_wl, norm_neid, label="NEID {}".format(order+52))
#                     axs[0].plot(line_wl, norm_line_korg, label="Korg")
#                     if model=='bilane':
#                         plt.plot(line_wl, CubicSpline(korg_wl, raising_flux/korg_level)(line_wl), '--', color='blue', label='Raising')
#                         # plt.plot(line_wl, CubicSpline(korg_wl, ((1-scale_factor)/(1+scale_factor)) * falling_lane/korg_level)(line_wl), '--', color='red', label='Falling')
#                     axs[1].plot(line_wl, residue, color='k', label="Residue")
#                     axs[0].tick_params(axis='both', labelsize=18)
#                     axs[1].tick_params(axis='both', labelsize=18)
#                     axs[1].set_xlabel("Wavelength $\AA$", fontsize=18)
#                     axs[0].set_ylabel("Flux", fontsize=18)
#                     plt.tight_layout()
                    
#                     # plt.plot(line_wl[sel_mask], norm_neid[sel_mask], 'o-',color='k', label='Selected portion', alpha=0.4)
#                     axs[0].legend()
#                     axs[1].legend()
#                     # plt.title("{} \n raising coeffs {} \n falling coeffs {} \n scale factor {}".format(line, raising_params, falling_params, scale_factor))
#                     # plt.savefig(result_dict +"/test_spectra_{}_{}.pdf".format(np.nanmin(line_wl), np.nanmax(line_wl)))
#                     pdf.savefig()
#                     plt.close()
#                 # for res in residue:
#                 #     residue_list.append(res)
#                 if save_interactive_plot:
#                     from utils import plot_plotly
#                     plotlyfig = make_subplots(
#                         rows=2, cols=1, shared_xaxes=True,
#                         vertical_spacing=0.02)
#                     plot_plotly(plotlyfig, line_wl, norm_neid, 1, 1, 'red', 'NEID')
#                     plot_plotly(plotlyfig, line_wl, norm_line_korg, 1, 1, 'green', 'Korg')
#                     plot_plotly(plotlyfig, line_wl, residue, 2, 1, 'black', 'Residue')
#                     spectra_dir = os.path.join(result_dict, 'Spectra')
#                     if not os.path.exists(spectra_dir):
#                         os.makedirs(spectra_dir)
#                     pyo.plot(plotlyfig, filename=spectra_dir+"/Fitted_spectra_order{}.html".format(order))# )
#                 residue_list = np.concatenate((residue_list, residue))
#     # residue_list.append(penalty_fact*10)
#     # residue_list.append(penalty_fact_derivative*10)
#     residue_list = np.concatenate((residue_list, np.array([penalty_fact*10, penalty_fact_derivative*10])))
#     print("Penalty_added: {}".format(penalty_fact*10))
#     print("Derivative Penalty added: {}".format(penalty_fact_derivative*10))
#     residue_list = np.array(residue_list)
#     if plot_lines:
#         # fig.suptitle("raising coeffs {} \n falling coeffs {} \n scale factor {}".format(raising_params, falling_params, scale_factor))
#         # plt.savefig(result_dict+"/"+plot_fname+"_o{}.pdf".format(len(raising_params)-1))
#         # plt.close()
#         pdf.close()
#     if timer is not None:
#         current = time.time()
#         time_passed = current - timer
#         print(f"Time passed (s):{time_passed}")
#     # print("Residue in function", np.sum(residue_list**2))
#     return residue_list


# def gaussian(x, A, mu, sigma, C):
#     return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + C

# # Residual function for least_squares fitting
# def residuals_gaussian(params, x, y):
#     return gaussian(x, *params) - y



def calculate_parameter_errors(least_squares_result):
    """
    Calculate parameter standard errors from least_squares optimization results.
    
    Args:
        least_squares_result: Result object from scipy.optimize.least_squares
        
    Returns:
        tuple: (parameter_errors, covariance_matrix)
            - parameter_errors: array of standard errors for each parameter
            - covariance_matrix: estimated covariance matrix
            
    Note: Assumes Gaussian errors and uses linear approximation for uncertainty estimation.
    """
    # Extract components from optimization result
    J = least_squares_result.jac  # Jacobian matrix
    residuals = least_squares_result.fun  # Residual vector
    m, n = J.shape  # m = data points, n = parameters
    
    if m <= n:
        raise ValueError("More parameters than data points - cannot compute errors")
    
    # Calculate residual variance (sigma^2)
    residuals_sum_sq = np.sum(residuals**2)
    sigma_sq = residuals_sum_sq / (m - n)
    
    # Compute covariance matrix using pseudoinverse for numerical stability
    try:
        cov_matrix = sigma_sq * np.linalg.pinv(J.T @ J)
    except np.linalg.LinAlgError:
        cov_matrix = sigma_sq * np.linalg.inv(J.T @ J)
    
    # Calculate standard errors from covariance matrix diagonal
    param_errors = np.sqrt(np.diag(cov_matrix))
    
    return param_errors, cov_matrix
