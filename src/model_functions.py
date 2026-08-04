
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
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
from functools import lru_cache

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
    print('ddoppler Shifting for {}'.format(velocity))
    factor = (velocity/c + 1)
    shifted_wl = wavelength * factor # (velocity/c + 1)
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


def reconstruct_params(P):

    # A = np.array([[-0.032937977555381484, 0.024074427096568495],
    #              [0.031571960296695074, -0.013240645075579883],
    #              [-0.006694995026312799, 0.0014737690483261512],
    #              [0.00067386918102729, 0.0014417111589189976]])
    # mu = np.array([
    #     0.7037505720667833,
    #     -2.0015633043781254,
    #     -1.14648246759313,
    #     0.11264740019457169
    #     ])
    A = np.array([[-0.04287715757435157, 0.05372485611503212, 0.02930869611959398],
                  [0.038819627859144924, 0.0015364590207488983, 0.05397474027581484], 
                  [-0.0076783534791742625, -0.009284848285410607, 0.005786718230432543]])
    mu = np.array([
        0.6789978900546851,
        -1.9016249932538416,
        -1.138065025442601
    ])
    P = np.atleast_2d(P)
    X = mu + P @ A.T
    # print("in function", X, P @ A.T, mu)
    return X.squeeze()


def parabolic_profile(params, x, param_errs=None, purpose="velocity"):
    # profile = params[0] * np.sqrt(x - params[1]) + params[2]
    profile = params[0] * np.sqrt(x) + params[1]
    print("Parabolic profile {}".format(params))
    return profile


def generate_profile(params, x, param_errs=None, purpose="velocity", profile='poly'):
    if profile == 'parabola':
        profile = parabolic_profile(params, x)
        return profile
    vel_profile = 0
    vel_profile_err = 0
    # print("Generate_profile params", params)
    for n, p in enumerate(params[::-1]):
        # print('Deg', n, p)
        if purpose == 'velocity':
            deg = n
            # print(params)
        else:
            deg = n
        poly_deg = x**deg # shifted_legendre(n, x)
        vel_profile += poly_deg * p
        # if purpose == 'velocity':
        #     print('vel_profile', vel_profile, deg)
        if param_errs is not None:
            vel_profile_err += (poly_deg*param_errs[::-1][n]) ** 2
    if param_errs is not None:
        return vel_profile, np.sqrt(vel_profile_err)
    # print(vel_profile)
    return vel_profile        


def synt_spectra_for_wavelength(velocity, wl_array, synt_spectra=None, stellar_params=None, cont_divide=True, profile='poly'):
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
    temp_mask = temp_layers < 8000
    vel_profile = np.ones(np.size(temp_layers))
    temp_layers = temp_layers[temp_mask]
    x = (temp_layers - np.min(temp_layers)) / (np.max(temp_layers) - np.min(temp_layers))
    if velocity is None:
        transformed_flux = CubicSpline(synt_spectra['Wl'], synt_spectra['Flux'])(wl_array)
        return transformed_flux
    
    if (np.size(velocity)==1) and velocity is not None:
        velocity = velocity[0]
    if (np.size(velocity) > 1):
        # n_layers = 56
        # x = np.linspace(0, 1, n_layers)
        velocity = generate_profile(velocity, x, profile=profile)
    if synt_spectra is None:
        if isinstance(velocity, (int, float)):
            n_layers = 56
            # x = np.linspace(0, 1, n_layers)
            velocity = generate_profile([velocity], x)
            # print("Velocity profile", velocity)
            # velocity =
        vel_profile[temp_mask] = velocity
        vel_profile[~temp_mask] = velocity[-1]
        velocity = vel_profile
        # print("velocity", velocity)
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


def profile_penalty(params=None, param_pos=None, raise_params=None, fall_params=None, const_term=None, profile='poly'):
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
    temp_layers = np.load("data/Temp_layers.npy")
    temp_mask = temp_layers < 8000
    # vel_profile1 = np.ones(np.size(temp_layers))
    # vel_profile2 = np.ones(np.size(temp_layers))

    temp_layers = temp_layers[temp_mask]
    
    x = (temp_layers - np.min(temp_layers)) / (np.max(temp_layers) - np.min(temp_layers))# np.linspace(0, 1, n_layers)
    velocity_raise = generate_profile(raise_params, x, profile=profile)
    velocity_fall = generate_profile(fall_params, x, profile=profile)


    
    # vel_profile = np.ones(np.size(temp_layers))

    # print("Size", vel_profile1.size, temp_mask.size, velocity_raise.size)
    # vel_profile1[temp_mask] = velocity_raise
    # # vel_profile1[~temp_mask] = velocity_raise[-1]
    # vel_profile2[temp_mask] = velocity_fall
    # # vel_profile2[~temp_mask] = velocity_fall[-1]

    # vel_pro

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
    # fall_der_mask = fall_der < 0

    raise_der_penalty = np.sum(np.abs(raise_der[raise_der_mask]))
    # fall_der_penalty = np.sum(np.abs(fall_der[fall_der_mask]))

    # der_penalty = np.array([raise_der_penalty, fall_der_penalty]) * 1000000000
    der_penalty = np.array([raise_der_penalty])
        
    penalty_array = der_penalty * 100000 # 100000 # np.concatenate((prof_penalty, der_penalty)) * 1000000000
    if const_term is not None:
        penalty_array = np.concatenate((penalty_array, np.abs(const_term) * 0))
    # print("Penalty: {}".format(penalty_array))
    return penalty_array
    

def array_to_key(arr, tol=1e-10):
    return tuple((np.asarray(arr)/tol).astype(int))



def residue_profile(neid_data, synt_spectra=None,
                    raise_cache=None,
                    fall_cache=None,
                    param_pos=np.array(['r','f']),
                    ref_params=None,
                    profile='poly',
                    area_fact=0.5, # None,
                    scale_fact=None,
                    contnorm=True,
                    pca_comp=True,
                    required='residue',
                    ref_res=None,
                    ip_data='epoch',
                    sigmacut=None,
                    cache_dir='/data/varghese/Stellar_activity_project_results/spectra_cache/',
                    algorithm='ls'):
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
    
    # avgepoch_result = "/home/varghese/Desktop/Stellar_activity_mitigation/25_Avg/Result_avgdiff/Result_dir_Comb_spectra_3700-9000/fitted_params.pkl"
    # if ip_data == 'epoch':
    #     with open(avgepoch_result, 'rb') as avgres:
    #         avgepo = pickle.load(avgres)
    #     params = params + avgepo['params']
    # Parameter mask
    raise_mask = param_pos == 'r'
    fall_mask = param_pos == 'f'
    add_shift_mask = param_pos == 'v' # This is the additional velocity for doppler shift the entire spectra
    pca_components_mask = param_pos == 'p'
    lasso_comp = None
    scale_mask = param_pos == 'a'
    # additional_refshift = param_pos == 'rs'
    if np.size(param_pos) == 1:
            pca_comp = False
            
    # Extracting data
    neid_wl_array = dict_to_array(neid_data["Wave"])
    neid_flux_array = dict_to_array(neid_data["Flux"])
    neid_err_array = dict_to_array(neid_data["Err"])
    neid_telluricmask = dict_to_array(neid_data["mask"])
    # Sorting_wl_array
    sort_ind = np.argsort(neid_wl_array)
    neid_wl_array = neid_wl_array[sort_ind]
    neid_flux_array = neid_flux_array[sort_ind]
    neid_err_array = neid_err_array[sort_ind]
    neid_telluricmask = neid_telluricmask[sort_ind]
    diff_wl = np.diff(neid_wl_array) < 0
    # print(np.where(diff_wl))

    @lru_cache(maxsize=7)
    def cached_synthesis(raise_params_key, fall_params_key):
        raise_params = np.array(raise_params_key, dtype=float)
        print("Synthesise {} {}".format(raise_params_key, fall_params_key))
        spectra, cont = synt_spectra_for_wavelength(raise_params,
                                                    neid_wl_array,
                                                    synt_spectra=synt_spectra,
                                                    cont_divide=False,
                                                    profile=profile)
        
        if fall_params_key is not None:
            fall_params = np.array(fall_params_key, dtype=float)
            stellar_params = {'temp':5321}
            fall_spectra, fall_cont = synt_spectra_for_wavelength(fall_params,
                                                                  neid_wl_array,
                                                                  synt_spectra=synt_spectra,
                                                                  stellar_params=stellar_params,
                                                                  cont_divide=False,
                                                                  profile=profile)
            # plt.figure()
            # plt.plot(neid_wl_array, spectra)
            # plt.plot(neid_wl_array, fall_spectra / (cont+fall_cont))
            # plt.show()
            return spectra, cont, fall_spectra, fall_cont

        return spectra, cont

    def residue_fun(params):
        print("==============================================")
        print("Params: {}".format(params))
        itertime_beg = time.time()

        if ref_params is not None:
            params = ref_params + params
            # Additional velocity
        if np.sum(add_shift_mask) == 0:
            add_vel = 0
        else:
            add_vel = params[add_shift_mask]

        # Generating spectra
        lasso_comp = None
        if pca_comp:
            pca_components = params[pca_components_mask]
            lasso_comp = 0 * np.sqrt(np.abs(pca_components))
            print('pca_comps {}'.format(pca_components))
            fitting_params = reconstruct_params(pca_components)
            raise_params = fitting_params# [:-1]
            # print("raise params in if", raise_params)
            const_term = 0 # fitting_params[-1]
            # raise_params[-1] += const_term
            add_vel = add_vel + const_term
            const_term_penalty = add_vel
            # print('raise_params in cond', raise_params)
        else:
            raise_params = params[raise_mask]
            if np.size(raise_params) == 0:
                raise_params = np.array([0, 0, 0])
            elif np.size(raise_params) == 1:
                raise_params = np.array([0, 0, raise_params[0]])
            const_term_penalty = None
        # raise_params_added = np.insert(raise_params, 0, 0.66181421)
        print("raise_params {}".format(raise_params))
        raise_params_shifted = raise_params.copy() # add_vel
        profile_constant = raise_params_shifted[-1]
        raise_params_shifted[-1] = 0
        # raise_params_shifted[-1] += add_vel
        # print("added with {} {}".format(add_vel, raise_params_shifted))
        # raise_params
        raise_key = tuple(np.asarray(raise_params_shifted, dtype=float).tolist())
        # raise_spectra, raise_cont = cached_synthesis(raise_key)
        fall_spectra = 0
        # total_cont = raise_cont
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

            if profile == 'poly':
                fall_params = params_scale_fact * raise_params # np.insert(params[raise_mask], 0, 0.66181421)
            elif profile == 'parabola':
                fall_params = raise_params.copy()
                fall_params[0] = -1 * raise_params[0]
                fall_params[-1] = -1 * raise_params[-1] 
            falling_lane = True

        if falling_lane:

            # stellar_params = {'temp':5321} #5570}
            # fall_params[-1] += add_vel
            fall_params[-1] = 0
            print('fall params {}'.format(fall_params))
            filename_suffix = "_".join(map(str, raise_key))
            if pca_comp:
                cache_filename = "Raise_Fall_pca1_" + profile + "_" + filename_suffix + ".fits"
            else:
                cache_filename = "Raise_Fall_" + profile + "_" + filename_suffix + ".fits"
            cache_fullpath = os.path.join(cache_dir, cache_filename)
            if os.path.isfile(cache_fullpath):
                print("Cache file exists. Calling it")
                raise_spectra = fits.getdata(cache_fullpath, ext=1)
                raise_cont = fits.getdata(cache_fullpath, ext=2)
                fall_spectra = fits.getdata(cache_fullpath, ext=3)
                fall_cont = fits.getdata(cache_fullpath, ext=4)
                cache_wl = fits.getdata(cache_fullpath, ext=5)
                raise_spectra = CubicSpline(cache_wl, raise_spectra, extrapolate=True)(neid_wl_array)
                raise_cont = CubicSpline(cache_wl, raise_cont, extrapolate=True)(neid_wl_array)
                fall_spectra = CubicSpline(cache_wl, fall_spectra, extrapolate=True)(neid_wl_array)
                fall_cont = CubicSpline(cache_wl, fall_cont, extrapolate=True)(neid_wl_array)
            else:
                fall_key = tuple(np.asarray(fall_params, dtype=float).tolist())
                raise_spectra, raise_cont, fall_spectra, fall_cont = cached_synthesis(raise_key, fall_key)
                primary_hdu = fits.PrimaryHDU()
                hdu_raise_spe = fits.ImageHDU(data=raise_spectra, name="RAISE_SPEC")
                hdu_raise_cont = fits.ImageHDU(data=raise_cont, name="RAISE_CONT")
                hdu_fall_spe = fits.ImageHDU(data=fall_spectra, name="FALL_SPEC")
                hdu_fall_cont = fits.ImageHDU(data=fall_cont, name="FALL_CONT")
                hdu_wl = fits.ImageHDU(data=neid_wl_array, name="WAVE")
                hdul = fits.HDUList([primary_hdu,
                                     hdu_raise_spe,
                                     hdu_raise_cont,
                                     hdu_fall_spe,
                                     hdu_fall_cont,
                                     hdu_wl])
                hdul.writeto(cache_fullpath, overwrite=True)
                # print("Written cache", cache_fullpath)
            # plt.figure()
            # plt.plot(neid_wl_array, raise_spectra / (raise_cont + fall_cont))
            # plt.plot(neid_wl_array, fall_spectra / (raise_cont + fall_cont))
            # plt.show()
            
            raise_spectra = shifting_korg_flux(neid_wl_array, raise_spectra, profile_constant)
            fall_spectra = shifting_korg_flux(neid_wl_array, fall_spectra, -1*profile_constant)
            # print("Generating falling lane")
            penalty = profile_penalty(raise_params=raise_params, fall_params=fall_params, const_term=np.array([params[-1]]),
                                      profile=profile)
            if lasso_comp is not None:
                penalty = np.concatenate((lasso_comp, penalty))
                
            # params[-1])
            # params[-1])
            # if area_fact is None:
            #     a = params[scale_mask]
            # else:
            #     a = scale_fact

            total_flux = area_fact * raise_spectra + (1-area_fact) * fall_spectra # raise_spectra + ((1-a)/(1+a)) * fall_spectra
            total_cont = area_fact * raise_cont + (1-area_fact)*fall_cont
            if not contnorm:
                total_count = 1
            synt_spectra = total_flux / total_cont
            # plt.figure()
            # plt.plot(neid_wl_array, total_flux)
            # plt.plot(neid_wl_array, total_cont)
            # plt.plot(neid_wl_array, area_fact * raise_spectra, color='blue')
            # plt.plot(neid_wl_array, area_fact * raise_cont)
            # plt.plot(neid_wl_array, (1-area_fact) * fall_spectra, color='green')
            # print("Raise", area_fact * raise_spectra)
            # print("Fall", (1-area_fact) * fall_spectra)
            # plt.plot(neid_wl_array, (1-area_fact) * fall_cont)
            # plt.show()

        else:
            if not contnorm:
                raise_cont = 1
            synt_spectra = raise_spectra / raise_cont
            fall_spectra = np.nan
        synt_spectra = shifting_korg_flux(neid_wl_array, synt_spectra, add_vel) # Adding a global doppler shift
       #  print("*******Ref res", ref_res)
        if ref_res is not None:
            print("Residue subtraction happens")
            ref_residue = ref_res['residue']
            ref_err = ref_res['err']
            ref_wl = ref_res['Wl']
            residue_0 = CubicSpline(ref_wl, ref_residue)(neid_wl_array)
            ref_flux = ref_res['data']
            ref_synt = ref_res['synt']
            # print(additional_refshift)
            additional_refshift = add_shift_mask
            if np.sum(additional_refshift) > 0:
                print(" additional shift", params[additional_refshift])
                ref_flux = shifting_korg_flux(ref_wl, ref_flux, params[additional_refshift])
                ref_synt = shifting_korg_flux(ref_wl, ref_synt, params[additional_refshift])
                err_0 = shifting_korg_flux(ref_wl, ref_err, params[additional_refshift])
            flux0 = CubicSpline(ref_wl, ref_flux)(neid_wl_array)
            err_0 = CubicSpline(ref_wl, ref_err)(neid_wl_array)
            synt0 = CubicSpline(ref_wl, ref_synt)(neid_wl_array)
            # fig, axs = plt.subplots(3)
            # plt.plot(ref_wl, ref_flux)
            data_diff = neid_flux_array - flux0
            synt_diff = synt_spectra - synt0
            total_err = np.sqrt(neid_err_array**2 + err_0**2)
            # axs[0].plot(neid_wl_array, neid_flux_array, alpha=0.7)
            # axs[0].plot(ref_wl, ref_flux, alpha=0.7)
            # axs[0].plot(neid_wl_array, flux0, alpha=0.7)
            # axs[1].plot(neid_wl_array, synt_spectra - synt0)
            # axs[1].plot(neid_wl_array[neid_telluricmask == 1], (neid_flux_array - ref_flux)[neid_telluricmask == 1])
            # axs[1].plot(neid_wl_array, neid_telluricmask)
            err_0 = CubicSpline(ref_wl, ref_err)(neid_wl_array)
            err_mask = neid_telluricmask == 1
            # print(np.sum(err_mask), np.size(err_mask))

            # err_final = np.sqrt(neid_err_array**2 + err_0**2)
            # res_check = data_diff / err_final
            # print("Fundamental lower limit on residue", np.sum((res_check[err_mask])**2))
            # axs[2].plot(neid_wl_array[err_mask], neid_err_array[err_mask])

            # axs[2].plot(neid_wl_array[err_mask], err_0[err_mask])
            # plt.show()


            current_residue = (neid_flux_array - synt_spectra)
            # residue = (current_residue - residue_0) / total_err
            telluricmask = neid_telluricmask == 0
            if sigmacut is not None:
                specdiff = neid_flux_array - flux0
                ratio = specdiff / total_err
                sigmamask = ratio > sigmacut
                total_err[sigmamask] = total_err[sigmamask] * 10000
            total_err[telluricmask] = total_err[telluricmask] * 100000
            # plt.figure()
            # plt.plot(neid_wl_array, total_err)
            # plt.plot(neid_wl_array[telluricmask], total_err[telluricmask], 'ok')
            # plt.show()
            residue = (data_diff - synt_diff) / total_err
            residue_noerr = (current_residue - residue_0)
            err_array = total_err
        else:
            
            # res_0 = np.load('data/Reference_residue.npy')
            telluricmask = neid_telluricmask == 0
            neid_err_array[telluricmask] = neid_err_array[telluricmask] * 10000
            residue = (neid_flux_array - synt_spectra) / neid_err_array
            # plt.figure()
            # plt.plot(neid_wl_array, neid_flux_array)
            # plt.plot(neid_wl_array, synt_spectra)
            # plt.plot(neid_wl_array[telluricmask], neid_flux_array[telluricmask], '.k', alpha=0.5)
            # plt.ylim(0, 1.5)
            # plt.title("data and synt")
            # plt.show()
            residue_noerr = (neid_flux_array - synt_spectra) # - res_0
            residue_0 = residue_noerr
            current_residue = residue_0
        # print(f"removed {np.where(np.isnan(residue))}")
        nanmask = ~np.isnan(residue) & (neid_telluricmask == 1)
        residue_masked = residue# [nanmask]
        print("Penalty: {}".format(penalty))
        residue = np.concatenate((residue_masked, penalty))
        cost = np.sum(residue**2)/2
        no_points = np.sum(neid_telluricmask == 1)
        print("cost {}".format(cost))
        print("Reduced chi2: {}".format(cost / no_points))
        # plt.figure()
        # plt.plot(neid_wl_array, residue)
        # plt.show()

        print("Time taken for iteration: {}s".format(time.time() - itertime_beg))
        if required == 'residue':
            if algorithm == 'ls':                
                return residue
            elif algorithm == 'diff':
                square_sum = np.sum(np.concatenate((residue**2, penalty)))
                return square_sum
        elif required == 'spectra':
            raise_spectra = area_fact*raise_spectra/total_cont
            fall_spectra = (1-area_fact)*fall_spectra/total_cont
            spectra_dict = {"Raise":raise_spectra,
                            "Fall":fall_spectra,
                            "Total":synt_spectra,
                            "data":neid_flux_array,
                            "Res":residue[:-2],
                            "Res_noerr": residue_noerr,
                            "err": neid_err_array,
                            "r0": residue_0,
                            "r": current_residue,
                            "Wl":neid_wl_array,
                            "nanmask":nanmask}
            return spectra_dict

    return residue_fun
    # elif required == 'spectra':
        

def residue_ccf(params, neid_fname, opdir, ccf_function=None):
    print("Params", params)
    print("neid_fname", neid_fname)
    if ccf_function is None:
        from utils import save_synt_data as ccf_function
    ccf_function(params, neid_fname, opdir)

    data_ccf_fname = os.path.join(opdir, "Data_CCF.fits")
    synt_ccf_fname = os.path.join(opdir, "Synt_CCF.fits")

    data_ccf = fits.getdata(data_ccf_fname)[-1]
    synt_ccf = fits.getdata(synt_ccf_fname)[-1]
    os.remove(data_ccf_fname)
    os.remove(synt_ccf_fname)
    residue = data_ccf - synt_ccf
    return residue


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


def calculate_dotproduct_wrt_vel(lsq_res, residue_fun, resultdir):
    x = lsq_res.x # lsq_res.x
    # x = lsq_res
    result_dict = residue_fun(x)
    wavelength = result_dict['Wl']
    korg_spectra = result_dict['Total']
    step_size = 1e-6
    shifted_flux_pos = shifting_korg_flux(wavelength, korg_spectra, step_size)
    shifted_flux_neg = shifting_korg_flux(wavelength, korg_spectra, -1*step_size)
    derivative = (shifted_flux_pos - shifted_flux_neg) / (2*step_size)
    print("Calculated derivative of spectra")
    print(derivative)
    fig, axs = plt.subplots(np.size(x)+2, figsize=(16,16),sharex=True)
    fig.subplots_adjust(hspace=0)
    axs[0].plot(wavelength, korg_spectra)
    axs[0].set_ylabel("F")
    axs[1].plot(wavelength, derivative)
    axs[1].set_ylabel(r"$\frac{dF}{dV}$", fontsize=16)

    dot_products = []
    for i in range(np.size(x)):
        checking_param = x[i]
        x_copy = x.copy()
        print(x_copy)
        param_pos = checking_param + step_size
        x_copy[i] = param_pos
        result_pos = residue_fun(x_copy)
        spec_pos = result_pos['Total']
        param_neg = checking_param - step_size
        x_copy[i] = param_neg
        result_neg = residue_fun(x_copy)
        spec_neg = result_neg['Total']
        spec_der = (spec_pos - spec_neg) / (2*step_size)
        axs[i+2].plot(wavelength, spec_der)
        if i == (np.size(x)-1):
            axs[i+2].set_ylabel(r"$\frac{dF}{ddC}$", fontsize=16)
        else:
            axs[i+2].set_ylabel(rf'$\frac{{dF}}{{dP_{{{2-i}}}}}$', fontsize=16)

        dot_product = np.sum(derivative * spec_der)
        dot_products.append(dot_product)
    print("dot products {}".format(dot_products))
    plt.xlabel("Wavelength", fontsize=16)
    plt.savefig(resultdir+"/Derivatives.pdf")
    plt.show()
    return np.array(dot_products)

_result_fun = None

def init_worker(result_fun):
    global _result_fun
    _result_fun = result_fun

def process_r0(args):
    r0, c_array, a_array = args
    local_dict = {}
    for a in a_array:
        for c in c_array:
            params = np.array([r0, c, a])
            residue = np.sum(_result_fun(params)**2)
            local_dict[(r0, c)] = residue
    return local_dict

def explore_param_space(result, result_fun, resultdict, nproc=50):
    x = result.x
    R0 = x[0]
    C = x[1]
    add = x[2]
    print(R0, C, add)
    r0_array = np.arange(R0-0.01, R0+0.01, 0.001)
    c_array = np.arange(C-0.01, C+0.01, 0.001)
    add_array = np.arange(add-0.01, add+0.01, 0.005)
    with Pool(processes=nproc,
              initializer=init_worker,
              initargs=(result_fun,)) as pool:
        results = pool.map(
            process_r0,
            [(r0, c_array, add_array) for r0 in r0_array]
            )
        
    residue_dict = {}
    for d in results:
        residue_dict.update(d)
    # for r0 in r0_array:
    #     for c in c_array:
    #         params = np.array([r0, c])
    #         residue = np.sum(result_fun(params)**2)
    #         params_key = (r0, c)

    #         residue_dict[params_key] = residue
    #         # print(residue_dict)
    print("Saving residues")
    with open(resultdict + "/residue_dict.pkl", 'wb') as resd:
        pickle.dump(residue_dict, resd)
    print("Residue dictionary saved")
