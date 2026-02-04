# This code is to estimate the temperature of the spectra for intergranular lane
# given the granulation contrast in Abramenko (2012)

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from functools import *

from spectral_synthesis_functions import generate_with_korg


def define_granulation_contrast(Fg, Fig, a):
    F0 = a*Fg + (1-a) * Fig
    # print("Fg: {} Fig: {}".format(Fg, Fig))
    G = np.sqrt(a*(Fg-F0)**2 + (1-a)*(Fig - F0)**2) / F0
    return G


def residue_temp_gc(temp, Fg, req_wl=7050, wl_wind=(7000, 7100), scale_fact=1, area_fact=1/2):
    # print("Temperature: {}".format(temp))
    stellar_params = {'temp':temp[0]}
    spectra_ig = generate_with_korg(velocity=False, wl_wind=wl_window, stellar_params=stellar_params,
                                    cont_divide=False)
    flux = spectra_ig['Flux'][10:-10] / scale_fact
    wl = spectra_ig['Wl'][10:-10]
    fluxig_7050 = CubicSpline(wl, flux)(req_wl)
    calc_G = define_granulation_contrast(Fg, fluxig_7050, a=area_fact)
    # print("calc_G", calc_G)
    residue = 0.15 - calc_G
    return residue
    

def residue_temp(temp, F_ig, req_wl=7050, wl_wind=(7000, 7100), scale_fact=1):
    print("Temperature: {}".format(temp))
    stellar_params = {'temp':temp[0]}
    spectra_ig = generate_with_korg(velocity=False, wl_wind=wl_window, stellar_params=stellar_params,
                                    cont_divide=False)
    flux = spectra_ig['Flux'][10:-10] / scale_fact
    wl = spectra_ig['Wl'][10:-10]

    fluxig_7050 = CubicSpline(wl, flux)(req_wl) # / scale_fact
    print("new_flux: {}".format(fluxig_7050))
    print("Flux needed: {}".format(F_ig))

    residue = np.array([fluxig_7050 - F_ig])
    print("Difference: {}".format(residue))

    print("===================")
    return residue


wl_window = (7000, 7100)
stellar_params = {'temp':5770}

req_wl = 7050

# Granular layer
spectra_g = generate_with_korg(velocity=False, wl_wind=wl_window, stellar_params=stellar_params, cont_divide=False)
flux = spectra_g['Flux'][10:-10]
wl = spectra_g['Wl'][10:-10]
flux_7050 = CubicSpline(wl, flux)(req_wl)
# subscript g stands for intergranular lane

gc = 0.15

scale_fact = np.nanmedian(flux)

# Here, keeping the area to half
F_ig = ((1-gc)/(1+gc)) * flux_7050 / scale_fact # flux_7050

# Question is if this is the flux, what will be the temperature. Temperature is the free parameter.

residue_function = partial(residue_temp, F_ig=F_ig, req_wl=req_wl,
                            wl_wind=wl_window, scale_fact=scale_fact)
result = least_squares(residue_function, x0=5321, jac='2-point',
                       bounds=(-np.inf, np.inf), method='trf',
                       verbose=2)
# print(result)
print("The fitted temperature is (Method 1): {}".format(result.x))

plt.figure(figsize=(16,8))
area_fact_array = np.arange(0.3, 0.7, 0.01)
for area_fact in area_fact_array:
    residue_function = partial(residue_temp_gc, Fg=flux_7050/scale_fact, req_wl=7050, wl_wind=(7000, 7100), scale_fact=scale_fact, area_fact=area_fact)
    result = least_squares(residue_function, x0=5470, jac='2-point',
                           bounds=(-np.inf, np.inf), method='trf',
                           verbose=2)
    # print(result)
    plt.plot(area_fact, result.x[0], 'o')
    print("The fitted temperature is (Method 2): {}".format(result.x))
plt.savefig("area_vs_igtemp.pdf")
