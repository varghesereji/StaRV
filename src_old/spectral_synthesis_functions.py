import numpy as np

from PyAstronomy import pyasl

import juliapkg
from juliacall import Main as jl
jl.seval("using Korg"); Korg=jl.Korg
import time

# def generate_with_korg(velocity, wl_wind=None, temp=5770, logg=4.438, R=110000, window_size=4, continuum=False, fakedata=False):
def generate_with_korg(velocity, wl_wind=None, stellar_params=None, spectral_params=None, continuum=False, cont_divide=False):    
    '''
    generating stellar spectra with given temperature and logg
    velocity: array of velocity associated with each layer
    wl_wind: Wavelength window to generate the spectra
    stellar_params: temp, logg, vmic, vsini, epsilon.
    spectral_params: mu_values, R(resolution), window_size
    continuum: Bool. Keepint True will return the continuum.
    cont_divide: Bool. Keeping True will return the continuum and flux separately.
    '''
    # start = time.time()
    if wl_wind is None:
        wlmin = 3700
        wlmax = 9000
    else:
        wlmin, wlmax = wl_wind
    # if stellar_params is None:
    #     temp = 5770
    #     logg = 4.438
    #     vmic = 0.78
    #     vsini = 1.84
    #     epsilon = 0.6
    # else:
    #     temp, logg, vmic, vsini, epsilon = stellar_params
    defaults = {
        'temp': 5770,
        'logg': 4.438,
        'vmic': 0.78,
        'vsini': 1.84,
        'epsilon': 0.6
    }
    if stellar_params is not None:
        defaults.update(stellar_params)

    temp = defaults['temp']
    logg = defaults['logg']
    vmic = defaults['vmic']
    vsini = defaults['vsini']
    epsilon = defaults['epsilon']
    if spectral_params is None:
        mu_vals = 100
        R = 110000
        window_size = 4
    # print("velocity array", velocity)
    # print("But trying without velocity array now")
    lines = Korg.get_VALD_solar_linelist()
    # print("Called linelist")
    A_X = Korg.format_A_X(0)
    

    atm = Korg.interpolate_marcs(temp, logg, A_X)
    sol = Korg.synthesize(atm, lines, A_X, wlmin, wlmax,
                          mu_values=100,
                          vmic=0.78,
                          verbose=False,
                          I_scheme="linear_flux_only",
                          velocity_profile=velocity)
    flux = sol.flux
    cont = sol.cntm
    wavelengths = sol.wavelengths
    if not continuum:
        conv_LSF = Korg.apply_LSF(flux, wavelengths, R, window_size=window_size)
        conv_flux = np.array(conv_LSF)
        cont = np.array(cont)
        rbflux = Korg.apply_rotation(conv_LSF, wavelengths, vsini, epsilon)
        if not cont_divide:
            korg_data = {'Flux':np.array(rbflux), 'Wl':np.array(wavelengths), 'cont':cont}
        else:
            korg_data = {'Flux':np.array(rbflux/cont), 'Wl':np.array(wavelengths)}
        # print("Korg_data", korg_data)
        del sol
        del conv_LSF
        del conv_flux
        del wavelengths
        del flux
        del cont
        del rbflux
        # end = time.time()
        # print(end-start)
        return korg_data
    else:
        del sol
        del flux
        return {'cont':cont, 'Wl':wavelengths}


def generate_full_korgspectra(velocity, wlwindow=(3700, 9000)):
    '''
    This function is to generate the spectra with Korg for large windows.
    '''
    window = 100
    wl_regions = np.arange(wlwindow[0], wlwindow[1], window)
    full_flux = np.array([])
    full_wl = np.array([])
    for n, wl in enumerate(wl_regions[:-1]):
        print(wl, wl_regions[n+1])
        spectra = generate_with_korg(velocity=velocity, wl_wind=(wl-2, wl_regions[n+1]+2))
        flux = spectra['Flux'][10:-10]
        wl = spectra['Wl'][10:-10]
        full_flux = np.concatenate((full_flux, flux))
        full_wl = np.concatenate((full_wl, wl))
    # Making the wavelength array into strictly increasing function
    wl_dict = defaultdict(list)
    for wl_i, flux_i in zip(full_wl, full_flux):
        wl_dict[wl_i].append(flux_i)
    wl_unique = np.array(sorted(wl_dict.keys()))
    flux_avg = np.array([np.mean(wl_dict[wl_i]) for wl_i in wl_unique])
    korg_spectra = {'Flux':flux_avg,
                    'Wl': wl_unique}
    return korg_spectra
