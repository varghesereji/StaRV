import numpy as np
import matplotlib.pyplot as plt
import os
import configparser
import pickle
from scipy.interpolate import CubicSpline

from utils import call_neiddata_full
from utils import save_dict_to_pickle


def concatenate_dict_to_array(inputdict):
    '''
    This function is to convert the dictnories which the keys are the order and values are wavelength or flux.
    '''
    order_list = list(inputdict.keys())
    # print(order_list)
    flat_array = np.array([])
    for order in order_list:
        subarray = inputdict[order]
        flat_array = np.concatenate((flat_array, subarray))
    return flat_array

    
configfile = 'Spectral_fitting.config'
config = configparser.ConfigParser()
config.read(configfile)

data_path = config['data_dir']['NEID_DIR']

file_list = os.listdir(data_path)

with open('data/Ref_spectra_korg.pkl', 'rb') as ref:
    refspec = pickle.load(ref) 

flux = np.array([])
wave = np.array([])

for epoch in file_list:
    print("Epoch:", epoch)
    epoch_data = call_neiddata_full(neid_filename=os.path.join(data_path, epoch),
                                    resultdict='avg_data',
                                    refspec=refspec,
                                    verbose=False)
    flatten_wave = concatenate_dict_to_array(epoch_data['Wave'])
    flatten_flux = concatenate_dict_to_array(epoch_data['Flux'])
    wlsort = np.argsort(flatten_wave)
    flatten_wave = flatten_wave[wlsort]
    flatten_flux = flatten_flux[wlsort]
    # print("Flatten_flux", np.shape(flatten_flux))
    if np.size(wave) == 0:
        wave = flatten_wave
        flux = flatten_flux
    else:
        transformed_flux = CubicSpline(flatten_wave, flatten_flux)(wave)
        flux = np.vstack((flux, transformed_flux))


median_flux = np.nanmedian(flux, axis=0)
avg_neid_data = {"Flux":median_flux,
                 "Wl":wave}

opdir = os.path.join('data', 'Avg_NEID_data.pkl')
save_dict_to_pickle(avg_neid_data, opdir)
    
    

