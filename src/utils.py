import csv
import os
from astropy.io import fits
import numpy as np
import pickle
from collections import defaultdict
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from functools import *
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import airvacuumvald as avv
# import mpld3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

from model_functions import generate_with_korg, generate_profile
from model_functions import synt_spectra_for_wavelength
from model_functions import shifting_korg_flux


def calling_linelist(filename):
    '''
    Function to call the line from the file saved in csv format.
    In the csv file, first column is the left side of the window, second column is the right side
    and third column is the centroid of the line.
    Returns a dictonary. Key is a float which is the centroid of the line.
    Values are lists, which first element is the infremum and second element is the supremum of the window.
    '''
    line_dict = {}
    with open(filename, 'r') as f:
        csvFile = csv.reader(f)
        for n, lines in enumerate(csvFile):
            if n==0:
                continue
            if lines[0][0] == '#':
                continue
            line_dict[float(lines[2])] = [float(lines[0]), float(lines[1])]
    return line_dict


def call_neid_data(order=100, neid_filename='/home/varghese/Desktop/Stellar_activity_mitigation/neidL2_20220110T180951.fits'):
    '''
    This function is to call NEID data, continuum divided using RASSINE.
    Input is the order of the spectra. It call the corresponding file, and extract the flux and wavelength.
    '''

    fsr_mask = fits.getdata('../neidMaster_FSR_Mask20210218_v002.fits')[order]
    master_file = '/home/varghese/Desktop/Stellar_activity_mitigation/neidL2_20220110T180951.fits'
    neid_filename = neid_filename
    # print("NEID filename: {}".format(neid_filename))
    neid_flux = fits.getdata(neid_filename, ext=1)[order]# [~fsr_mask]
    neid_var = fits.getdata(neid_filename, ext=4)[order]# [~fsr_mask]
    sciblaze = fits.getdata(master_file, ext=15)[order]# [~fsr_mask]
    wl = fits.getdata(neid_filename, ext=7)[order]# [~fsr_mask]
    wl = np.ma.MaskedArray(wl, fsr_mask).filled(np.nan)
    scaled_flux = neid_flux / sciblaze # / neid_flux[0]
    scaled_var = neid_var / sciblaze**2 # (np.nanmedian(neid_flux))**2
    flux_err = np.sqrt(scaled_var)
    strnum = str(173-order)
    while len(strnum) < 3:
        strnum = '0' + strnum
    zfact = fits.getheader(neid_filename)['SSBZ'+strnum]
    # print("SSSBZ{}:{}".format(strnum, zfact))
    # print("Wl array after multiply", wl.astype(np.float64)*(1+zfact))
    return wl.astype(np.float64)*(1+zfact), scaled_flux, flux_err.astype(np.float64)


def call_neiddata_full(neid_filename, resultdict, refspec=None, scaled_order=True, save_interactive_plots=False,
                       verbose=False, ref_velocity=0):
    "This function is to call all the orders of neid data"
    '''
    neid_filename: The name of the NEID file.
    resultdict: The dictionary to save the plots
    refspec: Reference spectra. If None, the function will generate the Korg spectra for reference. Time consuming.
    scaled_order: Do the scaling of order by fitting polynomial. False if not needed.
    '''
    # print = __builtins__.print if verbose else lambda *a, **k: None
    neid_arrays = 121
    neid_data = {'Flux':{},
                 'Wave':{},
                 'Err':{}}
    pdf = PdfPages(resultdict+"/Generated_spectra.pdf")
    # fig, axs = plt.subplots(3, sharex=True)
    # plotlyfig = make_subplots(
    #     rows=2, cols=1, shared_xaxes=True,
    #     # subplot_titles=("Subplot 1", "Subplot 2"),
    #     vertical_spacing=0.02
    # )
    buttons = []
    # index_list = [88, 87, 86, 85, 84, 80]
    for index in range(neid_arrays):
        if ((173-index) < 70) or ((173-index) > 162) :# 69 165):
            continue
        # if (173-index) not in index_list:
        #     continue

        # print("Working on order {} (index {})".format(173-index, index))
        neid_wl, neid_flux, neid_err = call_neid_data(index, neid_filename)
        wlmask = np.isnan(neid_wl) | (neid_wl < 1000) | np.isinf(neid_flux) | np.isnan(neid_flux) | np.isnan(neid_err)
        if np.sum(wlmask) == np.size(neid_wl):
            print("This index is not useful")
        else:
            filtered_wl = neid_wl[~wlmask]
            filtered_flux = neid_flux[~wlmask]
            filtered_err = neid_err[~wlmask]

            telluric_mask = mask_creation(filtered_wl)
            
            filtered_wl = filtered_wl[telluric_mask]
            filtered_flux = filtered_flux[telluric_mask]
            filtered_err = filtered_err[telluric_mask]
            # print("Ref velocity {}".format(ref_velocity))

            if abs(ref_velocity) != 0:
                # print("Applying the doppler shift of {} km/s".format(ref_velocity))
                filtered_flux = shifting_korg_flux(filtered_wl, filtered_flux, ref_velocity)
                filtered_err = shifting_korg_flux(filtered_wl, filtered_err, ref_velocity)
                
            
            fig, axs = plt.subplots(3, sharex=True)
            if save_interactive_plots:
                plotlyfig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    # subplot_titles=("Subplot 1", "Subplot 2"),
                    vertical_spacing=0.02
                )
                # plot_plotly(plotlyfig, filtered_wl, scaled_flux, 1, 1, 'red', 'NEID')
            else:
                plotlyfig = False
                
            axs[0].plot(filtered_wl, filtered_flux, color='red', label="NEID")
            # if plotlyfig !=False:
            #     plot_plotly(plotlyfig, filtered_wl, filtered_flux, 1, 1, 'red', 'NEID')
            if scaled_order:
                scaled_flux, scaled_err = order_scaleing(filtered_flux, filtered_wl, filtered_err, axs,
                                                         korg_spectra=refspec, plotlyfig=plotlyfig,
                                                         verbose=verbose)
            else:
                scaled_flux = filtered_flux
                scaled_err = filtered_err
            
            axs[1].plot(filtered_wl, scaled_flux, color='red')

            fig.suptitle("Order {}".format(173-index))
            axs[0].legend()
            plt.subplots_adjust(hspace=0)
            # axs[1].legend()
            plt.subplots_adjust(hspace=0)
            pdf.savefig()
            plt.close()
            neid_data["Flux"][173-index] = scaled_flux
            neid_data["Wave"][173-index] = filtered_wl
            neid_data["Err"][173-index] = scaled_err
            espresso_file = '../G2_espresso.txt'
            espresso_lines = import_espresso_lines(espresso_file)
            # for cent in espresso_lines:
            #     # print(cent)
            #     if (np.nanmin(filtered_wl) < cent) & (np.nanmax(filtered_wl) > cent):
            #     # print(cent)
            #         plotlyfig.add_shape(
            #             type='line',
            #             x0=cent, x1=cent,         # vertical line → constant x
            #             y0=0, y1=5,         # full height → from bottom to top
            #             line=dict(color='goldenrod', width=2, dash='dash'),
            #             xref='x2',          # subplot 2 x-axis
            #             yref='paper'        # always full height of subplot
            #         )
            # visibility = [False] * neid_arrays
            # visibility[index] = True

            # add button config
            # buttons.append(dict(label=f'Order {173-index}',
            #                     method='update',
            #                     args=[{'visible':visibility},
            #                           {'title':'Spectra'}]))
            spectra_dir = os.path.join(resultdict, 'Spectra')
            if save_interactive_plots:
                if not os.path.exists(spectra_dir):
                    os.makedirs(spectra_dir)
                pyo.plot(plotlyfig, filename=spectra_dir+"/NEID_spectra_comparison_order{}.html".format(173-index))# )format(173-index))
    
            # html_str = mpld3.fig_to_html(fig)
            # with open(resultdict+"/Spectra/NEID_spectra_comparison{}.html".format(173-index), 'w') as f:
            #     f.write(html_str)
    pdf.close()
    # html_str = mpld3.fig_to_html(fig)
    # with open(resultdict+"/NEID_spectra_comparison.html", 'w') as f:
    #     f.write(html_str)
    return neid_data


def generate_fakedata(velocity_params, snr, reference_file, resultdict):
    '''
    This function will create a fake data with given SNR
    velocity_params: Parameters of velocity profile
    snr: Required SNR
    reference_file: NEID file to refer the wavelength region
    resultdict: The directory to save the results
    '''
    fakedata_name = os.path.join(resultdict, "Fake_NEID_data.pkl")
    if os.path.exists(fakedata_name): # If the fake data already exists, use that. 
        print("The fakedata already exists")
        with open(fakedata_name, 'rb') as fakedata:
            korg_result = pickle.load(fakedata)
        return korg_result

    print("Generating fake data with velocity params {}, SNR {}".format(velocity_params, snr))
    vel_profile = generate_profile(velocity_params, np.linspace(0, 1, 56))
    print("vel_profile", vel_profile)
    reference_data = call_neiddata_full(reference_file, resultdict, scaled_order=False)
    reference_wls = reference_data["Wave"]
    # np.set_printoptions(threshold=np.inf)
    orders_list = list(reference_wls.keys())

    
    # Generating the scale factor to convert into the mean of poisson distribution
    synt_spectra = generate_with_korg(vel_profile, wl_wind=(3700, 9000), # wl_wind=(6000-0.02, 6000+0.02),
                                     fakedata=True)
    print("Spectrum synthesis done")
    snr_ref_flux = synt_spectra_for_wavelength(velocity=None, wl_array=6000, synt_spectra=synt_spectra)
                                        
    # snr_ref_flux = # snr_ref_spectra['Flux']
    median_flux = np.median(snr_ref_flux)
    scale_factor = snr**2/median_flux
    # del snr_ref_spectra

    korg_result = {'Flux':{},
                   'Wave':{},
                   'Err':{}}
    pdf = PdfPages(resultdict+"/Generated_spectra.pdf")
    for order in orders_list: # Iterations over orders
        # print(order)
        print("Working on order {}".format(order))
        wl_order = reference_wls[order]
        # print(wl_order)
        wlmask = np.isnan(wl_order) | (wl_order < 1000) # Removing useless portions of wavelength array
        # print(np.sum(wlmask), np.size(wl_order))
        if np.sum(wlmask) == np.size(wl_order):
            korg_result['Flux'][order] = np.zeros(np.size(wl_order)) + np.nan
            korg_result['Err'][order] = np.zeros(np.size(wl_order)) + np.nan
            korg_result['Wave'][order] = np.zeros(np.size(wl_order)) + np.nan
            # continue
        else:
            filtered_wl = wl_order[~wlmask]
            wlmin, wlmax = np.nanmin(filtered_wl), np.nanmax(filtered_wl)
            
            # korg_data = generate_with_korg(vel_profile, wl_wind=(wlmin-2,wlmax+2), fakedata=True) # Generating the spectrum
            korg_flux = synt_spectra['Flux']
            korg_wl = synt_spectra['Wl']
            cont = synt_spectra['cont'] * scale_factor # Normalizing the continuum to required SNR level.
            flux_shifted_to_neid = CubicSpline(korg_wl, korg_flux)(filtered_wl) # Transforming Korg flux onto NEID wavelength
            cont = CubicSpline(korg_wl, cont)(filtered_wl) # Transforming Korg continuum onto NEID wavelength
            scaled_korg_flux = flux_shifted_to_neid * scale_factor
            # print(scaled_korg_flux)
            np.random.seed(42)
            generated_flux = np.random.poisson(scaled_korg_flux)
            
            cont_norm_flux = generated_flux / cont # Continuum normalization
            cont_norm_err = np.sqrt(scaled_korg_flux / cont**2) # We are using the mean of the poisson distribution as the variance.
            fig, axs = plt.subplots(3, figsize=(16,9), sharex=True)
            axs[0].plot(filtered_wl, cont_norm_flux, color='red', label="NEID")
            axs[0].legend()
            # print("Finding the scale factor for order {}".format(order))
            scaled_flux, scaled_err = order_scaleing(cont_norm_flux, filtered_wl, cont_norm_err, axs)
            axs[1].plot(filtered_wl, scaled_flux, color='red')
            plt.subplots_adjust(hspace=0)
            korg_result['Flux'][order] = scaled_flux
            korg_result['Err'][order] = scaled_err
            korg_result['Wave'][order] = filtered_wl
            pdf.savefig()
            plt.close()
    pdf.close()
    print("################# Generated Fake data ##########################")
    with open(fakedata_name, 'wb') as fakedata:
        pickle.dump(korg_result, fakedata)
    return korg_result





def order_scaling_residue(params, neid_flux, neid_wl, neid_err, korg_flux, profile_fitting=True):
    # The function to get the continuum profile of the spectra.
    from model_functions import generate_profile

    shifted_wlarr = neid_wl - np.nanmin(neid_wl)
    mapped_wlarr = shifted_wlarr / np.nanmax(shifted_wlarr)
    # print(neid_wl)
    # print(mapped_wlarr)
    profile = generate_profile(params, mapped_wlarr)
    if profile_fitting:
        # print('profile_fitting')
        # print(params)
        # print(profile)
        scaled_flux = neid_flux * profile
        residue = (scaled_flux - korg_flux) / neid_err
        # print("Residue", residue, "chi2", np.sum(residue**2))
        # print(residue)
        return residue
    else:
        return profile
    

    


def order_scaleing(neid_flux, neid_wl, neid_err, axs, korg_spectra=None, plotlyfig=None,verbose=True):
    'This function is to find the polynomial which is using to scale the spectrum'
    vel_profile = False # generate_profile(velocity_params, np.linspace(0, 1, 56))
    if verbose==True:
        verbose=1
    else:
        verbose=0
    if korg_spectra is None:
        print("Pre-defined korg spectra does not exist. Generating new")
        korg_spectra = generate_with_korg(vel_profile,
                                          wl_wind=(np.nanmin(neid_wl)-2, np.nanmax(neid_wl)+2)
                                          )
    else:
        korg_spectra = korg_spectra
    korg_flux = korg_spectra['Flux']
    korg_wl = korg_spectra['Wl']
    transformed_korg_flux = CubicSpline(korg_wl, korg_flux)(neid_wl)

    init_cond = [1, 0, 0]
    residue_function = partial(order_scaling_residue, neid_flux=neid_flux,
                               neid_wl=neid_wl,
                               neid_err=neid_err,
                               korg_flux=transformed_korg_flux)
    fitted_result = least_squares(residue_function, x0=init_cond,
                                  method='trf',
                                  verbose=verbose)
    # print(fitted_result)
    fitted_params = fitted_result.x
    # print("Fitted_scale_factor:", fitted_params)
    profile = residue_function(fitted_params, profile_fitting=False)
    
    axs[0].plot(neid_wl, transformed_korg_flux, color='k', label="Korg")
    axs[1].plot(neid_wl, transformed_korg_flux, color='k')
    axs[0].plot(neid_wl, profile, label='Scale factor')

    residue = residue_function(fitted_params, profile_fitting=True)
    
    axs[2].plot(neid_wl, residue, color='b', label='residue')
    # axs.add_trace(go.Scatter(x=neid_wl, y=residue, color='blue', name='residue'),
    grad_korg = np.gradient(transformed_korg_flux, axis=0) / np.gradient(neid_wl, axis=0)
    grad2_korg = np.gradient(grad_korg, axis=0) / np.gradient(neid_wl, axis=0)
    axs[2].plot(neid_wl, grad_korg, color='g', alpha=0.6, label='d Korg')
    axs[2].plot(neid_wl, grad2_korg, color='orange', alpha=0.4, label='d2 Korg')
    
    axs[2].legend()
    scaled_flux = neid_flux * profile
    scaled_err = neid_err * profile

    if plotlyfig:
        # plot_plotly(plotlyfig, neid_wl, transformed_korg_flux, 1, 1, 'black', "Korg")
        plot_plotly(plotlyfig, neid_wl, transformed_korg_flux, 1, 1, 'black', "Korg")
        plot_plotly(plotlyfig, neid_wl, scaled_flux, 1, 1, 'red', "Korg")
        # plot_plotly(plotlyfig, neid_wl, profile, 1, 1, 'green', "Scale")
        plot_plotly(plotlyfig, neid_wl, residue, 2, 1, 'black', "residue")
        plot_plotly(plotlyfig, neid_wl, grad_korg, 2, 1, 'green', "d Korg")
        # plot_plotly(plotlyfig, neid_wl, grad2_korg, 3, 1, 'green', "d2 Korg")
        
        
    return scaled_flux, scaled_err

    
def mask_creation(wavelengths):
    '''
    This function is to generate a mask for telluric regions
    '''
    masking_region = []
    with open('data/Masked_regions.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            masking_region.append((float(row[0]), float(row[1])))
    mask = np.ones_like(wavelengths, dtype=bool)
    # print(wavelengths)
    for start, end in masking_region:
        # if (np.nanmin(wavelengths) <= start) & (np.nanmax(wavelengths) >= end):
            # print(np.nanmin(wavelengths)
            # print(start, end)
        mask &= ~((wavelengths >= start) & (wavelengths <= end))
        # print(start, end, mask)
    # print(np.size(wavelengths), np.sum(mask))
    return mask
    

    
    

def save_dict_to_pickle(dictionary, file_path):
    """
    Save a dictionary to a pickle file.
    
    Args:
        dictionary (dict): The dictionary to save.
        file_path (str): The path where the pickle file will be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)
    print(f"Dictionary saved to {file_path}")


def import_espresso_lines(filename):
    # Function to call the ESPRESSO lines
    cent_list = []
    with open(filename, 'r') as maskfile:
        for i in maskfile:
            line = i.strip().split(" ")
            cent = float(line[0])
            cent_vac = avv.air_to_vacuum(cent)
            cent_list.append(cent_vac)
    return cent_list


# Plotting

def plot_plotly(fig, x, y, row, col, color,name):
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines',line=dict(color=color),name=name), row=row, col=col)


def plot_matplotlib(axs, x, y, row, col, **kwargs):
    axs[row].plot(x, y, **kwargs)


def plotting_spectra(neid_data, korg_data, filename, interactive=False):

    orders_list = list(neid_data["Flux"].keys())
    pdf = PdfPages(filename)
    korg_raise = korg_data["Raise"]
    korg_fall = korg_data["Fall"]
    korg_total = korg_data["Total"]
    korg_residue = korg_data["Res"]
    korg_wl = korg_data["Wl"]

    for order in orders_list:
        fig, axs = plt.subplots(2, figsize=(16, 8), sharex=True)
        neid_flux = neid_data["Flux"][order]
        neid_wls = neid_data["Wave"][order]
        neid_wl_mask = np.isin(korg_wl, neid_wls)
        # print(np.sum(neid_wl_mask), np.size(neid_wls))
        # print(neid_wls)
        # plt.figure()
        # plt.plot(neid_wls)
        # plt.savefig("NEID_wls.png")
        # diff_neid_wls = np.diff(neid_wls)
        # diff_mask = diff_neid_wls < 0
        # print(np.sum(diff_mask))
        korg_raise_order = korg_raise[neid_wl_mask]
        korg_total_order = korg_total[neid_wl_mask]
        korg_residue_order = korg_residue[neid_wl_mask]
        plot_matplotlib(axs, neid_wls, neid_flux, 0, 0, color="black", label="NEID")
        plot_matplotlib(axs, neid_wls, korg_raise_order, 0, 0, color="red", label="Korg Raise")
        plot_matplotlib(axs, neid_wls, korg_total_order, 0, 0, color="blue", label="Korg Total")
        plot_matplotlib(axs, neid_wls, korg_residue_order, 1, 0, color="blue", label="Residue")

        if korg_fall is not None:
            korg_fall_order = korg_fall[neid_wl_mask]
            plot_matplotlib(axs, neid_wls, korg_fall_order, 0, 0, color="green", label="Korg Fall")
        

        axs[0].legend()
        plt.subplots_adjust(wspace=0, hspace=0)

        fig.suptitle("NEID spectra order {}".format(order))
        axs[1].set_xlabel("Wavelength $\AA$")
        axs[0].set_ylabel("Flux")
        axs[1].set_ylabel("Residue")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    pdf.close()
    


def plot_lines(neid_data, korg_data, filename, mask_filename='data/sol_line_window.csv'):

    # Reading the csv file
    lines_list = calling_linelist(mask_filename)
    print(len(lines_list.keys()))

    korg_raise = korg_data["Raise"]
    korg_fall = korg_data["Fall"]
    korg_total = korg_data["Total"]

    from model_functions import dict_to_array
    neid_flux = dict_to_array(neid_data["Flux"])

    fig, axs = plt.subplots(6, 3, figsize=(16, 16))

    index = 0
    for cent, edges in lines_list.items():
        wl_array = korg_data['Wl']

        wl_mask = (wl_array > edges[0]) & (wl_array < edges[-1])

        wl_masked = wl_array[wl_mask]
        raise_masked = korg_raise[wl_mask]
        fall_masked = korg_fall[wl_mask]
        total_masked = korg_total[wl_mask]

        neid_masked = neid_flux[wl_mask]
        
        axs[index//3, index%3].plot(wl_masked, raise_masked, color='blue')
        axs[index//3, index%3].plot(wl_masked, fall_masked, color='red')
        axs[index//3, index%3].plot(wl_masked, total_masked, color='green')
        axs[index//3, index%3].plot(wl_masked, neid_masked, color='black')
        index += 1
    plt.tight_layout()
    plt.savefig(filename)

        

    
    
    
# Example usage
# my_dict = {'name': 'Alice', 'age': 25, 'city': 'Wonderland'}
# save_dict_to_pickle(my_dict, 'my_dictionary.pkl')

