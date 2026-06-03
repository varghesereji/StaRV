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
import shutil
# import mpld3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

from model_functions import generate_with_korg, generate_profile
from model_functions import synt_spectra_for_wavelength
from model_functions import shifting_korg_flux

from PRVccf.modules import neid_calculate_mask_RV

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
    This function is to call NEID data.
    Input is the order of the spectra. It call the corresponding file, and extract the flux and wavelength.
    '''

    fsr_mask = fits.getdata('../neidMaster_FSR_Mask20210218_v002.fits')[order]
    master_file = '/home/varghese/Desktop/Stellar_activity_mitigation/neidL2_20220110T180951.fits'
    neid_filename = neid_filename
    # print("NEID filename: {}".format(neid_filename))
    neid_flux = fits.getdata(neid_filename, ext=1)[order]# [~fsr_mask]
    neid_var = fits.getdata(neid_filename, ext=4)[order]# [~fsr_mask]
    sciblaze = fits.getdata(neid_filename, ext=15)[order]# [~fsr_mask]
    wl = fits.getdata(neid_filename, ext=7)[order]# [~fsr_mask]

    tellurics = fits.getdata(neid_filename, ext=10)
    full_tellurics = tellurics[:, :, 0] * tellurics[:, :, 1]
    neid_flux = neid_flux / full_tellurics[order]
    neid_var = neid_var / (full_tellurics[order])**2
    # print(wl, np.all(fsr_mask))
    # np.set_printoptions(threshold=np.inf)
    # print(fsr_mask)
    wl = np.ma.MaskedArray(wl, fsr_mask).filled(np.nan)
    # np.set_printoptions(threshold=np.inf)
    # print("masked", wl)
    scaled_flux = neid_flux / sciblaze # / neid_flux[0]
    scaled_var = neid_var / sciblaze**2 # (np.nanmedian(neid_flux))**2
    flux_err = np.sqrt(scaled_var)
    strnum = str(173-order)
    while len(strnum) < 3:
        strnum = '0' + strnum
    zfact = fits.getheader(neid_filename)['SSBZ'+strnum]
    # print("SSSBZ{}:{}".format(strnum, zfact))
    # print("Wl array after multiply", wl.astype(np.float64)*(1+zfact))
    # print(zfact)
    # plt.close('all')
    # plt.figure()
    # plt.plot(np.sqrt(neid_var) / neid_flux)
    # plt.plot(flux_err / scaled_flux)
    # plt.show()
    # fig, axs = plt.subplots(2)
    # axs[0].plot(scaled_flux)
    # axs[1].plot(flux_err)
    # plt.title(order)
    # plt.show()
    return wl.astype(np.float64)*(1+zfact), scaled_flux, flux_err.astype(np.float64)


def call_neiddata_full(neid_filename, resultdict, refspec=None, scaled_order=True, save_interactive_plots=False,
                       verbose=False, ref_velocity=0, sel_order=None):
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
                 'Err':{},
                 'mask':{}}
    pdf = PdfPages(str(resultdict)+"/Generated_spectra.pdf")
    # fig, axs = plt.subplots(3, sharex=True)
    # plotlyfig = make_subplots(
    #     rows=2, cols=1, shared_xaxes=True,
    #     # subplot_titles=("Subplot 1", "Subplot 2"),
    #     vertical_spacing=0.02
    # )
    buttons = []
    # index_list = [88, 87, 86, 85, 84, 80]
    skip_order = []
    # shifted_flux_array = fits.getdata(neid_filename, ext=1).copy()
    # shifted_var_array = fits.getdata(neid_filename, ext=4).copy()
    # print(shifted_flux_array, 'before_shift')
    # fig1, axs1 = plt.subplots()
    if sel_order is None:
        indices = range(neid_arrays)
    else:
        indices = [173 - sel_order]
        if 173-sel_order in skip_order:
            return None
    for index in indices:
        if ((173-index) < 70) or ((173-index) > 162) :# 69 165):
            continue
        if (173-index) in skip_order:
            continue
        # print("Working on order {} (index {})".format(173-index, index))
        neid_wl, neid_flux, neid_err = call_neid_data(index, neid_filename)
        # np.set_printoptions(threshold=np.inf)
        # print('neid err', neid_err)
        # print(neid_wl.size)
        # print(np.sum(np.isnan(neid_wl)), np.sum(np.isinf(neid_flux)), np.sum(np.isnan(neid_flux)))
        wlmask = np.isnan(neid_wl) | (neid_wl < 1000) | np.isinf(neid_flux) | np.isnan(neid_flux) | np.isnan(neid_err)
        if np.sum(wlmask) == np.size(neid_wl):
            print("This index is not useful")
        else:
            filtered_wl = neid_wl[~wlmask]
            filtered_flux = neid_flux[~wlmask]
            filtered_err = neid_err[~wlmask]

            telluric_mask = mask_creation(filtered_wl)
            
            # print("Ref velocity {}".format(ref_velocity))

            if abs(ref_velocity) != 0:
                # np.set_printoptions(threshold=np.inf)
                # print("Applying the doppler shift of {} km/s, order {}".format(ref_velocity, 173-index))
                # print('before', filtered_flux)
                # filetred_flux_orig = filtered_flux.copy()
                filtered_flux = shifting_korg_flux(filtered_wl, filtered_flux, ref_velocity)
                filtered_err = shifting_korg_flux(filtered_wl, filtered_err, ref_velocity)
                # print('after', 173-index, filtered_flux - filetred_flux_orig)
                # shifted_flux_array[index][~wlmask] = filtered_flux
                # shifted_var_array[index][~wlmask] = filtered_err**2
                
            # filtered_wl = filtered_wl[telluric_mask]
            # filtered_flux = filtered_flux[telluric_mask]
            # filtered_err[~telluric_mask] = 1000000000*filtered_err[~telluric_mask]

            
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
                scaled_flux, scaled_err = order_scaleing(filtered_flux, filtered_wl, filtered_err, axs, telluric_mask,
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
            neid_data["mask"][173-index] = telluric_mask
            # plt.figure()
            # plt.plot(filtered_wl, filtered_err, color='red')
            # plt.plot(filtered_wl, scaled_err, color='blue')
            # plt.show()
            # print('plotting axs1')
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
    # hdul = fits.open(neid_filename, mode='update')
    # print("Saving {} with doppler shift".format(neid_filename))
    # print(shifted_flux_array, "after")
    # hdul[1].data = shifted_flux_array
    # hdul[1].header.add_history("Replaced with shifted flux array")
    # hdul[4].data = shifted_var_array
    # hdul.flush()
    # hdul.close()
    # fig1.show()
    pdf.close()
    # html_str = mpld3.fig_to_html(fig)
    # with open(resultdict+"/NEID_spectra_comparison.html", 'w') as f:
    #     f.write(html_str)
    return neid_data


def inject_vel_neidata(neid_fname, inj_vel=0):
    flux = fits.getdata(neid_fname, ext=1)
    var = fits.getdata(neid_fname, ext=4)
    wl = fits.getdata(neid_fname, ext=7)
    # np.set_printoptions(threshold=np.inf)
    # print(np.diff(wl[173-127]))
    for index, flux_ord in enumerate(flux):
        if ((173-index) < 70) or ((173-index) > 162) :# 69 165):
            continue

        var_ord = var[index]
        wl_ord = wl[index]
        mask = ~np.isnan(wl_ord) & ~np.isnan(flux_ord) & ~np.isnan(var_ord) & (wl_ord > 3500) & ~np.isinf(flux_ord)
        # print(173-index, np.sum(~mask), np.size(wl_ord))
        # np.set_printoptions(threshold=np.inf)
        # print(wl_ord[mask])
        if np.sum(~mask) == np.size(wl_ord):
            continue
        shifted_flux = shifting_korg_flux(wl_ord[mask], flux_ord[mask], inj_vel)
        shifted_var = shifting_korg_flux(wl_ord[mask], var_ord[mask], inj_vel)
        # print(shifted_flux - flux_ord[mask])
        flux_ord[mask] = shifted_flux
        var_ord[mask] = shifted_var
        flux[index] = flux_ord
        var[index] = var_ord
    hdul = fits.open(neid_fname, mode='update')
    print("Saving {} with doppler shift".format(neid_fname))
    hdul[1].data = flux
    hdul[1].header.add_history("Replaced with shifted flux array, vel {} km/s".format(inj_vel))
    hdul[4].data = var
    hdul.flush()
    hdul.close()
    

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
    vel_profile = generate_profile(velocity_params, np.linspace(0, 1, 56), purpose='profile')
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





def order_scaling_residue(params, neid_flux, neid_wl, neid_err, korg_flux, mask, profile_fitting=True):
    # The function to get the continuum profile of the spectra.
    from model_functions import generate_profile

    shifted_wlarr = neid_wl - np.nanmin(neid_wl)
    mapped_wlarr = shifted_wlarr / np.nanmax(shifted_wlarr)
    # print(neid_wl)
    # print(mapped_wlarr)
    profile = generate_profile(params, mapped_wlarr, purpose='scaling')
    if profile_fitting:
        # print('profile_fitting')
        # print(params)
        scaled_flux = neid_flux * profile
        residue = (scaled_flux - korg_flux) / (neid_err)
        # print("Residue", residue, "chi2", np.sum(residue**2))
        # print(residue)
        # print(np.isnan(residue))
        return residue[mask]
    else:
        return profile
    



def order_scaleing(neid_flux, neid_wl, neid_err, axs, telluric_mask, korg_spectra=None, plotlyfig=None,verbose=True):
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
                               korg_flux=transformed_korg_flux, mask=telluric_mask)
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
    
    axs[2].plot(neid_wl[telluric_mask], residue, color='b', label='residue')
    # axs.add_trace(go.Scatter(x=neid_wl, y=residue, color='blue', name='residue'),
    grad_korg = np.gradient(transformed_korg_flux, axis=0) / np.gradient(neid_wl, axis=0)
    grad2_korg = np.gradient(grad_korg, axis=0) / np.gradient(neid_wl, axis=0)
    # axs[2].plot(neid_wl, grad_korg, color='g', alpha=0.6, label='d Korg')
    # axs[2].plot(neid_wl, grad2_korg, color='orange', alpha=0.4, label='d2 Korg')
    
    axs[2].legend()
    scaled_flux = neid_flux * profile
    scaled_err = neid_err * profile
    # fig2, axs2 = plt.subplots(2)
    # axs2[0].plot(neid_err)
    # axs2[1].plot(scaled_err)
    # plt.show()
    if plotlyfig:
        # plot_plotly(plotlyfig, neid_wl, transformed_korg_flux, 1, 1, 'black', "Korg")
        plot_plotly(plotlyfig, neid_wl, transformed_korg_flux, 1, 1, 'black', "Korg")
        plot_plotly(plotlyfig, neid_wl, scaled_flux, 1, 1, 'red', "NEID")
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
    #$ filename = 'data/Masked_regions_modified.csv'
    filename = 'data/Telluric_mask.csv'
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            masking_region.append((float(row[0]), float(row[1])))
    filename_bad = 'data/Bad_regions_mask.csv'
    with open(filename_bad, newline='') as csvfile:
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
    # mask = np.ones_like(wavelengths, dtype=bool)
    # print(wavelengths)

    return mask
    


def save_synt_data(params, neid_fname, opdir, save_interactive_plots=False):
    directory, filename = os.path.split(neid_fname)
    prefix = "Synt_"
    new_filename = prefix + filename
    dst = os.path.join(opdir, new_filename)
    shutil.copy(neid_fname, dst)
    # synt_data = os.path.join(opdir, filename)
    blaze_array = fits.getdata(neid_fname, ext=15)
    if not os.path.isfile(os.path.join(opdir, "Data_CCF.fits")):
        os.makedirs(os.path.join(opdir, "CCFs"))
        make_ccf(neid_fname, "Data_CCF.fits", opdir, os.path.join(opdir, "CCFs", filename))
        os.remove(os.path.join(opdir, filename))
    print("###############33Data using################33", os.path.join(opdir, "CCFs", filename), os.path.isfile(os.path.join(opdir, "CCFs", filename)))
    if os.path.isfile(os.path.join(opdir, "CCFs", filename)):
        os.remove(os.path.join(opdir, "CCFs", filename))
        print(os.path.join(opdir, "CCFs", filename), "removed")
        os.rmdir(os.path.join(opdir, "CCFs"))
    if os.path.isfile(os.path.join(opdir, "Synt_CCF.fits")):
        print("CCF file already exists")
    else:
        print("CCF file does not exist. Making now")
        print("Calling ", dst)
        neid_arrays = 121
        new_flux_array = fits.getdata(dst, ext=1).copy()
        original_flux_array = fits.getdata(dst, ext=1).copy() # duplicated for plotting
        neid_data = call_neiddata_full(dst, opdir, scaled_order=False)
        
        # if save_interactive_plots:
        #     plotlyfig = make_subplots(
        #         rows=2, cols=1, shared_xaxes=True,
        #         vertical_spacing=0.02
        #         )
        # else:
        #     plotlyfig = False
        from model_functions import residue_profile
        print("Cost will look so high because here we are not continuum normalising or scaling to match with Korg spectra")
        synt_spectra = residue_profile(neid_data,
                                       param_pos=np.array(['r', 'r', 'r', 'v']),
                                       scale_fact=-1,
                                       area_fact=0.5,
                                       contnorm=False,
                                       pca_comp=False,
                                       required='spectra')(params)
        if save_interactive_plots:
            pickle.dump(synt_spectra, open(os.path.join(opdir, "Synthetic_spectra.pkl"), "wb"))
            
            # print(synt_spectra)
            # from spectral_synthesis_functions import generate_with_korg
            
            # synt_spectra = generate_with_korg(velocity=False, cont_divide=False)
        synt_flux = synt_spectra['Total'] # ['Flux'] # ['Total']
        synt_flux = synt_flux / np.nanmedian(synt_flux)
        wl = synt_spectra['Wl']
        sort_mask = np.argsort(wl)
        synt_flux = synt_flux[sort_mask]
        wl = wl[sort_mask]
        spline = CubicSpline(wl, synt_flux, extrapolate=True)
        # print(neid_data)
        hdul = fits.open(dst, mode='update')
        header = hdul[0].header
        new_wl_data = hdul[7].data
        for index in range(neid_arrays):
            # print(index)

            neid_wl_order = new_wl_data[index] # fits.getdata(dst, ext=7)[index]
            strnum = str(173-index)
            while len(strnum) < 3:
                strnum = '0' + strnum

            zfact = header['SSBZ' + strnum]
            neid_wl_order = neid_wl_order.astype(np.float64) * (1+zfact)
            hdul[0].header['SSBZ' + strnum] = 0
            if np.nanmax(neid_wl_order) < 3600:
                continue
            elif np.nanmin(neid_wl_order) > 10000:
                continue
            synt_flux_order = spline(neid_wl_order)
            # print('orig', new_flux_array[index])
            # print('synt', synt_flux_order)
            new_flux_array[index] = synt_flux_order * blaze_array[index]
            # print("Flux replaced for index", index)
            new_wl_data[index] = neid_wl_order
            save_interactive_plots = False
            if save_interactive_plots:
                plotlyfig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.02
                    )
                plot_plotly(plotlyfig, neid_wl_order, synt_flux_order * blaze_array[index], 1, 1, 'black', "Korg")
                plot_plotly(plotlyfig, neid_wl_order, original_flux_array[index], 2, 1, 'red', "NEID")
                spectra_dir = os.path.join(opdir, "Spectra")
                if not os.path.exists(spectra_dir):
                    os.makedirs(spectra_dir)
                pyo.plot(plotlyfig, filename=spectra_dir+"/Fitted_spectra_comparison_order{}.html".format(173-index))
            # hdul.flush()
            # hdul.close()

        # hdul = fits.open(dst, mode="update")
        upext = 1
        hdul[upext].data = new_flux_array
        hdul[upext].header.add_history("Replaced flux array with synthetic flux")
        wlext = 7
        hdul[wlext].data = new_wl_data
        hdul[wlext].header.add_history("Replaced wavelength array with barycorrected wavelengths. SSBZ values kept 0")

        hdul.flush()
        hdul.close()
        print("Replaced the flux with synthetic flux")

        # fsr_mask = '../neidMaster_FSR_Mask20210218_v002.fits'
        # neid_calculate_mask_RV.main([dst,
        #                              opdir,
        #                              '/home/varghese/Desktop/CCF_package/PRVccf_Oct2025/PRVccf-master/PRVccf/config/neid_calculate_ccf.config',
        #                              '/home/varghese/Desktop/CCF_package/NEIDDRP_MasterFiles/neidMaster_StarDB_v000.config',
        #                              '--FSRMaskFile', fsr_mask])

        synt_data = os.path.join(opdir, "CCFs", filename)
        print("synt_data", os.path.isfile(synt_data))
        print("Synt_data", synt_data)
        make_ccf(dst, "Synt_CCF.fits", opdir, synt_data)
        # opfname = os.path.join(opdir, "Synt_CCF.fits")
        # ext_to_keep = 12
        # with fits.open(synt_data, mode='readonly') as hdul:
        #     # Create a new HDUList with only the desired extension
        #     hdu_to_keep = hdul[ext_to_keep]

        #     primary_hdu = fits.PrimaryHDU(hdu_to_keep.data,
        #                                   header=hdu_to_keep.header)  # empty primary header
        #     # new_hdul = fits.HDUList([primary_hdu])
        #     primary_hdu.writeto(opfname, overwrite=True)
        #     print(f"Saved {opfname} with only extension {ext_to_keep}")
        print("Deleting", synt_data)
        os.remove(synt_data)
        os.remove(dst)


def make_ccf(ipfname, opfname, opdir, generated_fname):
    print("making CCF for", ipfname)
    
    fsr_mask = '../neidMaster_FSR_Mask20210218_v002.fits'
    # os.makedirs(os.path.join(opdir, "CCFs"))
    neid_calculate_mask_RV.main([ipfname,
                                 os.path.join(opdir, "CCFs"),
                                 '/home/varghese/Desktop/CCF_package/PRVccf_Oct2025/PRVccf-master/PRVccf/config/neid_calculate_ccf.config',
                                 '/home/varghese/Desktop/CCF_package/NEIDDRP_MasterFiles/neidMaster_StarDB_v000.config',
                                 '--FSRMaskFile', fsr_mask])
    opfname = os.path.join(opdir, opfname)
    ext_to_keep = 12
    with fits.open(generated_fname, mode='readonly') as hdul:
        hdu_to_keep = hdul[ext_to_keep]
        primary_hdu = fits.PrimaryHDU(hdu_to_keep.data,
                                      header=hdu_to_keep.header)
        primary_hdu.writeto(opfname, overwrite=True)
        print(f"Saved {opfname} with only extension {ext_to_keep} inside the function")
        

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

def plotting_jacobian(result, korg_data):
    korg_wl = korg_data["Wl"]
    jacobian = result.jac
    residue = result.fun
    fig, axs = plt.subplots(5, sharex=True)
    axs[0].plot(korg_wl, residue[:-1])
    axs[0].set_ylabel("Residue")
    for i in range(4):
        axs[i+1].plot(korg_wl, jacobian[:, i][:-1], color='red')
        axs[i+1].set_ylabel(f"J{4-i}")
    plt.show()

    
def plotting_spectra(neid_data, korg_data, filename, interactive=False):

    orders_list = list(neid_data["Flux"].keys())
    pdf = PdfPages(filename)
    korg_raise = korg_data["Raise"]
    korg_fall = korg_data["Fall"]
    korg_total = korg_data["Total"]
    korg_residue = korg_data["Res"]
    residue_0 = korg_data["r0"]
    current_residue = korg_data["r"]
    data = korg_data['data']
    korg_wl = korg_data["Wl"]
    opdir = os.path.split(filename)[0]
    fig, axs = plt.subplots(2, sharex=True)
    axs[0].plot(korg_wl, data)
    axs[0].plot(korg_wl, korg_total)
    axs[0].set_ylim(-0.1, 2)
    axs[1].plot(korg_wl, korg_residue)
    # mask = np.abs(korg_residue) > 6
    # spread_mask = mask.copy()
    # for shift in range(1, 2):
    #     spread_mask[shift:] |= mask[:-shift]
    #     spread_mask[:-shift] |= mask[shift:]
    # axs[1].plot(korg_wl, korg_residue, 'ok')
    # axs[1].plot(korg_wl[spread_mask], korg_residue[spread_mask], 'ok')
    plt.show()
    # from make_telluricmask import get_mask_edges
    # start, ends = get_mask_edges(spread_mask, korg_wl)
    # stacked = np.vstack((start, ends)).T
    # np.savetxt("data/Bad_regions_mask.csv", stacked, delimiter=",", header="Start, End")
    # print(korg_raise.shape, korg_wl.shape)
    for order in orders_list:
        fig, axs = plt.subplots(2, figsize=(16, 8), sharex=True)
        neid_flux = neid_data["Flux"][order]
        neid_wls = neid_data["Wave"][order]
        neid_wl_mask = np.isin(korg_wl, neid_wls)
        # print(np.sum(neid_wl_mask), np.size(neid_wls))
        # print(order, neid_wls.shape)
        # plt.figure()
        # plt.plot(neid_wls)
        # plt.savefig("NEID_wls.png")
        # diff_neid_wls = np.diff(neid_wls)
        # diff_mask = diff_neid_wls < 0
        # print(np.sum(diff_mask))
        korg_raise_order = CubicSpline(korg_wl, korg_raise)(neid_wls) # korg_raise[neid_wl_mask]
        korg_total_order = CubicSpline(korg_wl, korg_total)(neid_wls) # korg_total[neid_wl_mask]
        korg_residue_order = CubicSpline(korg_wl, korg_residue)(neid_wls) # korg_residue[neid_wl_mask]
        residue_0_order = CubicSpline(korg_wl, residue_0)(neid_wls)
        current_residue_order = CubicSpline(korg_wl, current_residue)(neid_wls)
        # print(korg_raise_order.shape)
        plot_matplotlib(axs, neid_wls, neid_flux, 0, 0, color="black", label="NEID")
        # neid_wls_korg = neid_wls[neid_wl_mask]
        plot_matplotlib(axs, neid_wls, korg_raise_order, 0, 0, color="red", label="Korg Raise")
        plot_matplotlib(axs, neid_wls, korg_total_order, 0, 0, color="blue", label="Korg Total")
        plot_matplotlib(axs, neid_wls, korg_residue_order, 1, 0, color="blue", label="Residue")
        # print("Fitted_spectra", order, "saved")
        # print("interactive", interactive)
        if interactive:
            plotlyfig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                vertical_spacing=0.2
                )
            print("Saving interactive plot")
            plot_plotly(plotlyfig, neid_wls, neid_flux, 1, 1, 'black', "NEID")
            plot_plotly(plotlyfig, neid_wls, korg_raise_order, 1, 1, 'red', "NEID")
            if korg_fall is not None:
                korg_fall_order = korg_fall[neid_wl_mask]
                plot_plotly(plotlyfig, neid_wls, korg_fall_order, 1, 1, 'blue', "NEID")
            plot_plotly(plotlyfig, neid_wls, korg_total_order, 1, 1, 'green', "NEID")
            plot_plotly(plotlyfig, neid_wls, korg_residue_order, 2, 1, 'black', "Residue")
            plot_plotly(plotlyfig, neid_wls, residue_0_order, 3, 1, 'blue', "Residue_0")
            plot_plotly(plotlyfig, neid_wls, current_residue_order, 3, 1, 'green', "Current Residue")
            spectra_dir = os.path.join(opdir, "Spectra")
            if not os.path.exists(spectra_dir):
                os.makedirs(spectra_dir)
            html_path = spectra_dir+"/Fitted_spectra_order{}.html".format(order)
            pyo.plot(plotlyfig, filename=html_path)
            with open(html_path, "a") as f:
                f.write("""
                <script>
                document.addEventListener("DOMContentLoaded", function () {
                var plot = document.getElementsByClassName("js-plotly-plot")[0];
                
                plot.on('plotly_click', function(data){
                var xval = data.points[0].x;
                console.log("Clicked x:", xval);
                navigator.clipboard.writeText(xval);
                alert("Copied: " + xval);
                });
                });
                </script>
                """)
            
        if korg_fall is not None:
            korg_fall_order = CubicSpline(korg_wl, korg_fall)(neid_wls)  # korg_fall[neid_wl_mask]

            plot_matplotlib(axs, neid_wls, korg_fall_order, 0, 0, color="green", label="Korg Fall")
        

        axs[0].legend()
        plt.subplots_adjust(wspace=0, hspace=0)

        fig.suptitle("NEID spectra order {}".format(order), fontweight='bold')
        axs[1].set_xlabel("Wavelength $\AA$", fontsize=16, fontweight='bold')
        axs[0].set_ylabel("Flux", fontsize=16, fontweight='bold')
        axs[1].set_ylabel("Residue", fontsize=16, fontweight='bold')
        axs[1].tick_params(axis='both', which='major', labelsize=16)
        axs[0].tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    pdf.close()
    


def plot_lines(neid_data, korg_data, filename, mask_filename='data/sol_line_window.csv'):

    # Reading the csv file
    lines_list = calling_linelist(mask_filename)
    # print(len(lines_list.keys()))

    korg_raise = korg_data["Raise"]
    korg_fall = korg_data["Fall"]
    korg_total = korg_data["Total"]
    wl_array = korg_data['Wl']
    
    from model_functions import dict_to_array
    import matplotlib as mpl
    wl_array_neid = dict_to_array(neid_data['Wave'])
    wl_sort = np.argsort(wl_array_neid)
    wl_array_neid = wl_array_neid[wl_sort]
    mpl.rcParams['axes.formatter.useoffset'] = False
    mpl.rcParams['axes.formatter.limits'] = (-9, 9)   # disables scientific notation
    neid_flux = dict_to_array(neid_data["Flux"])
    neid_flux = neid_flux[wl_sort]
    neid_flux = CubicSpline(wl_array_neid, neid_flux)(wl_array)
    fig, axs = plt.subplots(3, 3, figsize=(25, 16))

    index = 0
    for cent, edges in lines_list.items():
        

        wl_mask = (wl_array > edges[0]) & (wl_array < edges[-1])

        wl_masked = wl_array[wl_mask] - cent
        vel_ar = 299792.458 * wl_masked / cent
        raise_masked = korg_raise[wl_mask]
        fall_masked = korg_fall[wl_mask]
        total_masked = korg_total[wl_mask]

        neid_masked = neid_flux[wl_mask]
        axs[index//3, index%3].annotate(
            fr"$\lambda_0 = {cent:.4f}\,\mathrm{{\AA}}$",
            xy=(0.3, 0.96),              # position inside the axes
            xycoords="axes fraction",
            fontsize=16,
            ha="left", va="top",
            fontweight='bold'
        )
        axs[index//3, index%3].plot(vel_ar, raise_masked, color='blue', label='Raising')
        axs[index//3, index%3].plot(vel_ar, fall_masked, color='red', label='Falling')
        axs[index//3, index%3].plot(vel_ar, total_masked, color='green', label='Resultant')
        axs[index//3, index%3].plot(vel_ar, neid_masked, color='black', label="NEID")
        axs[index//3, index%3].tick_params(axis='both', labelsize=22)
        # ax = axs[index//3, index%3]
        axs[index//3, index%3].ticklabel_format(style='plain')

        index += 1
    handles, labels = axs.ravel()[0].get_legend_handles_labels()
    fig.legend(handles,
               labels,
               loc='upper center',
               ncol=4,
               bbox_to_anchor=(0.5, 1.0),
               fontsize=22)
    # fig.subplots_adjust(top=)
    fig.text(0.5, 0.0, r'$\Delta v$ (km/s)', ha='center', fontsize=22, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename, bbox_inches='tight')

        
def plot_profile(params, params_errs, ax, profile='poly', inset=None, **kwargs):
    temp_array = np.load("data/Temp_layers.npy")
    # x = np.linspace(0, 1, np.shape(temp_array)[0])
    temp_mask = temp_array < 8000
    vel_array = np.ones(np.size(temp_array))
    temp_array = temp_array[temp_mask]
    x = (temp_array - np.min(temp_array)) / (np.max(temp_array) - np.min(temp_array))
    vel_profile = generate_profile(params, x, profile=profile)
    vel_array[temp_mask] = vel_profile
    vel_array[~temp_mask] = vel_profile[-1]
    vel_profile = vel_array
    # vel_var = generate_profile_err(params_errs, x)
    # vel_err = np.sqrt(vel_var)
    
    temp_array = np.load("data/Temp_layers.npy")
        
    # if inset is None:
    ax.plot(temp_array, vel_profile, **kwargs)
    #else:
        # print("Plotting inset")
        # x_mask = (temp_array >= inset[0]) & (temp_array <= inset[1]) # & (vel_profile >= inset[2]) & (vel_profile <= inset[3])
        # ax.plot(temp_array[x_mask], vel_profile[x_mask], **kwargs)

        
    # ax.fill_between(temp_array, vel_profile-vel_err, vel_profile+vel_err, **kwargs, alpha=0.3)

    
    
    
# Example usage
# my_dict = {'name': 'Alice', 'age': 25, 'city': 'Wonderland'}
# save_dict_to_pickle(my_dict, 'my_dictionary.pkl')

