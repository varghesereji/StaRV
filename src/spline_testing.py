import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from utils import make_ccf
from utils import call_neiddata_full
from astropy.io import fits
from scipy.interpolate import CubicSpline
from scipy.interpolate import make_interp_spline
from PRVccf.modules import neid_calculate_mask_RV
import pickle
# from utils import inject_vel_neidata

def shifting_korg_flux(wavelength, flux, velocity,
                       algo='CubicSpline'):
    c = 299792.458
    # print('Shifting for ', velocity)
    factor = (velocity/c + 1)
    shifted_wl = wavelength * factor # (velocity/c + 1)
    if algo == 'CubicSpline':
        shifted_flux = CubicSpline(shifted_wl, flux)(wavelength)
    else:
        shifted_flux_spline = make_interp_spline(shifted_wl, flux)
        shifted_flux = shifted_flux_spline(wavelength)
    return shifted_flux


def make_ccf(ipfname, opfname, opdir, generated_fname):
    print("making CCF for", ipfname)
    
    fsr_mask = '../neidMaster_FSR_Mask20210218_v002.fits'
    # os.makedirs(os.path.join(opdir, "CCFs"))
    neid_calculate_mask_RV.main([str(ipfname),
                                 str(opdir),
                                 '/home/varghese/Desktop/CCF_package/PRVccf_Oct2025/PRVccf-master/PRVccf/config/neid_calculate_ccf.config',
                                 '/home/varghese/Desktop/CCF_package/NEIDDRP_MasterFiles/neidMaster_StarDB_v000.config',
                                 '--FSRMaskFile', fsr_mask])
    # opfname = os.path.join(opdir, opfname)
    # ext_to_keep = 12


def inject_vel_neidata(neid_fname, inj_vel=0,
                       algo='CubicSpline'):
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
        shifted_flux = shifting_korg_flux(wl_ord[mask], flux_ord[mask], inj_vel, algo=algo)
        shifted_var = shifting_korg_flux(wl_ord[mask], var_ord[mask], inj_vel, algo=algo)
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



path = Path('/home/varghese/Desktop/Stellar_activity_mitigation/spline_fitting_test')


ref_fname = "neidL2_20220402T173047.fits"


data_path = Path('/data/varghese/NEID_data/pmode_avg_noblaze') / ref_fname

# header = ['inj_vel', 'CCFRVMOD', 'FWHMMOD', 'BISMOD']
methods = ["CubicSpline"]#, 3, 4, 5, 6]
colors = ['blue', 'green', 'red', 'orange', 'black']
fig, axs = plt.subplots(3, sharex=True)
for ind, algo in enumerate(methods):
    pklfile_name = path / "injected_vel_test{}.pkl".format(algo)
    if pklfile_name.exists():
        with open(pklfile_name, 'rb') as dat:
            info_dir = pickle.load(dat)
    else:
        info_dir = {}
    # vel = 0.1
    # vels_array = np.arange(-0.1, 0.1, 0.0001)
    # vels_array = np.array([0, 10, 20, 30, 40, 50])
    vels_array = np.arange(0, 10000000, 1000)
    vels_array = np.concatenate((-vels_array, vels_array))
    for vel_cm in vels_array:
        print("Injection", vel_cm)
        vel = vel_cm * 1e-5
        if vel_cm in list(info_dir.keys()):
            print("This one already done")
            rv = info_dir[vel_cm][0]
            fwhm = info_dir[vel_cm][1]
            bis = info_dir[vel_cm][2]
        else:
            shutil.copy(data_path, path)
            copied_path = path / ref_fname
            inject_vel_neidata(copied_path, vel, algo=algo)
            # _ = call_neiddata_full(copied_path, path,
            #                        refspec=None,
            #                        ref_velocity=vel)

            shifted_path = path / 'shifted_ccf'
            shifted_fname = shifted_path / ref_fname
            make_ccf(copied_path, "Data_CCF.fits", shifted_path, shifted_fname)
            header = fits.getheader(shifted_fname, ext=12)
            rv = header['CCFRVMOD']
            fwhm = header['FWHMMOD']
            bis = header['BISMOD']
            print("rv", rv)
            print("bis", bis)
            print("fwhm", fwhm)
            info_items = np.array([rv, fwhm, bis])
            info_dir[vel_cm] = info_items
            with open(pklfile_name, 'wb') as dat:
                pickle.dump(info_dir, dat)
            shifted_fname.unlink(missing_ok=True)
        axs[0].plot(vel_cm, rv, 'o', color=colors[ind])
        axs[1].plot(vel_cm, fwhm, 'o', color=colors[ind])
        axs[2].plot(vel_cm, bis, 'o', color=colors[ind])
        axs[2].set_xlabel("Injected vels (cm/s)")
        axs[2].set_ylabel("bis (km/s)")
        axs[1].set_ylabel("FWHM (km/s)")
        axs[0].set_ylabel("CCF RV (km/s)")
        plt.savefig(path / "Spline_test.pdf")
