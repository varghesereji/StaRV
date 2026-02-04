import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
import sys
from astropy.timeseries import LombScargle
import configparser

from scipy.stats import zscore
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm, colors

from astropy.io import fits

from model_functions import generate_profile
from utils import plot_profile



def plotting_velocity_profile(params, axs, annotate=True, color=None, plotting="Separate", inset=None):
    raise_params = params[:-1]

    # raise_params = np.insert(raise_params, 0, 0.66181421)
    raise_errs = errs[:-1]
    scale_fact = -1
    add_term = params[-1]
    fall_params = raise_params * scale_fact
    # print(raise_params, fall_params)
    if annotate:
        axs.annotate("vraise_coeffs($R_2$,$R_1$,$R_0$)\n="+str(raise_params), xy=(0.01, 0.05),
                     xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round, pad=0.3", fc="palegoldenrod", ec='blue', lw=2))
        axs.annotate("constant="+str(add_term), xy=(0.01, 0.95),
                     xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round, pad=0.3", fc="palegoldenrod", ec='blue', lw=2))
        axs.annotate("scale_fact="+str(scale_fact), xy=(0.95, 0.95),
                     xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round, pad=0.3", fc="palegoldenrod", ec='blue', lw=2))
    
    raise_params[-1] += add_term
    fall_params[-1] += add_term
    if color is None:
        color_raise = 'blue'
        color_fall = 'red'
    else:
        color_raise = color
        color_fall = color

    if plotting=="Separate":
        plot_profile(raise_params, None, axs, color=color_raise, label="$V_{raise}$", inset=inset)
        plot_profile(fall_params, None, axs, color=color_fall, label="$V_{fall}$", inset=inset)
        axs.axhline(y=add_term, color=color_raise)
        axs.tick_params(axis='both', labelsize=22)
        if inset is not None:
            axs.set_ylim(inset[2], inset[3])
            axs.set_xlim(inset[0], inset[1])
    
    elif plotting=="Together":
        print("making raise", raise_params)
        axs[0].tick_params(axis='both', labelsize=22)
        axs[1].tick_params(axis='both', labelsize=22)
        plot_profile(raise_params, None, axs[1], color=color_raise, alpha=0.3, inset=inset)
        print("making fall", fall_params)
        plot_profile(fall_params, None, axs[0], color=color_fall, alpha=0.3, inset=inset)


def making_profile_allvels(resultdict, axs, color=None, inset=None):
    print("resultdict", resultdict)
    print("inset", inset)
    result_filename = os.path.join(resultdict, "fitted_params.pkl")
    with open(result_filename, 'rb') as res:
        results = pickle.load(res)
    params = results['params']
    plotting_velocity_profile(params, axs, annotate=False, color=color, inset=inset)
    
    

args = sys.argv[1:]
# print(args)

configfile = 'Spectral_fitting.config'
config = configparser.ConfigParser()
config.read(configfile)

master_resdir = '31_higherorder_pmodeavg' # config['output_dir']['OP_MAIN']
master_subdir = config['output_dir']['OP_SUB'] # + '_-0.0002' # + # "_-0.0002"
if (len(args) == 1) & (args[0] != 'all'):
    print("adding suffix")
    print(args)
    if args[0] != '0':
        master_subdir += '_' + args[0]
# master_resdir = '27_pmode_avg' # config['output_dir']['OP_MAIN']
# master_subdir = 'Result_fix2nddegree_75thpercentile' # config['output_dir']['OP_SUB']

resultsubdir_prefix = config['output_dir']['RESDIR_PREFIX']

wlwinds = config['inputs']['WL_WINT'].strip().split(', ')
maindir = config['data_dir']['NEID_DIR']
# maindir = '/data/varghese/NEID_data/pmode_avg_data' # config['data_dir']['NEID_DIR']

files_list = os.listdir(maindir)

script_path = os.path.abspath(__file__)
srcdir = os.path.dirname(os.path.dirname(script_path))

resmaindir = os.path.join(srcdir, master_resdir, master_subdir)
if args[0] == 'all':
    plot_fname = os.path.join(resmaindir, "All_Vel_profile.pdf")
else:
    plot_fname = os.path.join(resmaindir, "Vel_profile.pdf")
plot_fname_ccf = os.path.join(resmaindir, "CCFs.pdf")

pdf = PdfPages(plot_fname)
pdf2 = PdfPages(plot_fname_ccf)
k = 1

header_kws = ['CCFJDMOD', 'CCFRVMOD', 'BISMOD', 'FWHMMOD']
full_params = []



ref_vels = np.arange(-0.1, 0.1, 0.01)

# avg_results = 'Result_dir_Comb_spectra_3700-9000'
# avg_resultdict = os.path.join(srcdir, master_resdir, master_subdir, resultsubdir_prefix + "Comb_spectra" + "_{}-{}".format(wlwinds[0], wlwinds[1]))
# avg_result_filename = os.path.join(avg_resultdict, "fitted_params.pkl")
# with open(avg_result_filename, 'rb') as avg_res:
#     avg_results = pickle.load(avg_res)
# print(avg_results)

vel_vals = np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20]) * 1e-5
vel_cm = vel_vals / 1e-5 # np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20])
norm = Normalize(vmin=vel_cm.min(), vmax=vel_cm.max())
cmap = plt.cm.coolwarm
color_list = cmap(norm(vel_cm))
for n, neid_filename in enumerate(files_list):
    fig, axs = plt.subplots(1, figsize=(16, 8), sharex=True)
    # data_name = os.path.join(maindir, neid_filename)
    # header = fits.getheader(data_name, ext=12)
    
    # header_params = np.array([header[i] for i in header_kws])
    
    basename = os.path.splitext(neid_filename)[0]

    resultdict = os.path.join(srcdir, master_resdir, master_subdir, resultsubdir_prefix + basename + "_{}-{}".format(wlwinds[0], wlwinds[1]))
    print("Resultdict", resultdict)
    data_name = os.path.join(resultdict, "Data_CCF.fits")
    result_filename = os.path.join(resultdict, "fitted_params.pkl")
    syntccf_filename = os.path.join(resultdict, "Synt_CCF.fits")
    if not os.path.exists(result_filename) or not os.path.exists(syntccf_filename):
        print("Result is not generated")
        plt.close()
        continue
    data_ccf = fits.getdata(data_name)[-1]
    synt_ccf = fits.getdata(syntccf_filename)[-1]
    # print(np.shape(data_ccf))
    rv_array = np.linspace(-200, 201, 804)
    header = fits.getheader(data_name, ext=0)
    header_params = np.array([header[i] for i in header_kws])
    if header_params[-1] > 7.10:
        print("Here is the problem", neid_filename)
    # print(resultdict)
    print("{}/{}: {}".format(k, len(files_list), neid_filename))

    # try:
    synt_header = fits.getheader(syntccf_filename)
    synt_ccfrvmod = synt_header['CCFRVMOD']
    try:
        synt_header_params = np.array([synt_header[i] for i in header_kws])[1:]
    except KeyError:
        print("KeyError exists")
        synt_header_params = np.array([np.nan for i in header_kws])[1:]
     # except FileNotFoundError:
       #  synt_ccfrvmod = np.nan
    header_params = np.concatenate((header_params, synt_header_params))# np.insert(header_params, 2, synt_ccfrvmod)
    # print(result_filename)
    # if os.path.result_filename


    k+=1
    with open(result_filename, 'rb') as res:
        results = pickle.load(res)

    params = results['params']
    # print("params", params)
    epoch_params = np.concatenate((header_params, params))
    # print("epoch", epoch_params)
    full_params.append(epoch_params)
    errs = results['params_err']
    print(params)
    if args[0] == 'all':
        axins = axs.inset_axes([0.25, 0.55, 0.4, 0.2])
        axins_2 = axs.inset_axes([0.25, 0.2, 0.4, 0.2])
        inset_regs = [7000, 9000, 1.84, 1.9]
        inset_regs_2 = [4000, 5000, 0.057, 0.058]
        for n, v in enumerate(vel_vals):
            if v == 0:
                master_subdir_suffix = master_subdir
            else:
                master_subdir_suffix = master_subdir + "_" + str(round(v, 5))
            resultdict = os.path.join(srcdir, master_resdir, master_subdir_suffix, resultsubdir_prefix + basename + "_{}-{}".format(wlwinds[0], wlwinds[1]))
            making_profile_allvels(resultdict, axs, color=color_list[n])
            making_profile_allvels(resultdict, axins, color=color_list[n], inset=inset_regs)
            making_profile_allvels(resultdict, axins_2, color=color_list[n], inset=inset_regs_2)
            axs.indicate_inset_zoom(axins)
            axs.indicate_inset_zoom(axins_2)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(sm, ax=axs,
                            fraction=0.045,
                            pad=0.04,
                            shrink=0.9)
        pos = cbar.ax.get_position()
        cbar.ax.set_position([pos.x0-0.1, pos.y0+0.05, pos.width, pos.height])
        cbar.set_label("Injected velocities (cm/s)")

    else:
        plotting_velocity_profile(params, axs)
        axs.legend(loc="upper center",
                   bbox_to_anchor=(0.75, 0.5),
                   ncol=1)

    fig.suptitle(neid_filename, fontsize=16)
    # axs.plot(rv_array, data_ccf, label="Data CCF")
    # axs.plot(rv_array, synt_ccf, label="Synt CCF")
    # axs.set_ylabel("CCF")
    # axs.set_xlabel("RV (km/s)")
    axs.set_ylabel("Falling profile", fontsize=18)
    axs.set_ylabel("Raising profile", fontsize=18)
    axs.set_ylabel("Velocity profile (km/s)", fontsize=18)
    axs.set_xlabel("Temperature (K)", fontsize=18)
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    fig1, axs1 = plt.subplots(1, figsize=(16, 8), sharex=True)
    axs1.plot(rv_array, data_ccf, label="Data CCF")
    axs1.plot(rv_array, synt_ccf, label="Synt CCF")
    axs1.set_xlim(-15, 15)
    # axs1.legend()
    axs1.set_ylabel("CCF")
    axs1.set_xlabel("RV (km/s)")
    axs1.legend(loc="upper center",
           bbox_to_anchor=(0.75, 0.5),
           ncol=1)
    pdf2.savefig()
    plt.close()


pdf.close()
pdf2.close()

print("Velocity profiles are saved at", plot_fname)
# avg_results_fullarray = np.concatenate((np.array([np.nan, np.nan, np.nan, np.nan]),
#                                         avg_results['params']))
full_params = np.array(full_params)
params_dict_kws = ["BJD", "CCFRVMOD", "BIS", "FWHMMOD", "Synt_CCFRVMOD", "Synt_BIS", "Synt_FWHMMOD", "$R_2$", "$R_1$", "$R_0$", "C"]
params_dict = {i:full_params.T[n] for n, i in enumerate(params_dict_kws)}
pkl_filename = os.path.join(resmaindir, "Fitted_params_full.pkl")
with open(pkl_filename, "wb") as resultdict_full:
    pickle.dump(params_dict, resultdict_full)
    
# print(params_dict)
n_figs = len(params_dict_kws)
# fig, axs = plt.subplots(n_figs, n_fits, figsize=(16,16)
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(n_figs, n_figs, hspace=0, wspace=0)
# axs = gs.subplots(sharex='col', sharey='row')

# Create axes one by one (no sharing at first)
axs = np.empty((n_figs, n_figs), dtype=object)
for i in range(n_figs):
    for j in range(n_figs):
        axs[i, j] = fig.add_subplot(gs[i, j])
        
for i_x, kw_x in enumerate(params_dict_kws):
    print("\n")
    for i_y, kw_y in enumerate(params_dict_kws):
        x_data = params_dict[kw_x]
        y_data = params_dict[kw_y]

        ax = axs[i_y, i_x]
        if i_x == i_y:
            zsc = np.abs(zscore(x_data))
            mask = zsc < 3
            ax.hist(x_data[mask], bins=100)
        elif (i_x < i_y):
            print("{} {} vs {} {}|".format(i_x, kw_x, i_y, kw_y), end="")
            z_x = np.abs(zscore(x_data))
            z_y = np.abs(zscore(y_data))
            mask = (z_x <3) & (z_y < 3)
            # print(kw_x, kw_y, x_data, y_data)
            ax.plot(x_data[mask], y_data[mask], '.')
            # ax.plot(x_data, y_data, '.')
            # ax.set(xlabel=kw_x, ylabel=kw_y)
        
        else:
            ax.set_visible(False)

        if i_x != 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel(kw_y, fontsize=14)

        if i_y != n_figs - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel(kw_x, fontsize=14)

        if i_x != 0:
            ax.tick_params(labelleft=False)
        if i_y != n_figs - 1:
            ax.tick_params(labelbottom=False)
        # if i_x==0:
        #     ax.set_ylabel(kw_y)
        # if i_y==n_figs-1:
        #     ax.set_xlabel(kw_x)
        # ax.axhline(y=avg_results_fullarray[i_y], color='k')
        # print("hor line", avg_results_fullarray[i_y])
        # ax.axvline(x=avg_results_fullarray[i_x], color='k')
        # print("ver line", avg_results_fullarray[i_x])
for ax in axs.ravel():
    ax.tick_params(axis='x', labelrotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
fig.align_xlabels()
fig.align_ylabels()
plt.tight_layout()
corr_plot_fname = os.path.join(resmaindir, "Correlations_withsyntccf.png")
plt.savefig(corr_plot_fname)

fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(n_figs-1, hspace=0)

time = params_dict["BJD"]
window = np.ones_like(time)
axs = [fig.add_subplot(gs[i, 0]) for i in range(n_figs-1)]

for n, dictkwys in enumerate(params_dict_kws[1:]):
    data = params_dict[dictkwys]
    frequency, power = LombScargle(time, data).autopower()
    frequency_wind, power_wind = LombScargle(time, window).autopower()
    power_wind = power_wind / np.nanmax(power_wind)
    axs[n].plot(1/frequency_wind, power_wind, label="Window")
    axs[n].plot(1/frequency, power, label=dictkwys)
    axs[n].set_ylabel(dictkwys)
    axs[n].legend()

plt.subplots_adjust(hspace=0, wspace=0)
periodogram_name = os.path.join(resmaindir, "Periodograms.pdf")
plt.tight_layout()
plt.savefig(periodogram_name)

fig, axs = plt.subplots(2, figsize=(16, 8), sharex=True, gridspec_kw={'right':0.92})
rvmod_array = params_dict['CCFRVMOD']
norm = colors.Normalize(vmin=np.min(rvmod_array), vmax=np.max(rvmod_array))
cmap = LinearSegmentedColormap.from_list("BlueRed", ["blue", "red"])
scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
# print(full_params)
fitted_params = full_params[:, 7:]
# print(fitted_params)

for n, params in enumerate(fitted_params):
    color=scalar_map.to_rgba(rvmod_array[n])
    plotting_velocity_profile(params, axs, annotate=False, color=color, plotting="Together")

axs[1].set_xlabel("Temperature (K)", fontsize=22)
axs[1].set_ylabel("Raising Velocity (km/s)", fontsize=22)
axs[0].set_ylabel("Falling Velocity (km/s)", fontsize=22)
# axs[1].tick_params(axis="both", labelsize=18)
cax = fig.add_axes([0.90, 0.1, 0.02, 0.8])
cbar = fig.colorbar(scalar_map, ax=axs, cax=cax, orientation='vertical', shrink=0.9, pad=0.2)
cbar.set_label("CCFRVMOD (km/s)", fontsize=16, labelpad=15)
# fig.suptitle("Difference from mean profile")
# axs[0].set_title("Falling velocities")
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plot_fname = os.path.join(resmaindir, "Vel_profile_together.pdf")
plt.savefig(plot_fname)

    

