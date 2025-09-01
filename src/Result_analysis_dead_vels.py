import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import os
import pickle
import sys

import configparser

from scipy.stats import zscore
from collections import defaultdict

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm, colors

from astropy.io import fits

from model_functions import generate_profile
from utils import plot_profile



def plotting_velocity_profile(params, axs, annotate=True, color=None, plotting="Separate"):
    raise_params = params[:-1]
    raise_errs = errs[:-1]
    scale_fact = -1
    add_term = params[-1]
    fall_params = raise_params * scale_fact
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
        plot_profile(raise_params, None, axs, color=color_raise, label="$V_{raise}$")
        plot_profile(fall_params, None, axs, color=color_fall, label="$V_{fall}$")
        axs.axhline(y=add_term, color='k')
    elif plotting=="Together":
        plot_profile(raise_params, None, axs[1], color=color_raise)
        plot_profile(fall_params, None, axs[0], color=color_fall)



configfile = 'Spectral_fitting.config'
config = configparser.ConfigParser()
config.read(configfile)

master_resdir = config['output_dir']['OP_MAIN']
master_subdir = config['output_dir']['OP_SUB']

resultsubdir_prefix = config['output_dir']['RESDIR_PREFIX']
wlwinds = config['inputs']['WL_WINT'].strip().split(', ')
maindir = config['data_dir']['NEID_DIR']

files_list = os.listdir(maindir)

script_path = os.path.abspath(__file__)
srcdir = os.path.dirname(os.path.dirname(script_path))

resmaindir = os.path.join(srcdir, master_resdir, master_subdir)
plot_fname = os.path.join(resmaindir, "Vel_profile.pdf")

pdf = PdfPages(plot_fname)
k = 1

header_kws = ['CCFJDMOD', 'CCFRVMOD', 'BISMOD', 'FWHMMOD']
full_params = []



ref_vels = np.arange(-0.1, 0.1, 0.01)

params_dict_vels = defaultdict(list)
for n, neid_filename in enumerate(files_list):
    # fig, axs = plt.subplots(1,2, figsize=(16, 8), sharex=True)
    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2,1])
    axs = fig.add_subplot(gs[0])
    gs_right = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[1], hspace=0)
    data_name = os.path.join(maindir, neid_filename)
    header = fits.getheader(data_name, ext=12)
    header_params = np.array([header[i] for i in header_kws])
    print("\n{}/{}: {}".format(k, len(files_list), neid_filename))
    s = 0
    ax_right = []
    for vels in ref_vels:
        vels = round(vels, 4)
        # print(vels, end="")
        basename = os.path.splitext(neid_filename)[0]
        if vels == 0:
            master_subdir_vel = master_subdir
        else:
            master_subdir_vel = master_subdir + "_{}".format(vels)
        
        resultdict = os.path.join(srcdir, master_resdir, master_subdir_vel, resultsubdir_prefix + basename + "_{}-{}".format(wlwinds[0], wlwinds[1]))
        # print(resultdict)

        result_filename = os.path.join(resultdict, "fitted_params.pkl")
        # print(result_filename)
        # if os.path.result_filename
        
        
        if not os.path.exists(result_filename):
            # print("Result is not generated")
            # plt.close()
            continue
        
        with open(result_filename, 'rb') as res:
            results = pickle.load(res)
        params = results['params']
        epoch_params = np.concatenate((header_params, params))
        full_params.append(epoch_params)
        params_dict_vels[vels].append(epoch_params)
        # ax_rights = []
        labels = ["$R_2$","$R_1$","$R_0$","C"]
        for i, param in enumerate(params):
            if len(ax_right) == 4:
                ax = ax_right[i]
            else:
                ax = fig.add_subplot(gs_right[i])
                ax_right.append(ax)
                ax.set_ylabel(labels[i])
                if i == 3:
                    ax.set_xlabel("Injected velocity (km/s)")
            # print(float(vels), float(param))
            ax.plot(float(vels), float(param), '.', color='blue')
        errs = results['params_err']
        s +=1
        plotting_velocity_profile(params, axs, annotate=False)
    k+=1
    if s == 0:
        plt.close()
        continue
    print("{}/{} vels done".format(s, np.size(ref_vels)))
    fig.suptitle(neid_filename, fontsize=16)
    axs.set_ylabel("Falling profile")
    # axs.legend(loc="upper center",
    #            bbox_to_anchor=(0.75, 0.5),
    #            ncol=1)
    axs.set_ylabel("Raising profile")
    axs.set_xlabel("Temperature (K)")
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
pdf.close()

print("Velocity profiles are saved at", plot_fname)
fig = plt.figure(figsize=(12, 12))
params_dict_kws = ["BJD", "CCFRV", "BIS", "FWHMMOD", "$R_2$", "$R_1$", "$R_0$", "C"]
n_figs = len(params_dict_kws)
gs = fig.add_gridspec(n_figs, n_figs, hspace=0, wspace=0)
axs = gs.subplots(sharex='col', sharey='row')

# full_params = np.array(full_params)

norm = colors.Normalize(vmin=np.min(ref_vels), vmax=np.max(ref_vels))
cmap = LinearSegmentedColormap.from_list("BlueRed", ["blue", "red"])
scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
for vels, full_params in params_dict_vels.items():
    print("Doing for {} (km/s)".format(vels))
    full_params = np.array(full_params)
    params_dict = {i:full_params.T[n] for n, i in enumerate(params_dict_kws)}
    
    # fig, axs = plt.subplots(n_figs, n_fits, figsize=(16,16)
    
    for i_x, kw_x in enumerate(params_dict_kws):
        print("\n")
        for i_y, kw_y in enumerate(params_dict_kws):
            
            ax = axs[i_y, i_x]
            if not i_x<i_y:
                ax.set_visible(False)
                continue
            print("{} vs {}|".format(kw_x, kw_y), end="")
            x_data = params_dict[kw_x]
            y_data = params_dict[kw_y]
            z_x = np.abs(zscore(x_data))
            z_y = np.abs(zscore(y_data))
            mask = (z_x <3) & (z_y < 3)
            color=scalar_map.to_rgba(vels)
            ax.plot(x_data, y_data, 'o', color=color)
            # ax.set(xlabel=kw_x, ylabel=kw_y)
            if i_x==0:
                ax.set_ylabel(kw_y)
            if i_y==n_figs-1:
                ax.set_xlabel(kw_x)
cax = fig.add_axes([0.90, 0.1, 0.02, 0.8])
cbar = fig.colorbar(scalar_map, ax=axs, cax=cax, orientation='vertical', shrink=0.9, pad=0.2)
cbar.set_label("Injected Velocity (km/s)", fontsize=16, labelpad=15)
# plt.tight_layout()
corr_plot_fname = os.path.join(resmaindir, "Correlations.pdf")
plt.savefig(corr_plot_fname)

'''
fig, axs = plt.subplots(2, figsize=(16, 8), sharex=True, gridspec_kw={'right':0.92})
rvmod_array = params_dict['CCFRV']
norm = colors.Normalize(vmin=np.min(rvmod_array), vmax=np.max(rvmod_array))
cmap = LinearSegmentedColormap.from_list("BlueRed", ["blue", "red"])
scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

fitted_params = full_params[:, 4:]


for n, params in enumerate(fitted_params):
    color=scalar_map.to_rgba(rvmod_array[n])
    plotting_velocity_profile(params, axs, annotate=False, color=color, plotting="Together")

axs[1].set_xlabel("Temperature (K)", fontsize=18)
axs[1].set_ylabel("Raising Velocity (km/s)", fontsize=18)
axs[0].set_ylabel("Falling Velocity (km/s)", fontsize=18)
# axs[1].tick_params(axis="both", labelsize=18)
cax = fig.add_axes([0.90, 0.1, 0.02, 0.8])
cbar = fig.colorbar(scalar_map, ax=axs, cax=cax, orientation='vertical', shrink=0.9, pad=0.2)
cbar.set_label("CCFRVMOD (km/s)", fontsize=16, labelpad=15)
# fig.suptitle("Difference from mean profile")
# axs[0].set_title("Falling velocities")
# plt.tight_layout()
plt.subplots_adjust(hspace=0)
plot_fname = os.path.join(resmaindir, "Vel_profile_together.pdf")
plt.savefig(plot_fname)

    
'''
