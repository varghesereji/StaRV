from spectral_synthesis_functions import generate_with_korg
# from model_functions import generate_full_korgspectra
# from model_functions import generate_profile
# from model_functions import shifting_korg_flux

import time
import matplotlib.pyplot as plt
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

from utils import plot_plotly


window_size = [10, 50, 100, 200, 300, 500, 1000]

running_time = []

for window in window_size:
    wl_regions = np.arange(3700, 9200, window)
    t1 = time.time()
    for n, wl in enumerate(wl_regions[:-1]):
        # print(wl, wl_regions[n+1])
        generate_with_korg(velocity=False, wl_wind=(wl, wl_regions[n+1]))
    t2 = time.time()
    time_diff = t2-t1
    print("Window: {} timeL {}".format(window, time_diff))
    running_time.append(time_diff)

plt.figure()
plt.plot(window_size, running_time)
plt.xlabel("Wavelength window size")
plt.ylabel("Runnig time (s)")
plt.savefig("Running_time.pdf")


'''
t1 = time.time()

full_flux = np.array([])
full_wl = np.array([])
wl_regions = np.arange(3700, 9100, 100)
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

for n, wl in enumerate(wl_regions[:-1]):
    spectra = generate_with_korg(velocity=False, wl_wind=(wl-2, wl_regions[n+1]+2))

    flux = spectra['Flux'][10:-10]
    wl = spectra['Wl'][10:-10]
    plot_plotly(fig, wl, flux, 2, 1, None, 'Range {}'.format(wl))
    pyo.plot(fig, filename="Korg_flux.html")
    full_flux = np.concatenate((full_flux, flux))
    full_wl = np.concatenate((full_wl, wl))
print("Time taken:", time.time() - t1)
# plt.figure()
# plt.plot(full_wl, full_flux)
# plt.savefig("Korg_flux.pdf")


plot_plotly(fig, full_wl, full_flux, 1, 1, 'red', 'Korg')
pyo.plot(fig, filename="Korg_flux.html")
'''
'''
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

ref_vel = -0.1
vel_profile = generate_profile([ref_vel], np.linspace(0, 1, 56))

window = (4000, 6000)
injprof_spectra = generate_full_korgspectra(vel_profile, wlwindow=window)
inj_flux = injprof_spectra['Flux']
inj_wl = injprof_spectra['Wl']

import pickle
# with open('data/Ref_spectra_korg.pkl', 'rb') as ref:
#     zeroprof_spectra = pickle.load(ref)
zeroprof_spectra = generate_full_korgspectra(velocity=False, wlwindow=window)
zero_flux = zeroprof_spectra['Flux']
zero_wl = zeroprof_spectra['Wl']
shifted_flux = shifting_korg_flux(zero_wl, zero_flux, ref_vel)

plot_plotly(fig, inj_wl, inj_flux, 1, 1, 'red', "Injected")
plot_plotly(fig, inj_wl, shifted_flux, 1, 1, 'green', "DS")
plot_plotly(fig, inj_wl, shifted_flux-inj_flux, 2, 1, 'black', "DS")
pyo.plot(fig, filename="Compare_fluxes.html")


'''
