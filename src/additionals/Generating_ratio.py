import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from model_functions import generate_with_korg
from scipy.ndimage import median_filter
cntm_filename = "Scalefactordict_3800_9000.pkl"
if os.path.exists(cntm_filename):
    print("The continuums already exists.")
    with open(cntm_filename, 'rb') as korgfile:
        ratio_pkl = pickle.load(korgfile)
else:
    print("Pre-generated data does not exist. Generating now")

    # flux = generate_with_korg(np.zeros(56), (3800, 9000), continuum=False)
    cntm1_data = generate_with_korg(np.zeros(56), (3800, 9000), continuum=True)
    # cntm1_data = 
    cntm1 = np.array(cntm1_data['cont'])
    print(np.size(cntm1))
    cntm2_data = generate_with_korg(np.zeros(56), (3800, 9000), temp=5570, continuum=True)
    cntm2 = np.array(cntm2_data['cont'])
    print(type(cntm1))
    wl = np.array(cntm2_data['Wl'])
    ratio = cntm1/cntm2
    print(ratio)
    ratio_pkl = {'ratio':ratio, 'Wl': wl}
    with open(cntm_filename, 'wb') as korgfile:
        pickle.dump(ratio_pkl, korgfile)

ratio = ratio_pkl['ratio']
wl = ratio_pkl['Wl']
print('filtering')
t1 = time.time()
pts_gap = 500

filt_ratio = median_filter(ratio[::pts_gap], size=100, mode='reflect')# , size=100, mode='reflect')

x_fit = np.linspace(0, 1, np.size(filt_ratio))
coefficients = np.polyfit(x_fit, filt_ratio, 3)

polynomial = np.poly1d(coefficients)

x_new = np.linspace(0, 1, np.size(wl))
ratio_fit = polynomial(x_new)

t2 = time.time()
print("Time taken for filtering (s):", t2-t1)
print('Plotting')
plt.figure()
plt.plot(wl, ratio)
# plt.plot(wl[::1000], filt_ratio)
# plt.plot(wl[::pts_gap], ratio[::pts_gap], 'o-', color='k', alpha=0.3)
plt.plot(wl[::pts_gap], filt_ratio, '-', color='g', alpha=0.8)
plt.plot(wl, ratio_fit, '-', color='r', alpha=0.8)
plt.title('size = 6000, filtering in generation itself')
plt.savefig('Ratio_selected_points.pdf')
# ratio_pkl = {'ratio':ratio, 'Wl':wl}

wl_mask = (wl > 7049.9) & (wl < 7050.1)
ratio_scale = np.median(ratio_fit[wl_mask])
ratio_filename = "Ratio_3800_9000.pkl"
ratio_pkl = {'ratio':ratio_fit/ratio_scale, 'Wl': wl}
with open(ratio_filename, 'wb') as korgfile:
    pickle.dump(ratio_pkl, korgfile)
