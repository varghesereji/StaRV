# coding: utf-8
# get_ipython().run_line_magic('run', 'model_functions.py')
from model_functions import reconstruct_params
import pickle
import numpy as np
import matplotlib.pyplot as plt
data = pickle.load(open("/home/varghese/Desktop/Stellar_activity_mitigation/35_injection_dC/Result_2ndorder_47layers_deltac_1000/Fitted_params_full_0.pkl", 'rb'))
data
data.keys()
data_dC = data['dC']
data_C = data['C']
P = np.array([data['P2'],
data['P1'],
data['P0']])

P = np.array([data['P2'],
data['P1'],
data['P0']])

reconstruct_params(P.T)
R_coefffs = reconstruct_params(P.T)
dC = data['dC']
c_vals = R_coefffs[:, -1]
c_vals
dC
plt.figure()
plt.plot(dC*1e5, c_vals*1e5, 'o')
plt.xlabel("dC")
plt.ylabel("c (cm/s)")
plt.xlabel("dC (cm/s)")
# plt.show()


data = pickle.load(open("/home/varghese/Desktop/Stellar_activity_mitigation/35_injection_dC/Result_2ndorder_47layers_deltac_1000_-0.0001/Fitted_params_full_-0.0001.pkl", 'rb'))
data
data.keys()
data_dC = data['dC']
data_C = data['C']
P = np.array([data['P2'],
data['P1'],
data['P0']])

P = np.array([data['P2'],
data['P1'],
data['P0']])

reconstruct_params(P.T)
R_coefffs = reconstruct_params(P.T)
dC = data['dC']
c_vals = R_coefffs[:, -1]
c_vals
dC
# plt.figure()
plt.plot(dC*1e5, c_vals*1e5, 'o', label="Injected -10cm/s")
# plt.xlabel("dC (km/s)")
# plt.ylabel("c (km/s)")
# plt.xlabel("dC (km/s)")
# plt.show()
plt.legend()
plt.savefig("/home/varghese/Desktop/Stellar_activity_mitigation/35_injection_dC/cdC_plot_both.pdf")
