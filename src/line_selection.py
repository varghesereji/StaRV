import numpy as np
import matplotlib.pyplot as plt

from spectral_synthesis_functions import generate_with_korg


korg_spectra = generate_with_korg(np.zeros(56), wl_wind=(3600, 9000))

flux = korg_spectra['Flux'] / korg_spectra['cont']
wl  = korg_spectra['Wl']

# line_mask = flux < 0.999
# mask1 = np.zeros(flux.shape[0])
# mask2 = np.zeros(flux.shape[0])

# loc = 13
# mask1[:-loc] = line_mask[loc:]
# mask2[loc:] = line_mask[:-loc]

# mask = (mask1 == 1) | (mask2 == 1)
# plt.figure()
# plt.plot(wl, flux, 'o-')
# plt.plot(wl[mask], flux[mask], 'o', alpha=0.5)

# plt.show()

# Trial 2, small difference in flux = continuum

cont_mask = (flux > 0.999) # & (np.abs(np.gradient(flux)) < 1e-10)
plt.figure()
plt.plot(wl, flux, 'o-')
plt.plot(wl[cont_mask], flux[cont_mask], 'o', alpha=0.5)

# plt.show()

cont_pos = np.zeros(np.shape(cont_mask)[0])
cont_pos[cont_mask] = 1

cont_grad = np.gradient(cont_pos)

# plt.figure()
# plt.plot(wl, cont_grad)
# plt.show()
cont_start = cont_grad == 0.5
cont_end = cont_grad == -0.5

print(wl[cont_start])
print(wl[cont_end])

for n, i in enumerate(wl[cont_start]):
    print(round(i, 4), round(wl[cont_end][n], 4))
