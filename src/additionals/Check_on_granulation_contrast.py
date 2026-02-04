# coding: utf-8
get_ipython().run_line_magic('run', 'spectral_synthesis_functions.py')
generate_with_korg(velocity=False, wl_wind=(629,631))
generate_with_korg(velocity=False, wl_wind=(6290,6310))
stellar_param_dict = {'temp':5570}
generate_with_korg(velocity=False, wl_wind=(6290,6310), stellar_params=stellar_param_dict)
gra = generate_with_korg(velocity=False, wl_wind=(6290,6310))
int_gra = generate_with_korg(velocity=False, wl_wind=(6290,6310), stellar_params=stellar_param_dict)
gra_flux = gra['Flux']
int_gra_flux = gra]
int_gra_flux = gra['Flux']
a = 0.477
tot_flux = a*gra_flux + (1-a)*int_gra_flux
np.sqrt((1-a)*int_gra_flux**2 + a*gra_flux) / tot_flux**2
np.sqrt((1-a)*int_gra_flux**2 + a*gra_flux) / tot_flux
np.sqrt((1-a)*int_gra_flux**2 + a*gra_flux**2) / tot_flux
np.sqrt((1-a) * int_gra_flux**2 + a * gra_flux**2) / tot_flux
tot_flux
(1-a)*int_gra_flux**2+a*gra_flux**2
np.sqrt((1-a)*int_gra_flux**2+a*gra_flux**2)
np.sqrt(a*(1-a)*int_gra_flux**2+(1-a)*a*gra_flux**2)
np.sqrt(a*(1-a)*int_gra_flux**2+(1-a)*a*gra_flux**2) / tot_flux
np.sqrt(a*(gra_flux-tot_flux)**2+(1-a)*(int_gra_flux - tot_flux)**2)/tot_flux
gra_flux - tot_flux
tot_flux = a*gra_flux + (1-a)*int_gra_flux
gra_flux - tot_flux
gra_flux
int_gra_flux
gra_flux = 
gra_flux = gra['Flux']
gra_flux
int_gra_flux = int_gra['Flux']
int_gra_flux
tot_flux = a*gra_flux + (1-a)*int_gra_flux
np.sqrt(a*(gra_flux-tot_flux)**2+(1-a)*(int_gra_flux - tot_flux)**2)/tot_flux
