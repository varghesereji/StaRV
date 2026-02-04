from spectral_synthesis_functions import generate_with_korg

import plotly.express as px
import plotly.graph_objects as go
import pickle

# Granular lane
gra = generate_with_korg(velocity=False)

# Intergranular lane
int_gra = generate_with_korg(velocity=False, stellar_params={'temp':5570})


# fig = px.line(x=x, y=y, labels={'x': 'X Axis', 'y': 'Y Axis'}, title='Line Plot from Arrays')

x = gra['Wl']
y1 = gra['Flux']
y2 = int_gra['Flux']

with open("additionals/Lanes_spectra.pkl", 'wb') as spec:
    pickle.dump({"Wl":x, "gra":y1, "intgra":y2}, spec)
# fig = go.Figure()

# fig.add_trace(go.Scatter(x=x, y=y1, mode='lines+markers', name='T=5770 K'))
# fig.add_trace(go.Scatter(x=x, y=y2, mode='lines+markers', name='T=5570 K'))
# fig.update_layout(title='Two Lines in One Plot',
#                   xaxis_title='Wavelength',
#                   yaxis_title='Flux')
# fig.write_html("spectrum_comparison.html")

