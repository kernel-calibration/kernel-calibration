import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pathlib import Path
from itertools import chain
from scipy import interpolate
from mpl_toolkits.basemap import maskoceans
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
import seaborn as sns
from sklearn.metrics.pairwise import haversine_distances


PLOT_VARIABLE = "temperature"


def haversine_degree(x, y):
    x = np.pi * x / 180
    y = np.pi * y / 180
    return haversine_distances(x, y)


np.random.seed(0)
cmap = sns.cubehelix_palette(as_cmap=True)

m = Basemap(projection='cyl', resolution=None,
            llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, )
m.shadedrelief(scale=0.2)

climate_data_dir = Path("data/H-Divergence/climate_exps/data/")
data = pd.read_csv(climate_data_dir / "cropyield_processed.csv")
# data = data[data.crop == "Barley"]
data = data[data.year >= 2000]

# create a grid for making predictions
lat_grid = np.linspace(-np.pi, np.pi, 500)
lon_grid = np.linspace(-np.pi, np.pi, 500)

lat_m, lon_m = np.meshgrid(lat_grid, lon_grid)
X_pred = np.stack([lat_m, lon_m]).reshape(2, -1)

value = data[PLOT_VARIABLE]

X = np.column_stack([data.latitude, data.longitude]) * (np.pi / 180)
y = value.values.reshape(-1, 1)

reg = KNeighborsRegressor(n_neighbors=3, weights="distance", metric="haversine")
reg.fit(X, y)
pred = reg.predict(X_pred.T).reshape(lat_m.shape).T

cutoff = 170
lat_m = (180 / np.pi * lat_m)[cutoff:]
lon_m = (180 / np.pi * lon_m)[cutoff:]
value_masked = maskoceans(lat_m, lon_m, pred[cutoff:])
m.contourf(lat_m, lon_m, value_masked, cmap=cmap)

plt.colorbar(label='value', shrink=0.5)
plt.show()

