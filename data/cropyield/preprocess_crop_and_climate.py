import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

np.random.seed(0)

climate_data_dir = Path("data/cropyield/")

# read the crop yield data
crop_filename = climate_data_dir / 'FAOSTAT_data_main_crop_yield.csv'
crop_data = pd.read_csv(crop_filename)
crop_data = crop_data[["Area Code", "Item", "Year", "Value"]]
crop_data = crop_data.rename({"Area": "country", "Item": "crop", "Year": "year", "Value": "yield"}, axis=1)

# read the weather data
weather_filename1 = climate_data_dir / '19.csv'
weather_filename2 = climate_data_dir / '20.csv'
weather_data1 = pd.read_csv(weather_filename1)
weather_data2 = pd.read_csv(weather_filename2)
weather_data = pd.concat([weather_data1, weather_data2]).drop("dewp_info", axis=1).rename({"timestamp": "year"}, axis=1)

# additional country and regional data used to associate lat/long with crop yield data
country_filename = climate_data_dir / 'countries_cords.csv'
country_data = pd.read_csv(country_filename)
country_data = country_data.replace({'"': '', ' ': ''}, regex=True)
country_data = country_data[["Country", "Latitude", "Longitude", "Alpha-2 code"]]

region_filename = climate_data_dir / 'regional_code.csv'
region_data = pd.read_csv(region_filename)
region_data = region_data[["Country Code", "ISO2 Code"]]

location_data = pd.merge(country_data, region_data, left_on="Alpha-2 code", right_on="ISO2 Code")

cropyield_data = pd.merge(crop_data, location_data, left_on="Area Code", right_on="Country Code")
cropyield_data = cropyield_data[["crop", "year", "Latitude", "Longitude", "yield"]].rename({"Latitude": "latitude", "Longitude": "longitude"}, axis=1)

cropyield_data = cropyield_data[cropyield_data["year"] >= 1990]
weather_data = weather_data[weather_data["year"] >= 1990].dropna()

cropyield_all = pd.DataFrame(columns=list(cropyield_data.columns) + ["temperature", "wind_speed", "precipitation"])

# inefficient way to match weather data to crop yield data
weather_cols = ["temperature", "wind_speed", "precipitation"]
for year in set(cropyield_data['year']):
    cy_year = cropyield_data[cropyield_data["year"] == year]
    we_year = weather_data[weather_data["year"] == year]

    nn = NearestNeighbors(n_neighbors=3, metric="haversine").fit(we_year[["latitude", "longitude"]].astype('float').values * (np.pi / 180))
    distances, indices = nn.kneighbors(cy_year[["latitude", "longitude"]].astype('float').values * (np.pi / 180))

    weather_inferred = []
    for i in range(len(cy_year)):
        neigh_i = we_year.iloc[indices[i]]
        dist_i = distances[i]
        ker_i = np.exp(- dist_i)
        ker_i = ker_i / ker_i.sum()

        weather_i = (neigh_i[weather_cols].values * ker_i).sum(axis=0)
        weather_inferred.append(weather_i)

    weather_inferred = pd.DataFrame(weather_inferred, columns=weather_cols)
    cy_year_new = pd.concat([cy_year.reset_index(drop=True), weather_inferred], axis=1)
    cropyield_all = pd.concat([cropyield_all, cy_year_new], axis=0)
    print(year)

cropyield_all.dropna().to_csv(climate_data_dir / "cropyield_processed.csv", header=True, index=False)




