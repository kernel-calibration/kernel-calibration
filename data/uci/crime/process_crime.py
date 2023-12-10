import pandas as pd
from pathlib import Path
import numpy as np

crime_data_dir = Path("data/uci/crime/")
county = pd.read_csv(crime_data_dir / "county_info.csv", dtype=str)
state = pd.read_csv(crime_data_dir / "state_info.csv", dtype=str)
fips = pd.read_csv(crime_data_dir / "state_fips.csv", dtype=str)
data = pd.read_csv(crime_data_dir / "communities.data", header=None, na_values=["?"])

state = pd.merge(state, fips, left_on="State", right_on="stname")
state["fips_code"] = state["st"] + "999"

county = county[["fips_code", "lng", "lat"]]
state = state.rename({"Latitude": "lat", "Longitude": "lng"}, axis=1)[["fips_code", "lat", "lng"]]
county = pd.concat([county, state]).astype({"lat": float, "lng": float})

data["fips_code"] = data[[0]].astype(str).squeeze().apply(lambda x: x.zfill(2)) + data[[1]].replace(np.nan, 999).astype(int).astype(str).squeeze().apply(lambda x: x.zfill(3))
data = pd.merge(county, data, on="fips_code").drop("fips_code", axis=1)
data = data.dropna(thresh=len(data) - 100, axis=1)  # Drop any columns that have more than 100 nan
data = data[data.columns[data.dtypes == float]]
data = data.dropna()

data.to_csv(crime_data_dir / "communities_processed.csv", index=False)






