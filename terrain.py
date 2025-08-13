import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#raise SystemExit("Exiting program due to an error.")
def lv03_to_lv95(E, N):
    return (E + 2_000_000, N + 1_000_000)

# load datasets
# spass requires netCDF python package
spass = xr.open_dataset("HSCLQMD_ch01h.swiss.lv95_WY_1962_2023.nc")
#print(spass["HSCLQMD"].attrs)
print("loaded SPASS dataset")
E_vals = spass["E"].values
N_vals = spass["N"].values

wind_dataset = pd.read_csv("ogd-smn_wfj_d_historical.csv", sep=";")
print("loaded wind measurements dataset")

field_dataset = pd.read_csv("snow_instability_field_data.csv", sep=";").dropna(how="all")
print("loaded snowpit observations dataset")
print()

# normalize dates to TimestampSeries
dates = field_dataset["Date_time"].astype(str).str.split(" ").str[0]
dates = pd.to_datetime(dates, dayfirst=True, format="mixed", errors="raise").dt.normalize()

# field_dataset has coordinates in LV03 system, to convert to LV95, +2 mil for x (easting) and +1 mil for y (northing)
E_LV95 = pd.to_numeric(field_dataset["X-Coordinate (m)"], errors="raise") + 2_000_000
N_LV95 = pd.to_numeric(field_dataset["Y-Coordinate (m)"], errors="raise") + 1_000_000
assert(len(dates) == len(E_LV95) == len(N_LV95))

myData = pd.DataFrame({
    "Date": dates,
    "E_LV95": E_LV95,
    "N_LV95": N_LV95
})

# snow depth is usually measured in cm, but in the SPASS dataset it is assumed to be meters,
# following CF metadata conventions, since no units are provided in the nc file.
# manual observation also indicates that the depth values are in meters, otherwise they would be unnaturally small
# and wouldn't make sense for snow data in the Swiss alps during midwinter
print("calculating snow depth values...")
depth_values = []
for date, east, north in zip(myData["Date"], myData["E_LV95"], myData["N_LV95"]):
    # Exact depth of grid tile
    # spass["HSCLQMD"].sel(time=date, E=east, N=north, method="nearest").item()
    # Linear interpolation style
    depth = spass["HSCLQMD"].sel(time=date).interp(E=east, N=north).item()
    depth_values.append(depth)

myData["Depth(m)"] = depth_values


# calculate variance
print("calculating depth variance values...")
# spass grid step measurement should be 1 km
dE = float(np.median(np.diff(E_vals)))
dN = float(np.median(np.diff(N_vals)))
assert(dE == dN == 1000)

var_values = []
for date, east, north in zip(myData["Date"], myData["E_LV95"], myData["N_LV95"]):
    # find 2D grid of snow depth values at time=date
    depth_map = spass["HSCLQMD"].sel(time=date)

    # take a 3x3 window centered at the coordinates of the pit site
    E_targets = xr.DataArray(east + np.array([-dE, 0.0, dE]), dims="Ew")
    N_targets = xr.DataArray(north + np.array([-dN, 0.0, dN]), dims="Nw")

    # compute variance of window ignoring NaNs
    window = depth_map.interp(E=E_targets, N=N_targets, method="linear")

    variance = float("nan")
    if int(window.notnull().sum()) >= 1:
        variance = float(window.var(dim=("Nw", "Ew"), skipna=True, ddof=0).item())

    var_values.append(variance)

myData["Variance(m^2)"] = var_values
myData["CV"] = np.sqrt(myData["Variance(m^2)"]) / myData["Depth(m)"]
#print(myData["CV"].tolist())

# MeteoSwiss open data
# this column is daily mean wind speed in m/s
assert "fkl010d0" in wind_dataset.columns
assert "reference_timestamp" in wind_dataset.columns
print("loading wind speed values...")

wind_dataset["reference_timestamp"] = pd.to_datetime(wind_dataset["reference_timestamp"], dayfirst=True, errors="raise").dt.normalize()
wind_speed_indexed_by_date = wind_dataset.set_index("reference_timestamp")["fkl010d0"]
wind_speed_indexed_by_date = pd.to_numeric(wind_speed_indexed_by_date, errors="raise")
myData["Wind_speed(m/s)"] = myData["Date"].map(wind_speed_indexed_by_date)


# add binarized scheme outputs to the dataframe
# return true (unstable) if score is 1, 2, or 3 (very poor, poor, fair)
def five_to_binary(score):
    if not (1 <= score <= 5):
        raise ValueError("Score out of bounds")
    return score < 4

# return true (unstable) if score is 1 or 2 (very poor, poor)
def four_to_binary(score):
    if not (1 <= score <= 4):
        raise ValueError("Score out of bounds")
    return score < 3

# return true (unstable) if score is 2 or 3 (number of criteria in critical range)
def three_to_binary(score):
    if not (1 <= score <= 3):
        raise ValueError("Score out of bounds")
    return score > 1

# load binarized stability scheme scores and ground truth into the dataframe
myData["5_class_bin"] = field_dataset["5-class_Stability"].map(five_to_binary)
myData["4_class_bin"] = field_dataset["4-class_Stability [Techel]"].map(four_to_binary)
myData["3_class_bin"] = field_dataset["3-class_Stability [sum S2008: 1+2+3]"].map(three_to_binary)
myData["Avalanche(GT)"] = (field_dataset["Avalanche_activity"] == 1.0)
#print(myData)



depth_threshold = 0.9   # meters
cv_threshold = 0.28     # unitless
wind_threshold = 8      # meters / sec
exceed = pd.DataFrame({
    "hs": myData["Depth(m)"] >= depth_threshold,
    "cv": myData["CV"] >= cv_threshold,
    "wind": myData["Wind_speed(m/s)"] >= wind_threshold
}).fillna(False)

myData["terrain_check"] = (exceed.sum(axis=1) >= 2)
myData["5_class_w_terrain"] = myData["5_class_bin"] | myData["terrain_check"]
myData["4_class_w_terrain"] = myData["4_class_bin"] | myData["terrain_check"]
myData["3_class_w_terrain"] = myData["3_class_bin"] | myData["terrain_check"]
print(myData)


schemes = ["5_class", "4_class", "3_class"]
results = {}
for scheme in schemes:
    old_col = f"{scheme}_bin"
    new_col = f"{scheme}_w_terrain"
    
    # calculate counts
    false_stables_old = int(((myData["Avalanche(GT)"] == True) & (myData[old_col] == False)).sum())
    false_stables_new = int(((myData["Avalanche(GT)"] == True) & (myData[new_col] == False)).sum())
    
    # reduction numbers
    reduction = false_stables_old - false_stables_new
    reduction_pct = (reduction / false_stables_old * 100) if false_stables_old > 0 else 0

    # calculate errors where terrain check accidentally flipped
    false_positives_old = int(((myData["Avalanche(GT)"] == False) & (myData[old_col] == True)).sum())
    false_positives_new = int(((myData["Avalanche(GT)"] == False) & (myData[new_col] == True)).sum())
    introduction = false_positives_new - false_positives_old
    introduction_pct = (introduction / false_positives_old * 100) if false_positives_old > 0 else 0
    #introduction = int(((myData["Avalanche(GT)"] == False) & (myData[old_col] == False) & (myData[new_col] == True)).sum())
    
    # store in results dict
    table = {
        "false_stables_old": false_stables_old,
        "false_stables_new": false_stables_new,
        "reduction": reduction,
        "reduction_pct": reduction_pct,
        "false_positives_old": false_positives_old,
        "false_positives_new": false_positives_new,
        "introduction": introduction,
        "introduction_pct": introduction_pct
    }
    results[scheme] = table
    print(table)


def make_results_chart(results):
    reduction = [results[scheme_name]["reduction_pct"] for scheme_name in results]
    introduction = [results[scheme_name]["introduction_pct"] for scheme_name in results]

    # positions and bar width
    x = np.arange(len(results))
    width = 0.35

    # plot
    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width/2, reduction, width, label="Reduction (%)", color="tab:blue")
    bars2 = ax.bar(x + width/2, introduction, width, label="Introduction (%)", color="tab:orange")

    # labels and title
    ax.set_ylabel("Percentage")
    ax.set_title("Terrain Check Impact on Stability Schemes")
    ax.set_xticks(x)
    ax.set_xticklabels(list(results.keys()))
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    # remove top and right border
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # annotate values above bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()


def main():
    for date, east, north, measurement in zip(myData["Date"], myData["E_LV95"], myData["N_LV95"], myData["Depth"]):
        map = spass["HSCLQMD"].sel(time=date)
        print(f"Snow depth on {date}: {measurement:.8f} meters")

        '''
        map.plot()
        plt.scatter([east], [north], color='r')
        plt.show()
        '''

if __name__ == "__main__":
    #main()
    make_results_chart(results)
    print("main finished")


