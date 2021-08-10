'''
This programme create a "rain-map" of Berlin, in user-decided date range between
01-01-2000 and 31-12-2019, given a small and limited number of measuring points
(weather stations) by interpolating the rain between those.

This programme takes 3 files as input: 2 CSVs and one PNG
1. CSV containing information about all the weather stations, in our case
    all the stations in and around Berlin, including ID,name, longitude, latitude
2. CSV containing information about the relevant weather stations including ID,
    time stamp (YYYYmmDD) and rain per day (mm)
3. PNG map of berlin to be printed as background in the plots.

given 2 dates(start&end, can be edited in the global variables area)
this programme does the following process: (brief explanation)
1. create one, organised df and sums the rain
2. interpolate the data into a matrix using 5 different options
3. plot the interpolation

enjoy :)

to start - run main
'''

import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp2d, Rbf
import numpy as np
import os

# global variables
# choose start day and end date using date template: 'YYYY-MM-DD'
start_date = '2018-05-01'
end_date = '2018-08-28'
# preparing the subplots for later
fig, axs = plt.subplots(2, 2,  figsize=(15, 12))
cmap = 'seismic'

# map coordinates = (min_longi, max_longi, min_lati, max_lati)->(xmin xmax ymin ymax)
map_coor = (12.9216, 13.7714, 52.3177, 52.7180)
n = 500  # grid resolution


# daily rain data from max of 45 stations in berlin and it area
# files must be in same folder as this programme
path_to_map = os.path.join(os.getcwd(),"map_berlin.png")
rain_file_path = os.path.join(os.getcwd(),"data_RS_MN006.csv")
station_file_path = os.path.join(os.getcwd(),"sdo_RS_MN006.csv")
# translation dict from german to english
translate_dict = {'SDO_ID': 'station_id', 'Wert': 'rain', 'Zeitstempel': 'date',
                  'Qualitaet_Niveau': 'quality_level', 'Qualitaet_Byte': 'quality_byte',
                  'Produkt_Code':'product_code', 'SDO_Name': 'station_name',
                  'Geogr_Laenge':'longitude' , 'Geogr_Breite':'latitude'}


# opening function with further explanation
def input_please():
    print("This program is using rain data from stations in Berlin taken from DWD\n"
          "(the German Meteorological Service) at https://www.dwd.de/EN\n"
          "for further information, please refer to their website.\n")
    input("the program default date is may-july 2018.\n"
          "you can change the dates in the global variables\n"
          "press Enter continue")


''' 
the following functions are responsible for data organise and creating the
Data-frame that the next functions would use
'''
# cleaning the original rain-data csv from junk, translating german to english, sorting,
# moving from timestamp to pd.datetime and setting the date as the new index
def clean_df_rain_original(rain_file_path):
    df_rain = pd.read_csv(rain_file_path).rename(columns=translate_dict, inplace=False).\
        drop('quality_byte', axis=1).drop('quality_level', axis=1).drop('product_code', axis=1)
    df_rain['date'] = pd.to_datetime(df_rain['date'], format='%Y%m%d', errors='ignore').sort_values()
    df_rain = df_rain.set_index('date')
    return df_rain

# cleaning the original station-data csv from junk and translating german to english
def clean_df_stations_original(station_file_path):
    df_stations = pd.DataFrame(pd.read_csv(station_file_path)).\
        rename(columns=translate_dict, inplace=False).\
        drop('Hoehe_ueber_NN', axis=1).drop('Metadata_Link', axis=1).\
        drop('station_name', axis=1)
    return df_stations

# cut rain_df by date range in global variables
def cut_df_by_dates(df):
    return df.loc[start_date:end_date].sort_values(by=['date', 'station_id'])

# sum rain_df rain amounts through out the entire given df (after cutting by dates)
def sum_rain_df(df_station_rain):
    return_df = df_station_rain.groupby('station_id')['rain'].sum().reset_index()
    return return_df.round(3)

# combining both DFs to one df with: id, longitude, latitude and summed rain
def combine_df_id_lng_lat(cut_df_rain, df_stations):
    result = cut_df_rain.merge(df_stations, on=['station_id'])
    return result

'''
the following functions use the final DF from combine_df_id_lng_lat() to interpolate
the measure points into a full matrix, the matrix is ONLY between the the furthest stations
to avoid interpolations going to +- inf on the edges
'''


# this function is being used by the IDW to calculate how much each point
# will effect it surroundings
def distance_matrix(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T
    d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
    d1 = np.subtract.outer(obs[:, 1], interp[:, 1])
    return np.hypot(d0, d1)


# different interpreters, to more information please refer to added .docx
# Inverse distance weighting,  it resorts to the inverse of the distance
# to each known point ("amount of proximity") when assigning weights.
def simple_idw(df, xi, yi):
    interp_type = "Homemade IDW"
    xi, yi = np.meshgrid(xi, yi)
    xi, yi = xi.flatten(), yi.flatten()
    dist = distance_matrix(df['longitude'], df['latitude'], xi, yi)
    # In IDW, weights are 1 / distance
    weights = 1.0 / dist
    # Make weights sum to one
    weights /= weights.sum(axis=0)
    # Multiply the weights for each interpolated point by all observed Z-values
    z = np.dot(weights.T, df['rain'].to_numpy()).reshape((n, n))
    return z, interp_type


# classic interp2d by Scipy, if number of stations is less that 16, this function would
# interpolate in a linear, if number of stations equal or greater than 16, interpolate "cubic"
def interp2d_func(df, xi, yi):
    if len(df['rain'].to_numpy()) < 16:
        tck = scipy.interpolate.bisplrep(df['longitude'], df['latitude'], df['rain'], kx=2, ky=2)
        z = scipy.interpolate.bisplev(xi, yi, tck)
        interp_type = "Linear Bisplrep"
    else:
        f = interp2d(df['longitude'], df['latitude'], df['rain'], kind='cubic')
        interp_type = "Cubic Interp2d"
        z = np.array(f(xi, yi))
    z[z < 0] = 0 # making sure no point in the grid is lower than 0 mm (because, you know, logic)
    return z, interp_type


# Radial Basis Function is a Scipy interpolation The radial basis function, based on the radius,
# given by the norm, by default is ‘multiquadric’:
def rbf_func(df, xi, yi):
    interp_type = "Rbf Function"
    # 2-d tests - setup scattered data
    xi, yi = np.meshgrid(xi, yi)
    rbf = Rbf(df['longitude'], df['latitude'], df['rain'])
    z = rbf(xi, yi)
    return z, interp_type


# Smooth bivariate spline approximation by Scipy, does as it says (spline, but smooth)
def smoothBivariateSpline(df, xi, yi):
    if len(df['rain'].to_numpy()) > 15:
        f = scipy.interpolate.SmoothBivariateSpline(df['longitude'],
                                                    df['latitude'], df['rain'])
    else:
        f = scipy.interpolate.SmoothBivariateSpline(df['longitude'],
                                            df['latitude'], df['rain'], kx=2, ky=2)
    z = np.array(f(xi, yi))
    interp_type = "Smooth Bivariate Spline"
    return z, interp_type


# this function get the interpolated matrix and normalize it (0-1) in order
# to limit the power of interpolators that might go to unrealistic values
def normalize_grid(grid):
    grid_max = np.max(grid)
    new_grid = grid/grid_max
    return new_grid


'''
the following functions are responsible for all the plots of the programme
starting with the map.png and marking the stations and rain amounts on the subplots
'''


# plot map.png on all the subplots
def plot_map_from_image():
    map_png = plt.imread(path_to_map)
    for ax in axs.flat:
        ax.set(xlabel='longitude', ylabel='latitude')
        ax.imshow(map_png, extent = map_coor)
        ax.grid()


# scatter all the active stations (Black 'x') on subplots
def plot_active_station_map(merged_rain_lng_lat):
    for ax in axs.flat:
        for i,row in merged_rain_lng_lat.iterrows():
            ax.scatter(row['longitude'], row['latitude'], marker='x', c='k')


# write rain amounts next to scatter (only if summed amount is under 999mm to avoid
# too much numbers on the subplots
def annotate_stations_rain(merged_rain_lng_lat):
    for ax in axs.flat:
        for i,row in merged_rain_lng_lat.iterrows():
            if np.max(row['rain']) > 999:
                return
            ax.annotate(row['rain'], (row['longitude']+0.0095, row['latitude']-0.0095),c='k')


# plotting the matrix-grid to [0, 0]
def plot_grid1(grid, interp_type, plot_coords):
    title = "Interpolate: "+interp_type
    axs[0, 0].imshow(grid, cmap=cmap, alpha = 0.5, extent=plot_coords,origin='lower')
    cset = axs[0, 0].contour(grid, np.linspace(0, np.max(grid), num=5), linewidths=1,
                      extent=plot_coords, colors='k', linestyles='-')
    axs[0, 0].clabel(cset, inline=True, fmt='%1.1f', fontsize=8)
    axs[0, 0].set_title(title)


# plotting the matrix-grid to [0, 1]
def plot_grid2(grid, interp_type, plot_coords):
    title = "Interpolate: " + interp_type
    axs[0, 1].imshow(grid, alpha = 0.5, extent=plot_coords, cmap=cmap, origin='lower')
    cset = axs[0, 1].contour(grid, np.linspace(0, np.max(grid), num=5), linewidths=1,
                             extent=plot_coords, colors='k', linestyles='-')
    axs[0, 1].clabel(cset, inline=True, fmt='%1.1f', fontsize=8)
    axs[0, 1].set_title(title)


# plotting the matrix-grid to [1, 0]
def plot_grid3(grid,interp_type, plot_coords):
    title = "Interpolate: " + interp_type
    axs[1, 0].imshow(grid, cmap=cmap, alpha = 0.5, extent=plot_coords,origin='lower')
    cset = axs[1, 0].contour(grid, np.linspace(0, np.max(grid), num=5), linewidths=1,
                     extent=plot_coords, colors='k', linestyles='-')
    axs[1, 0].clabel(cset, inline=True, fmt='%1.1f', fontsize=8)
    axs[1, 0].set_title(title)


# plotting the avg of all 3 grids to [1, 1]
def plot_avg(grid, plot_coords):
    title = "statistical mean for all"
    im = axs[1, 1].imshow(grid, cmap=cmap, alpha = 0.5, extent=plot_coords,origin='lower', vmin=0, vmax=1)
    cset = axs[1, 1].contour(grid, np.linspace(0, np.max(grid), num= 5), linewidths=1,
                     extent=plot_coords, colors='k', linestyles='-')
    axs[1, 1].clabel(cset, inline=True, fmt='%1.1f', fontsize=8)
    axs[1, 1].set_title(title)
    fig.colorbar(im, ax=axs.ravel().tolist())


def main():

# opening print to start with a smile
    input_please()

# working together to create df for all the other functions "merged_rain_lng_lat"
    cleaned_df_rain = clean_df_rain_original(rain_file_path)
    cleaned_df_stations = clean_df_stations_original(station_file_path)
    rain_df_date_range = cut_df_by_dates(cleaned_df_rain)
    summed_rain = sum_rain_df(rain_df_date_range)
    merged_rain_lng_lat = combine_df_id_lng_lat(summed_rain, cleaned_df_stations)

# interpolations and plotting boundaries to stations square edges
    x = merged_rain_lng_lat['longitude']
    y = merged_rain_lng_lat['latitude']
    xi = np.linspace(np.min(x), np.max(x), n)
    yi = np.linspace(np.min(y), np.max(y), n)
    plot_coords = (np.min(x), np.max(x), np.min(y), np.max(y))

# to prevent interpolations going to +-inf, grids must be normalized so 0 is no rain and 1 is max
# Calculate home IDW
    grid1, interp_type_grid1 = simple_idw(merged_rain_lng_lat, xi, yi)
    grid1 = normalize_grid(grid1)

# calculate interp2d linear(less than 16 stations) / cubic(more -''-)
    grid2, interp_type_grid2 = interp2d_func(merged_rain_lng_lat, xi, yi)
    grid2 = normalize_grid(grid2)

# interp smooth spline or Rbf - depending if grid2 was linear or cubic
    if interp_type_grid2 == "Linear Bisplrep":
        grid3, interp_type_grid3 = rbf_func(merged_rain_lng_lat, xi, yi)
        grid3 = normalize_grid(grid3)
    else:
        grid3 , interp_type_grid3 = smoothBivariateSpline(merged_rain_lng_lat, xi, yi)
        grid3 = normalize_grid(grid3)


# plot the map and scatter the stations and rain amounts
    plot_map_from_image()
    plot_active_station_map(merged_rain_lng_lat)
    annotate_stations_rain(merged_rain_lng_lat)

# statistical grids mean
    grids_avg = (grid1 + grid2 + grid3) / 3


# plot all interpolations
    plot_grid1(grid1, interp_type_grid1, plot_coords)
    plot_grid2(grid2, interp_type_grid2, plot_coords)
    plot_grid3(grid3, interp_type_grid3, plot_coords)
    plot_avg(grids_avg, plot_coords)
    plt.show()


if __name__== "__main__":
  main()
