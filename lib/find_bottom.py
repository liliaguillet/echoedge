import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.ndimage import median_filter


# Svea
def get_beam_dead_zone(echodata): 
    """
    Get beam dead zone : size of echosounder near-zone (signal too strong at the top), number of bins to remove at the top of the echodata. 

    Parameters:
    - Echodata : the echograms  
    """
    echodata[np.isnan(echodata)] = 0
    row_sums = np.mean(echodata, axis=1).tolist()
    for i, row in enumerate(row_sums):
        if row == row:
            if row < (-50):
                return i


def detect_outliers(data, threshold=3):
    """
    Detect depth estimate outliers

    Parameters:
    - Data : depth estimate
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = [(x - mean) / std_dev for x in data]
    return np.abs(z_scores) > threshold


# Sailor
def moving_average(data, window_size):
    """
    Smooth the depth estimation on a window size

    Parameters:
    - Data : depth estimate
    - Window size : define in the initial parameters
    """
    
    series = pd.Series(data)
    moving_averages = series.rolling(window=window_size, center=True, min_periods=1).mean()
    return  moving_averages.tolist()

def interpolate_nan(lst, depth_if_all_nan):
    """
    Interpolate missing value of depth. 

    Parameters:
    - lst : depth estimate
    - depth_if_all_nan : in case all depth values are nan the default value is defined by this parameter
    """

    arr = np.array(lst)
    nan_indices = np.isnan(arr)
    non_nan_indices = np.arange(len(arr))[~nan_indices]

    if len(non_nan_indices) == 0:
        return [depth_if_all_nan] * len(arr)

    # Interpolate NaN values using linear interpolation
    arr[nan_indices] = np.interp(np.arange(len(arr))[nan_indices], non_nan_indices, arr[non_nan_indices])

    return arr.tolist()

def replace_outliers_with_nan(data):
    """
    Outliers are replaced with nan. First indentified outliers and then replace them by nan. 

    Parameters:
    -data :echodata
    """

    def detect_outliers(data, threshold=3):
        mean = np.mean(data)
        std_dev = np.std(data)
        z_scores = [(x - mean) / std_dev for x in data]
        return np.abs(z_scores) > threshold

    outliers_mask = detect_outliers(data)
    data = np.asarray(data, dtype=float)
    data[outliers_mask] = np.nan

    return data


def find_bottom(echodata, window_size, hardness_thresh):
    """
    Estimate the depth (bathymtry) from the echogram.

    Parameters:
    - echodata : calibrated acoustic data
    - window size : Windowsize to use when calculating moving averages of depth.
    - hardness_thresh : Threshold used to classify bottom. Signal needs to be stronger than threshold to be classified as bottom. If signal is weaker, depth > 100m
    """

    echodata_original = echodata.copy()
    #Get dead zone and slice it out from echodata
    dead_zone = get_beam_dead_zone(echodata) 
    echodata = echodata[dead_zone:, :] 

    #Finds the maxecho and depth
    depth = np.argmax(echodata, axis=0) 
    hardness = echodata[depth, np.arange(echodata.shape[1])]

    #Finding weak pings
    weak_ping_mask = np.isnan(np.where(hardness < hardness_thresh, np.nan, depth))
    #Findind outliers and set them to nan
    depth = replace_outliers_with_nan(np.where(hardness < hardness_thresh, echodata.shape[0], depth) )
    #Setting weak pings as nan as well
    depth[weak_ping_mask] = np.nan

    #calculating roughness on the values that aren't nan, if there aren't a bottom, depth roughness will be 0
    non_nan_depth = depth[~np.isnan(depth)]
    if len(non_nan_depth) == 0:
        depth_roughness = 0
    else:
        depth_roughness = np.round(np.median(np.abs(np.diff(non_nan_depth))), 2)

    #interpolating nan values and smoothing
    depth = interpolate_nan(depth, echodata.shape[0])
    depth = moving_average(depth , window_size)

    #Taking upper deadzone that was sliced to account and adding bottom deadzone to depth
    depth = [int(item + dead_zone) for item in depth]
    depth = [item - 30 if item != echodata_original.shape[0] else item for item in depth]

    #Removing the bottom
    for i in range(0, len(depth)):
        echodata_original[depth[i]:,(i)] = 0

    return depth, hardness, depth_roughness, echodata_original