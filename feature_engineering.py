import pandas as pd
import numpy as np
from scipy import stats
import neurokit2 as nk

def calculate_BVP_features(window):
    
    bvp = window['BVP'].values
    
    return {
        'BVP_mean': np.mean(bvp),
        'BVP_std': np.std(bvp),
        'BVP_min': np.min(bvp),
        'BVP_max': np.max(bvp),
        'BVP_range': np.max(bvp) - np.min(bvp),
        'BVP_median': np.median(bvp)
    }

def calculate_IBI_features(window):

    ibi_values = window['IBI'].values * 1000

    # Calculate standard HRV time-domain metrics ------------------------------
    # Standard deviation of NN intervals
    sdnn = np.std(ibi_values) 
    
    # Median of NN intervals
    median_nni = np.median(ibi_values)

    mean = np.mean(ibi_values)

    # Calculate standard HRB frqeuncey-domain metrics ---------------------------
    # Use neurokit2 for HRV frequency analysis
    #hrv_freq = nk.hrv_frequency(ibi_values, sampling_rate=None, show=False)
        
    # Extract key features
    #lf_power = hrv_freq['HRV_LF'].iloc[0]  # Low Frequency power
    #hf_power = hrv_freq['HRV_HF'].iloc[0]  # High Frequency power
    #lf_hf_ratio = hrv_freq['HRV_LFHF'].iloc[0]  # LF/HF ratio

    return {
        'SDNN': sdnn,
        'median_nni': median_nni,
        'IBI_mean': mean
        #'lf_power': lf_power,
        #'hf_power': hf_power,
        #'lf_hf_ratio': lf_hf_ratio
    }

def calculate_ACC_features(window: pd.DataFrame):

    acc_x = window['ACC_X'].values
    acc_y = window['ACC_Y'].values
    acc_z = window['ACC_Z'].values

    # Calculate magnitude of acceleration
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    # Calculate magnitude
    acc_magnitude_mean = np.mean(acc_mag)
    acc_magnitude_std = np.std(acc_mag)

    # Threshold = mean + 1 standard deviation
    threshold = acc_magnitude_mean + acc_magnitude_std 
    above_threshold = acc_mag > threshold 


    movement_density = np.mean(acc_mag > threshold) # Movement density (proportion of samples above threshold)
    transitions = np.diff(above_threshold.astype(int))
    burst_count = np.sum(transitions == 1) # A burst is defined as a sequence of consecutive samples above threshold

    return {
        'x_std': np.std(acc_x),
        'y_std': np.std(acc_y),
        'z_std': np.std(acc_z),
        'mag_mean': np.mean(acc_mag),
        'mag_std': np.std(acc_mag),
        'mag_max': np.max(acc_mag),
        'movement_density': movement_density,
        'burst_count': burst_count
    }

def calculate_EDA_features(window: pd.DataFrame):

    eda = window['EDA'].values
    
    # Basic statistics
    eda_mean = np.mean(eda)
    eda_std = np.std(eda)

    # Calculate trend (slope of linear regression)
    x = np.arange(len(eda))
    if len(eda) > 1:
        slope, _, _, _, _ = stats.linregress(x, eda)
        eda_trend = slope
    else:
        eda_trend = 0

    return {
        'EDA_mean': eda_mean,
        'EDA_std': eda_std,
        'EDA_trend': eda_trend
    }

def calculate_TEMP_features(window: pd.DataFrame):
    
    temp = window['TEMP'].values

    # Basic statistics
    temp_mean = np.mean(temp)
    temp_std = np.std(temp)
    
    # Calculate slope
    x = np.arange(len(temp))
    if len(temp) > 1:
        slope, _, _, _, _ = stats.linregress(x, temp)
        temp_slope = slope
    else:
        temp_slope = 0
    
    return {
        'TEMP_mean': temp_mean,
        'TEMP_std': temp_std,
        'TEMP_slope': temp_slope
    }
    
def calculate_HR_features(window: pd.DataFrame):

    hr = window['HR'].values

    # Basic statistics
    hr_mean = np.mean(hr)
    hr_std = np.std(hr)
    hr_range = np.max(hr) - np.min(hr)
    
    # Calculate trend
    x = np.arange(len(hr))
    if len(hr) > 1:
        slope, _, _, _, _ = stats.linregress(x, hr)
        hr_trend = slope
    else:
        hr_trend = 0
    
    return {
        'HR_mean': hr_mean,
        'HR_std': hr_std,
        'HR_range': hr_range,
        'HR_trend': hr_trend
    }

