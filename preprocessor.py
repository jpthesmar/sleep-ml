from feature_engineering import calculate_ACC_features, calculate_BVP_features, calculate_EDA_features, calculate_HR_features, calculate_IBI_features, calculate_TEMP_features
import pandas as pd
import os
import glob
from tqdm import tqdm

class SleepDataProcessor:
    """
    A class to process sleep data from E4 wearable devices and perform feature engineering
    on different time scales.
    """
    
    def __init__(self, data_dir, output_dir, sampling_freq = 1920):
        """
        Initialize the processor with data directory and original sampling frequency.
        
        Args:
            data_dir: Directory containing CSV files with sleep data
            output_dir: Directory to save processed data
            sampling_freq: Original sampling frequency of the data (Hz)
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.sampling_freq = sampling_freq
        self.feature_functions = {}

    def add_feature_function(self, name, function):
        """Register a feature calculation function"""
        self.feature_functions[name] = function
    
    def load_participant_data(self, file_path):

        '''
        input: 
        A csv file from the DREAMT dataset
    
        return: a dataframes containing 4 sleep stages (W, N1, N2, R), no mising values and certain columns dropped with high null ratios
        '''

        df = pd.read_csv(file_path).drop(columns=['Obstructive_Apnea', 'Central_Apnea', 'Hypopnea', 'Multiple_Events'])

        df = df[df['Sleep_Stage'] != 'P']

        return df

    def process_file(self, file_path, window_size):
        """
        Process a single file and compute features at the target frequency.
        
        Args:
            file_path: Path to the CSV file
            target_freq: Target frequency in Hz (e.g., 1 for 1Hz, 1/30 for 30-second windows)
            
        Returns:
            DataFrame with computed features
        """
        # Load data
        df = self.load_participant_data(file_path)
        
        # Get participant ID from filename
        participant_id = os.path.basename(file_path).split('.')[0][:4]
        
        # Process data in windows
        return self._process_in_windows(df, window_size, participant_id)

    def _process_in_windows(self, df, window_size, participant_id):
        """
        Process data in windows and compute features.
        
        Args:
            df: DataFrame containing the raw data
            window_size: Size of the window in rows
            participant_id: ID of the participant
            
        Returns:
            DataFrame with computed features
        """
        # Prepare result dataframe
        result_rows = []
        
        # Calculate number of windows
        num_windows = len(df) // window_size
        
        # Process each window
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            
            # Skip if we don't have a full window
            if end_idx > len(df):
                break
                
            window_df = df.iloc[start_idx:end_idx]
            
            # Calculate timestamp for this window (use first timestamp in window)
            timestamp = window_df['TIMESTAMP'].iloc[0]
            
            # Start building the feature row
            feature_row = {
                'participant_id': participant_id,
                'window_start_time': timestamp,
                'window_end_time': window_df['TIMESTAMP'].iloc[-1]
            }
            
            # Add sleep stage (most common in this window)
            feature_row['sleep_stage'] = window_df['Sleep_Stage'].mode()[0]
            
            # Apply each feature function
            for feature_name, feature_func in self.feature_functions.items():
                try:

                    feature_values = feature_func(window_df)
                    for sub_feature, value in feature_values.items():
                        feature_row[f"{sub_feature}"] = value

                except Exception as e:
                    print(f"Error calculating {feature_name} for window {i}: {e}")
                    break
            
            result_rows.append(feature_row)
        
        # Convert to DataFrame
        result_df = pd.DataFrame(result_rows)
        return result_df
    
    def process_all_files(self, window_size, file_pattern  = "*.csv"):
        """
        Process all CSV files in the data directory at the target frequency.
        
        Args:
            target_freq: Target frequency in Hz
            file_pattern: Pattern to match files
        """
        
        # Get all CSV files
        file_paths = glob.glob(os.path.join(self.data_dir, file_pattern))
        
        # Process each file
        for file_path in tqdm(file_paths, desc=f"Processing files at {window_size/64} seconds"):
            participant_id = os.path.basename(file_path).split('.')[0][:4]
            output_path = os.path.join(self.output_dir, f"{participant_id}_processed.csv")
            
            # Skip if already processed
            if os.path.exists(output_path):
                print(f"Skipping {file_path}, already processed.")
                continue
                
            # Process file
            result_df = self.process_file(file_path, window_size)
            
            # Save result
            result_df.to_csv(output_path, index=False)
    

if __name__ == '__main__': 

    processor = SleepDataProcessor(
        data_dir='/Users/daviddechantsreiter/Desktop/WPI/Courses/Machine Learning/sleep-ml/physionet_data',
        output_dir='/Users/daviddechantsreiter/Desktop/WPI/Courses/Machine Learning/sleep-ml/downsampled_data',
        sampling_freq=64
    )

    # Register feature functions
    processor.add_feature_function("bvp", calculate_BVP_features)
    processor.add_feature_function("ibi", calculate_IBI_features)
    processor.add_feature_function("acc", calculate_ACC_features)
    processor.add_feature_function("eda", calculate_EDA_features)
    processor.add_feature_function("temp", calculate_TEMP_features)
    processor.add_feature_function("hr", calculate_HR_features)

    processor.process_all_files(1920)

    print('finished process')