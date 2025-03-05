import feature_engineering

class SleepDataProcessor:
    """
    A class to process sleep data from E4 wearable devices and perform feature engineering
    on different time scales.
    """
    
    def __init__(self, data_dir: str, output_dir: str, sampling_freq: int = 64):
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

    def add_feature_function(self, name: str, function: Callable):
        """Register a feature calculation function"""
        self.feature_functions[name] = function

    def load_participant_data(self, file_path: str) -> pd.DataFrame:
        print(f"Loading data from {file_path}...")
        return pd.read_csv(file_path)

    def process_file(self, file_path: str, target_freq: float) -> pd.DataFrame:
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
        
        # Calculate window size in rows
        window_size = int(self.sampling_freq / target_freq)
        
        # Get participant ID from filename
        participant_id = os.path.basename(file_path).split('.')[0][:4]
        
        # Process data in windows
        return self._process_in_windows(df, window_size, participant_id)

    def _process_in_windows(self, df: pd.DataFrame, window_size: int, participant_id: str) -> pd.DataFrame:
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
            if 'Sleep_Stage' in window_df.columns:
                feature_row['sleep_stage'] = window_df['Sleep_Stage'].mode()[0]
            
            # Apply each feature function
            for feature_name, feature_func in self.feature_functions.items():
                try:
                    feature_value = feature_func(window_df)
                    
                    # If feature_value is a dictionary, add each key-value pair
                    if isinstance(feature_value, dict):
                        for sub_feature, value in feature_value.items():
                            feature_row[f"{feature_name}_{sub_feature}"] = value
                    else:
                        feature_row[feature_name] = feature_value
                except Exception as e:
                    print(f"Error calculating {feature_name} for window {i}: {e}")
                    feature_row[feature_name] = np.nan
            
            result_rows.append(feature_row)
        
        # Convert to DataFrame
        result_df = pd.DataFrame(result_rows)
        return result_df