"""
Time synchronization and data alignment for multi-rate sensor data.

Handles synchronization of:
- Concentration data: ~1 Hz
- Pressure data: 1000 Hz
- MFR data: 1000 Hz
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings


class SensorSynchronizer:
    """Synchronize multi-rate sensor data to a common timeline."""
    
    def __init__(self, target_rate_hz: float = 1.0):
        """
        Initialize synchronizer.
        
        Args:
            target_rate_hz: Target sampling rate in Hz (default: 1 Hz to match concentration)
        """
        self.target_rate_hz = target_rate_hz
        self.target_period_s = 1.0 / target_rate_hz
        
    def synchronize_all(
        self,
        concentration_df: pd.DataFrame,
        pressure_df: pd.DataFrame,
        mfr_data: Dict[str, pd.DataFrame],
        pressure_rate_hz: float = 1000.0,
        mfr_rate_hz: float = 1000.0
    ) -> pd.DataFrame:
        """
        Synchronize all sensor data to common timeline.
        
        Args:
            concentration_df: Concentration data (~1 Hz)
            pressure_df: Pressure data (high frequency)
            mfr_data: Dictionary with MFR channels
            pressure_rate_hz: Pressure sampling rate
            mfr_rate_hz: MFR sampling rate
            
        Returns:
            Synchronized DataFrame with all sensor data
        """
        # Step 1: Downsample pressure data
        print("Downsampling pressure data from 1000 Hz to 1 Hz...")
        pressure_downsampled = self.downsample_high_freq(
            pressure_df,
            source_rate_hz=pressure_rate_hz,
            columns_to_process=['P1', 'Trigger']
        )
        
        # Step 2: Downsample MFR data
        print("Downsampling MFR data from 1000 Hz to 1 Hz...")
        mfr_downsampled = {}
        for channel, df in mfr_data.items():
            if f'{channel}_mfr_gs' in df.columns:
                # Use converted values for CH2
                mfr_downsampled[f'{channel}_mfr'] = self.downsample_high_freq(
                    df,
                    source_rate_hz=mfr_rate_hz,
                    columns_to_process=[f'{channel}_mfr_gs']
                )
            elif f'{channel}_pressure_bar' in df.columns:
                # Use converted values for CH3
                mfr_downsampled[f'{channel}_pressure'] = self.downsample_high_freq(
                    df,
                    source_rate_hz=mfr_rate_hz,
                    columns_to_process=[f'{channel}_pressure_bar']
                )
            else:
                # Use raw values
                mfr_downsampled[channel] = self.downsample_high_freq(
                    df,
                    source_rate_hz=mfr_rate_hz,
                    columns_to_process=[channel]
                )
        
        # Step 3: Align all data by length
        print("Aligning data by length...")
        min_length = min(
            len(concentration_df),
            len(pressure_downsampled),
            *[len(df) for df in mfr_downsampled.values()]
        )
        
        # Create synchronized dataframe
        synced_df = pd.DataFrame()
        
        # Add index (sample number)
        synced_df['sample'] = np.arange(min_length)
        
        # Add concentration data
        if 'Output (%)' in concentration_df.columns:
            synced_df['h2_concentration'] = concentration_df['Output (%)'].iloc[:min_length].values
        
        if 'Alarm' in concentration_df.columns:
            synced_df['alarm'] = concentration_df['Alarm'].iloc[:min_length].values
        
        if 'Pt100 (°C)' in concentration_df.columns:
            synced_df['temperature'] = concentration_df['Pt100 (°C)'].iloc[:min_length].values
        
        if 'RH (%)' in concentration_df.columns:
            synced_df['humidity'] = concentration_df['RH (%)'].iloc[:min_length].values
        
        # Add pressure data (aggregated features)
        if 'P1_mean' in pressure_downsampled.columns:
            for col in pressure_downsampled.columns:
                synced_df[f'pressure_{col}'] = pressure_downsampled[col].iloc[:min_length].values
        
        # Add MFR data (aggregated features)
        for channel_name, channel_df in mfr_downsampled.items():
            for col in channel_df.columns:
                synced_df[f'{channel_name}_{col}'] = channel_df[col].iloc[:min_length].values
        
        print(f"Synchronized data shape: {synced_df.shape}")
        print(f"Columns: {synced_df.columns.tolist()}")
        
        return synced_df
    
    def downsample_high_freq(
        self,
        df: pd.DataFrame,
        source_rate_hz: float,
        columns_to_process: list,
        aggregation_funcs: Optional[Dict[str, list]] = None
    ) -> pd.DataFrame:
        """
        Downsample high-frequency data by aggregating into windows.
        
        Args:
            df: High frequency dataframe
            source_rate_hz: Source sampling rate in Hz
            columns_to_process: List of columns to downsample
            aggregation_funcs: Dict mapping column name to list of aggregation functions
                              Default: ['mean', 'std', 'min', 'max']
        
        Returns:
            Downsampled DataFrame with aggregated features
        """
        # Calculate window size
        samples_per_window = int(source_rate_hz / self.target_rate_hz)
        
        if aggregation_funcs is None:
            # Default: statistical features
            aggregation_funcs = {col: ['mean', 'std', 'min', 'max'] for col in columns_to_process}
        
        # Number of complete windows
        n_windows = len(df) // samples_per_window
        
        # Prepare output dataframe
        output_data = {}
        
        for col in columns_to_process:
            if col not in df.columns:
                warnings.warn(f"Column {col} not found in dataframe, skipping")
                continue
            
            # Extract data for this column
            col_data = df[col].values
            
            # Get aggregation functions for this column
            agg_funcs = aggregation_funcs.get(col, ['mean', 'std', 'min', 'max'])
            
            for func_name in agg_funcs:
                feature_values = []
                
                for i in range(n_windows):
                    start_idx = i * samples_per_window
                    end_idx = start_idx + samples_per_window
                    window_data = col_data[start_idx:end_idx]
                    
                    # Apply aggregation function
                    if func_name == 'mean':
                        value = np.nanmean(window_data)
                    elif func_name == 'std':
                        value = np.nanstd(window_data)
                    elif func_name == 'min':
                        value = np.nanmin(window_data)
                    elif func_name == 'max':
                        value = np.nanmax(window_data)
                    elif func_name == 'median':
                        value = np.nanmedian(window_data)
                    elif func_name == 'range':
                        value = np.nanmax(window_data) - np.nanmin(window_data)
                    else:
                        value = np.nan
                    
                    feature_values.append(value)
                
                # Store in output
                output_col_name = f"{col}_{func_name}"
                output_data[output_col_name] = feature_values
        
        return pd.DataFrame(output_data)


def create_windowed_dataset(
    synced_df: pd.DataFrame,
    window_size: int = 5,
    stride: int = 1,
    target_column: str = 'h2_concentration',
    feature_columns: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create windowed dataset for time-series modeling.
    
    Args:
        synced_df: Synchronized dataframe
        window_size: Number of time steps in each window
        stride: Stride between windows
        target_column: Column to use as target for labeling
        feature_columns: List of feature columns (if None, uses all except target)
        
    Returns:
        Tuple of (X_windows, y_labels, target_values)
        - X_windows: (n_windows, window_size, n_features)
        - y_labels: (n_windows,) - binary labels
        - target_values: (n_windows,) - actual H2 concentration values
    """
    if feature_columns is None:
        # Use all columns except target and sample index
        feature_columns = [col for col in synced_df.columns 
                          if col not in [target_column, 'sample', 'alarm']]
    
    # Extract features and target
    X_data = synced_df[feature_columns].values
    y_data = synced_df[target_column].values
    
    # Create windows
    n_windows = (len(X_data) - window_size) // stride + 1
    
    X_windows = []
    y_labels = []
    y_targets = []
    
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        if end_idx > len(X_data):
            break
        
        # Extract window
        window_features = X_data[start_idx:end_idx]
        
        # Target is at end of window (or max in window)
        window_target = np.max(y_data[start_idx:end_idx])
        
        # Binary label: leak if H2 >= 1%
        label = 1 if window_target >= 1.0 else 0
        
        X_windows.append(window_features)
        y_labels.append(label)
        y_targets.append(window_target)
    
    X_windows = np.array(X_windows)
    y_labels = np.array(y_labels)
    y_targets = np.array(y_targets)
    
    print(f"\nWindowed dataset created:")
    print(f"  X shape: {X_windows.shape}")
    print(f"  y labels shape: {y_labels.shape}")
    print(f"  Label distribution: {np.bincount(y_labels)}")
    
    return X_windows, y_labels, y_targets


# Example usage
if __name__ == "__main__":
    print("Synchronizer module - use via import")
