"""
Feature engineering for leak detection.

Extracts features from synchronized sensor data including:
- Statistical features
- Temporal features (derivatives, trends)
- Frequency domain features (FFT)
- Cross-sensor features
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from scipy import stats
from scipy.fft import fft, fftfreq


class LeakFeatureEngineer:
    """Extract predictive features from sensor data."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names = []
        
    def extract_all_features(self, synced_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features from synchronized data.
        
        Args:
            synced_df: Synchronized sensor dataframe
            
        Returns:
            DataFrame with engineered features
        """
        features_df = pd.DataFrame()
        
        # Add sample index
        features_df['sample'] = synced_df['sample']
        
        # 1. Statistical features from pressure
        print("Extracting statistical features...")
        if any('pressure_P1' in col for col in synced_df.columns):
            pressure_feats = self._extract_statistical_features(
                synced_df,
                prefix='pressure_P1'
            )
            features_df = pd.concat([features_df, pressure_feats], axis=1)
        
        # 2. Statistical features from MFR
        if any('CH2_mfr' in col for col in synced_df.columns):
            mfr_feats = self._extract_statistical_features(
                synced_df,
                prefix='CH2_mfr'
            )
            features_df = pd.concat([features_df, mfr_feats], axis=1)
        
        # 3. Temporal features (derivatives, trends)
        print("Extracting temporal features...")
        temporal_feats = self._extract_temporal_features(synced_df)
        features_df = pd.concat([features_df, temporal_feats], axis=1)
        
        # 4. Environmental features (temperature, humidity)
        print("Extracting environmental features...")
        if 'temperature' in synced_df.columns:
            features_df['temperature'] = synced_df['temperature']
        if 'humidity' in synced_df.columns:
            features_df['humidity'] = synced_df['humidity']
        
        # 5. Cross-sensor features
        print("Extracting cross-sensor features...")
        cross_feats = self._extract_cross_sensor_features(synced_df)
        features_df = pd.concat([features_df, cross_feats], axis=1)
        
        # 6. Target and label (if available)
        if 'h2_concentration' in synced_df.columns:
            features_df['h2_concentration'] = synced_df['h2_concentration']
        if 'alarm' in synced_df.columns:
            features_df['alarm'] = synced_df['alarm']
        
        # Store feature names (excluding target and sample)
        self.feature_names = [col for col in features_df.columns 
                             if col not in ['sample', 'h2_concentration', 'alarm']]
        
        print(f"\nTotal features extracted: {len(self.feature_names)}")
        print(f"Feature dataframe shape: {features_df.shape}")
        
        return features_df
    
    def _extract_statistical_features(
        self,
        df: pd.DataFrame,
        prefix: str
    ) -> pd.DataFrame:
        """
        Extract statistical features from aggregated sensor data.
        
        Args:
            df: DataFrame with aggregated features (mean, std, min, max)
            prefix: Column prefix to process (e.g., 'pressure_P1')
            
        Returns:
            DataFrame with derived statistical features
        """
        features = pd.DataFrame()
        
        # Find columns with this prefix
        mean_col = f'{prefix}_mean'
        std_col = f'{prefix}_std'
        min_col = f'{prefix}_min'
        max_col = f'{prefix}_max'
        
        if mean_col in df.columns:
            features[f'{prefix}_mean'] = df[mean_col]
        
        if std_col in df.columns:
            features[f'{prefix}_std'] = df[std_col]
            # Coefficient of variation
            if mean_col in df.columns:
                features[f'{prefix}_cv'] = df[std_col] / (df[mean_col] + 1e-10)
        
        if min_col in df.columns and max_col in df.columns:
            # Range
            features[f'{prefix}_range'] = df[max_col] - df[min_col]
            # Normalized range
            if mean_col in df.columns:
                features[f'{prefix}_norm_range'] = (df[max_col] - df[min_col]) / (df[mean_col] + 1e-10)
        
        return features
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features (derivatives, trends).
        
        Args:
            df: Synchronized dataframe
            
        Returns:
            DataFrame with temporal features
        """
        features = pd.DataFrame()
        
        # Pressure rate of change
        if 'pressure_P1_mean' in df.columns:
            pressure = df['pressure_P1_mean'].values
            
            # First derivative (rate of change)
            pressure_diff = np.gradient(pressure)
            features['pressure_rate_of_change'] = pressure_diff
            
            # Second derivative (acceleration)
            pressure_accel = np.gradient(pressure_diff)
            features['pressure_acceleration'] = pressure_accel
            
            # Rolling statistics of derivatives
            window = 5
            features['pressure_diff_std_5'] = pd.Series(pressure_diff).rolling(window, min_periods=1).std().values
            features['pressure_diff_mean_5'] = pd.Series(pressure_diff).rolling(window, min_periods=1).mean().values
        
        # MFR rate of change
        if 'CH2_mfr_mfr_gs_mean' in df.columns:
            mfr = df['CH2_mfr_mfr_gs_mean'].values
            
            # First derivative
            mfr_diff = np.gradient(mfr)
            features['mfr_rate_of_change'] = mfr_diff
            
            # Second derivative
            mfr_accel = np.gradient(mfr_diff)
            features['mfr_acceleration'] = mfr_accel
        
        # Trend detection (linear fit over short window)
        if 'pressure_P1_mean' in df.columns:
            pressure = df['pressure_P1_mean'].values
            trend_window = 10
            
            trends = []
            for i in range(len(pressure)):
                if i < trend_window:
                    trends.append(0)
                else:
                    window_data = pressure[i-trend_window:i]
                    x = np.arange(len(window_data))
                    slope, _ = np.polyfit(x, window_data, 1)
                    trends.append(slope)
            
            features['pressure_trend_10'] = trends
        
        return features
    
    def _extract_cross_sensor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from relationships between sensors.
        
        Args:
            df: Synchronized dataframe
            
        Returns:
            DataFrame with cross-sensor features
        """
        features = pd.DataFrame()
        
        # Pressure-MFR ratio
        if 'pressure_P1_mean' in df.columns and 'CH2_mfr_mfr_gs_mean' in df.columns:
            pressure = df['pressure_P1_mean']
            mfr = df['CH2_mfr_mfr_gs_mean']
            
            features['pressure_mfr_ratio'] = pressure / (mfr + 1e-10)
            features['pressure_mfr_product'] = pressure * mfr
        
        # Pressure-Temperature interaction
        if 'pressure_P1_mean' in df.columns and 'temperature' in df.columns:
            features['pressure_temp_ratio'] = df['pressure_P1_mean'] / (df['temperature'] + 273.15)  # Absolute temp
        
        # Deviation from baseline (first 10 samples assumed no leak)
        if 'pressure_P1_mean' in df.columns:
            baseline_pressure = df['pressure_P1_mean'].iloc[:min(10, len(df))].mean()
            features['pressure_deviation_from_baseline'] = df['pressure_P1_mean'] - baseline_pressure
            features['pressure_pct_change_from_baseline'] = (df['pressure_P1_mean'] - baseline_pressure) / (baseline_pressure + 1e-10) * 100
        
        if 'CH2_mfr_mfr_gs_mean' in df.columns:
            baseline_mfr = df['CH2_mfr_mfr_gs_mean'].iloc[:min(10, len(df))].mean()
            features['mfr_deviation_from_baseline'] = df['CH2_mfr_mfr_gs_mean'] - baseline_mfr
        
        return features


def prepare_dataset_for_ml(
    features_df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    leak_threshold: float = 1.0,
    random_state: int = 42
) -> Dict:
    """
    Prepare dataset for ML training.
    
    Args:
        features_df: DataFrame with engineered features
        test_size: Fraction for test set
        val_size: Fraction for validation set
        leak_threshold: H2 concentration threshold for leak label
        random_state: Random seed
        
    Returns:
        Dictionary with train/val/test splits
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Create binary labels
    if 'h2_concentration' not in features_df.columns:
        raise ValueError("h2_concentration column required for labeling")
    
    y = (features_df['h2_concentration'] >= leak_threshold).astype(int).values
    
    # Get feature columns
    feature_cols = [col for col in features_df.columns 
                   if col not in ['sample', 'h2_concentration', 'alarm']]
    
    X = features_df[feature_cols].values
    
    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check class distribution before splitting
    unique, counts = np.unique(y, return_counts=True)
    min_class_size = counts.min()
    can_stratify = (min_class_size >= 2)  # Need at least 2 samples per class for stratification
    
    if not can_stratify:
        print(f"\n⚠️  WARNING: Skipping stratification due to class imbalance")
        print(f"   Class distribution: {dict(zip(unique, counts))}")
    
    # Split data
    if can_stratify:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
    else:
        # Split without stratification
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_classes = np.unique(y_train)
    
    # Check if we have both classes
    if len(unique_classes) < 2:
        print(f"\n⚠️  WARNING: Only one class found in training data!")
        print(f"   Unique classes: {unique_classes}")
        print(f"   This dataset may not be suitable for binary classification.")
        print(f"   Consider using a different test scenario with leak events.")
        class_weight_dict = {0: 1.0, 1: 1.0}  # Default weights
    else:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        # Apply safety-focused multiplier to leak class if it exists
        if 1 in class_weight_dict:
            class_weight_dict[1] = class_weight_dict[1] * 2.5  # 2.5x emphasis on leak detection
    
    print(f"\n=== Dataset Summary ===")
    print(f"Total samples: {len(X)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    print(f"\nClass distribution (Train):")
    print(f"  No leak: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
    print(f"  Leak: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
    print(f"\nClass weights: {class_weight_dict}")
    
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_cols,
        'scaler': scaler,
        'class_weights': class_weight_dict
    }


# Example usage
if __name__ == "__main__":
    print("Feature engineering module - use via import")
