"""
Labeling utilities for leak detection.

Creates binary labels from H2 concentration data for supervised learning.
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple
import matplotlib.pyplot as plt


class LeakLabeler:
    """Create labels for leak detection from H2 concentration data."""
    
    def __init__(self, threshold_pct: float = 1.0):
        """
        Initialize labeler with leak threshold.
        
        Args:
            threshold_pct: H2 concentration threshold in % for leak detection
                          Default: 1.0% (early warning level)
        """
        self.threshold_pct = threshold_pct
        
    def create_binary_labels(self, h2_concentration: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Create binary leak labels from H2 concentration.
        
        Args:
            h2_concentration: H2 concentration values in %
            
        Returns:
            Binary labels (0 = no leak, 1 = leak detected)
        """
        if isinstance(h2_concentration, pd.Series):
            h2_concentration = h2_concentration.values
            
        labels = (h2_concentration >= self.threshold_pct).astype(int)
        return labels
    
    def create_severity_labels(self, h2_concentration: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Create multi-class severity labels.
        
        Classes:
        - 0: Safe (H2 < 1%)
        - 1: Warning (1% <= H2 < 4%)
        - 2: Critical (H2 >= 4%, explosive range)
        
        Args:
            h2_concentration: H2 concentration values in %
            
        Returns:
            Severity labels (0, 1, or 2)
        """
        if isinstance(h2_concentration, pd.Series):
            h2_concentration = h2_concentration.values
            
        labels = np.zeros(len(h2_concentration), dtype=int)
        labels[h2_concentration >= 1.0] = 1  # Warning
        labels[h2_concentration >= 4.0] = 2  # Critical (LEL for H2)
        
        return labels
    
    def analyze_class_distribution(self, labels: np.ndarray) -> dict:
        """
        Analyze label distribution for class imbalance.
        
        Args:
            labels: Binary or multi-class labels
            
        Returns:
            Dictionary with class counts and percentages
        """
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        distribution = {}
        for label, count in zip(unique, counts):
            distribution[int(label)] = {
                'count': int(count),
                'percentage': (count / total) * 100
            }
        
        return distribution
    
    def plot_label_distribution(self, labels: np.ndarray, title: str = "Label Distribution"):
        """
        Visualize label distribution.
        
        Args:
            labels: Labels to visualize
            title: Plot title
        """
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        colors = ['green', 'orange', 'red'][:len(unique)]
        bars = plt.bar(unique, counts, color=colors, alpha=0.7, edgecolor='black')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/len(labels)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Label', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        
        if len(unique) == 2:
            plt.xticks([0, 1], ['No Leak', 'Leak'])
        elif len(unique) == 3:
            plt.xticks([0, 1, 2], ['Safe\n(<1%)', 'Warning\n(1-4%)', 'Critical\n(≥4%)'])
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def calculate_class_weights(self, labels: np.ndarray, method='balanced') -> dict:
        """
        Calculate class weights for handling imbalance.
        
        Args:
            labels: Training labels
            method: 'balanced' or 'safety_focused'
                   'balanced': Standard sklearn balanced weights
                   'safety_focused': Extra penalty for missing leaks
        
        Returns:
            Dictionary mapping class to weight
        """
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        if method == 'balanced':
            # Standard balanced weighting
            weights = {}
            for label, count in zip(unique, counts):
                weights[int(label)] = total / (len(unique) * count)
        
        elif method == 'safety_focused':
            # Extra weight on leak class to minimize false negatives
            weights = {}
            for label, count in zip(unique, counts):
                base_weight = total / (len(unique) * count)
                if label == 1:  # Leak class
                    weights[int(label)] = base_weight * 2.5  # 2.5x emphasis
                else:
                    weights[int(label)] = base_weight
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return weights


def create_windowed_labels(
    h2_concentration: pd.Series,
    window_size: int = 5,
    stride: int = 1,
    threshold_pct: float = 1.0,
    aggregation: str = 'max'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create labels for windowed data.
    
    Args:
        h2_concentration: H2 concentration time series
        window_size: Window size in samples (e.g., 5 for 5 seconds at 1 Hz)
        stride: Stride between windows
        threshold_pct: Leak threshold in %
        aggregation: How to aggregate within window ('max', 'mean', 'any')
                    'max': Label as leak if max in window exceeds threshold
                    'mean': Label as leak if mean in window exceeds threshold
                    'any': Label as leak if any sample in window exceeds threshold
    
    Returns:
        Tuple of (window_labels, window_max_concentrations)
    """
    num_windows = (len(h2_concentration) - window_size) // stride + 1
    
    window_labels = []
    window_max_conc = []
    
    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        if end_idx > len(h2_concentration):
            break
        
        window_h2 = h2_concentration.iloc[start_idx:end_idx]
        
        if aggregation == 'max':
            window_value = window_h2.max()
        elif aggregation == 'mean':
            window_value = window_h2.mean()
        elif aggregation == 'any':
            window_value = window_h2.max()  # Same as max for threshold comparison
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Create label
        label = 1 if window_value >= threshold_pct else 0
        
        window_labels.append(label)
        window_max_conc.append(window_h2.max())
    
    return np.array(window_labels), np.array(window_max_conc)


def analyze_leak_onset(h2_concentration: pd.Series, threshold_pct: float = 1.0) -> dict:
    """
    Analyze leak onset characteristics.
    
    Args:
        h2_concentration: H2 concentration time series
        threshold_pct: Leak threshold
        
    Returns:
        Dictionary with leak onset statistics
    """
    labels = (h2_concentration >= threshold_pct).astype(int)
    
    # Find first leak detection
    leak_indices = np.where(labels == 1)[0]
    
    if len(leak_indices) == 0:
        return {
            'leak_detected': False,
            'onset_index': None,
            'onset_time': None,
            'max_concentration': h2_concentration.max(),
        }
    
    onset_idx = leak_indices[0]
    
    # Analyze concentration growth rate
    if onset_idx > 0:
        pre_leak_conc = h2_concentration.iloc[:onset_idx].mean()
        growth_rate = (h2_concentration.iloc[onset_idx] - pre_leak_conc)
    else:
        pre_leak_conc = 0
        growth_rate = h2_concentration.iloc[onset_idx]
    
    return {
        'leak_detected': True,
        'onset_index': int(onset_idx),
        'pre_leak_concentration_mean': float(pre_leak_conc),
        'concentration_at_onset': float(h2_concentration.iloc[onset_idx]),
        'growth_rate_at_onset': float(growth_rate),
        'max_concentration': float(h2_concentration.max()),
        'samples_above_threshold': int(len(leak_indices)),
        'leak_duration_percent': float(len(leak_indices) / len(h2_concentration) * 100),
    }


# Example usage function
def label_dataset_example():
    """Example of how to use the labeling module."""
    # This would typically be called from preprocessing pipeline
    
    # Example H2 concentration data
    h2_conc = pd.Series([0.1, 0.2, 0.5, 0.8, 1.2, 2.5, 4.1, 6.3, 5.9, 3.2])
    
    # Create labeler
    labeler = LeakLabeler(threshold_pct=1.0)
    
    # Binary labels
    binary_labels = labeler.create_binary_labels(h2_conc)
    print(f"Binary labels: {binary_labels}")
    
    # Severity labels
    severity_labels = labeler.create_severity_labels(h2_conc)
    print(f"Severity labels: {severity_labels}")
    
    # Analyze distribution
    distribution = labeler.analyze_class_distribution(binary_labels)
    print(f"Label distribution: {distribution}")
    
    # Calculate class weights
    weights = labeler.calculate_class_weights(binary_labels, method='safety_focused')
    print(f"Class weights: {weights}")
    
    # Analyze leak onset
    onset_info = analyze_leak_onset(h2_conc, threshold_pct=1.0)
    print(f"Leak onset analysis: {onset_info}")


if __name__ == "__main__":
    label_dataset_example()
