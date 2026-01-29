"""
Data loading utilities for H2 gas leak detection.

This module provides functions to load and parse:
- Concentration data (Excel files with dual-header format)
- Pressure data (tab-separated text files)
- Mass flow rate data (custom text format with metadata)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Optional, Iterator
import warnings


class ConcentrationLoader:
    """Load H2 concentration data from Excel files."""
    
    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize loader with file path.
        
        Args:
            file_path: Path to concentration Excel file
        """
        self.file_path = Path(file_path)
        
    def load(self) -> pd.DataFrame:
        """
        Load concentration data with proper header handling.
        
        Returns:
            DataFrame with columns: Time, Output (%), Pt100 (°C), RH (%), Alarm, etc.
        """
        try:
            # Read the file - the structure is:
            # Row 0: Device configuration labels (Device ID, Factory Device ID, ...)
            # Row 1: Device configuration values (04DC12, 04DC12, ...)
            # Row 2: Column names (Iteration, Time, Output (%), ...)
            # Row 3+: Data
            df = pd.read_excel(self.file_path, skiprows=[0, 1], header=0)  # Skip rows 0,1, use row 2 as header
            
            # Convert Output (%) to numeric
            if 'Output (%)' in df.columns:
                df['Output (%)'] = pd.to_numeric(df['Output (%)'], errors='coerce')
            
            # Convert Alarm to numeric
            if 'Alarm' in df.columns:
                df['Alarm'] = pd.to_numeric(df['Alarm'], errors='coerce')
            
            # Convert temperature and humidity
            if 'Pt100 (°C)' in df.columns:
                df['Pt100 (°C)'] = pd.to_numeric(df['Pt100 (°C)'], errors='coerce')
            if 'RH (%)' in df.columns:
                df['RH (%)'] = pd.to_numeric(df['RH (%)'], errors='coerce')
            
            # Parse time if possible
            if 'Time' in df.columns:
                try:
                    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f', errors='coerce')
                except:
                    warnings.warn(f"Could not parse Time column in {self.file_path.name}")
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.file_path.name}: {e}")
    
    @staticmethod
    def get_h2_concentration(df: pd.DataFrame) -> pd.Series:
        """Extract H2 concentration column."""
        if 'Output (%)' not in df.columns:
            raise ValueError("'Output (%)' column not found in data")
        return df['Output (%)']
    
    @staticmethod
    def get_alarm_status(df: pd.DataFrame) -> Optional[pd.Series]:
        """Extract alarm status if available."""
        if 'Alarm' in df.columns:
            return df['Alarm']
        return None


class PressureLoader:
    """Load pressure sensor data from text files."""
    
    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize loader with file path.
        
        Args:
            file_path: Path to pressure text file
        """
        self.file_path = Path(file_path)
        
    def load(self, chunksize: Optional[int] = None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """
        Load pressure data with proper header handling.
        
        Args:
            chunksize: If specified, returns iterator for chunked reading
            
        Returns:
            DataFrame or iterator with columns: Time, P1, Trigger, Event channels
        """
        try:
            # File structure:
            # Lines 0-5 (indices): Metadata (6 lines)
            # Line 6: Column names (Time, P1, Trigger, ...)
            # Line 7: Units row (s, kPa, V, ...)
            # Line 8+: Data
            # We skip metadata (6 lines) and use line 6 as header, then skip line 7 (units)
            
            # First read to get header
            with open(self.file_path, 'r', encoding='utf-8') as f:
                # Skip 6 metadata lines
                for _ in range(6):
                    f.readline()
                # Read header line
                header_line = f.readline().strip()
                column_names = [col.strip() for col in header_line.split('\t') if col.strip()]
            
            # Now read data, skipping first 8 lines (6 metadata + 1 header + 1 units)
            df = pd.read_csv(
                self.file_path,
                sep='\t',
                skiprows=8,  # Skip 6 metadata + 1 header + 1 units
                names=column_names,  # Use our extracted column names
                chunksize=chunksize,
                encoding='utf-8',
                on_bad_lines='skip'
            )
            
            if chunksize is None:
                # Convert columns to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.file_path.name}: {e}")
    
    def load_metadata(self) -> Dict[str, str]:
        """
        Extract metadata from file header.
        
        Returns:
            Dictionary with metadata fields
        """
        metadata = {}
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = [f.readline() for _ in range(6)]
                
            # Parse metadata lines
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
                    
        except Exception as e:
            warnings.warn(f"Could not extract metadata from {self.file_path.name}: {e}")
            
        return metadata


class MFRLoader:
    """Load mass flow rate data from custom text format."""
    
    def __init__(self, folder_path: Union[str, Path]):
        """
        Initialize loader with folder containing CH1, CH2, CH3 files.
        
        Args:
            folder_path: Path to MFR data folder
        """
        self.folder_path = Path(folder_path)
        
    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Load all MFR channels from folder.
        
        Returns:
            Dictionary with keys 'CH1', 'CH2', 'CH3' and DataFrame values
        """
        channels = {}
        
        # Find channel files
        ch_files = sorted(self.folder_path.glob('CH*.TXT'))
        
        for ch_file in ch_files:
            channel_name = ch_file.stem.split('_')[0]  # e.g., 'CH1' from 'CH1_01h.TXT'
            
            try:
                # Read metadata header
                metadata = self._read_metadata(ch_file)
                
                # Read data (skip header lines until 'DATA START')
                with open(ch_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                data_start_idx = None
                for i, line in enumerate(lines):
                    if 'DATA START' in line:
                        data_start_idx = i + 1
                        break
                
                if data_start_idx is None:
                    raise ValueError(f"'DATA START' marker not found in {ch_file.name}")
                
                # Read data values
                data_values = []
                for line in lines[data_start_idx:]:
                    line = line.strip()
                    if line:
                        try:
                            data_values.append(float(line))
                        except ValueError:
                            continue
                
                # Create DataFrame
                df = pd.DataFrame({
                    channel_name: data_values
                })
                
                # Apply unit conversions based on channel
                if channel_name == 'CH2':
                    # Mass flow rate: mfr (g/s) = (V - 1) * 6.25
                    df[f'{channel_name}_mfr_gs'] = (df[channel_name] - 1) * 6.25
                elif channel_name == 'CH3':
                    # Pressure: p (bar) = 62.5 * (V - 1)
                    df[f'{channel_name}_pressure_bar'] = 62.5 * (df[channel_name] - 1)
                
                # Add metadata as attributes
                df.attrs = metadata
                
                channels[channel_name] = df
                
            except Exception as e:
                warnings.warn(f"Failed to load {ch_file.name}: {e}")
                continue
        
        return channels
    
    def _read_metadata(self, file_path: Path) -> Dict[str, str]:
        """Read metadata from MFR file header."""
        metadata = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if 'DATA START' in line:
                    break
                if line and not line.startswith('Nicolet'):
                    # Try to extract key-value pairs
                    parts = line.split('\t') if '\t' in line else [line]
                    if len(parts) == 2:
                        metadata[parts[0]] = parts[1]
                    elif len(parts) == 1 and parts[0]:
                        # Store single-line values
                        if 'trigger' in line.lower():
                            metadata['Trigger Time'] = line
                        elif 'trace' in line.lower():
                            metadata['Trace Type'] = line
                        elif line.replace('.', '').isdigit() or line.startswith('-'):
                            # Numeric values
                            if 'Time per sample' not in metadata:
                                metadata['Time per sample (s)'] = line
        
        return metadata


# Convenience functions

def load_concentration_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """Quick load concentration data."""
    loader = ConcentrationLoader(file_path)
    return loader.load()


def load_pressure_file(file_path: Union[str, Path], chunksize: Optional[int] = None) -> pd.DataFrame:
    """Quick load pressure data."""
    loader = PressureLoader(file_path)
    return loader.load(chunksize=chunksize)


def load_mfr_folder(folder_path: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """Quick load MFR data."""
    loader = MFRLoader(folder_path)
    return loader.load()


def load_test_scenario(test_id: str, base_path: Union[str, Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Load all data for a single test scenario.
    
    Args:
        test_id: Test ID (e.g., 'HTE242USN00006')
        base_path: Base directory containing data folders
        
    Returns:
        Dictionary with 'concentration', 'pressure', and 'mfr' DataFrames
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent.parent
    else:
        base_path = Path(base_path)
    
    data = {}
    
    # Load concentration
    cont_path = base_path / 'HTE242USN00011CONT'
    cont_files = list(cont_path.glob(f'{test_id}*.xlsx'))
    if cont_files:
        data['concentration'] = load_concentration_file(cont_files[0])
    
    # Load pressure
    press_path = base_path / 'HTE242USN00011PRESS'
    press_files = list(press_path.glob(f'{test_id}*.txt'))
    if press_files:
        data['pressure'] = load_pressure_file(press_files[0])
    
    # Load MFR
    mfr_path = base_path / 'HTE242USN00011MFR' / f'{test_id}MFR190621'
    if mfr_path.exists():
        data['mfr'] = load_mfr_folder(mfr_path)
    
    return data
