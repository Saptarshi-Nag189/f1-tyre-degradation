"""Data validation and cleaning utilities."""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging


class F1DataValidator:
    """Validates F1 data against domain-specific rules.
    
    Validation rules follow Feature Contract Appendix Section 5.
    """
    
    def __init__(self):
        """Initialize validator with F1-specific validation rules."""
        self.validation_rules = {
            'speed_limits': {'min': 0, 'max': 400},
            'temperature_limits': {'ambient': (-10, 60), 'track': (0, 80)},
            'lap_time_limits': {'min': 60, 'max': 300}
        }
        self.logger = logging.getLogger(__name__)
    
    def validate_session_data(self, session_data: Dict) -> Tuple[Dict, Dict]:
        """Validate and fix session data.
        
        Args:
            session_data: Dictionary containing laps, weather, telemetry data.
            
        Returns:
            Tuple of (validated_data, report_dict).
        """
        validated = session_data.copy()
        report = {'warnings': [], 'fixes_applied': []}
        
        if 'laps' in session_data and session_data['laps'] is not None:
            validated['laps'], lap_report = self._validate_laps_data(session_data['laps'])
            report['fixes_applied'].extend(lap_report.get('fixes_applied', []))
        
        if 'weather' in session_data and session_data['weather'] is not None:
            validated['weather'], weather_report = self._validate_weather(session_data['weather'])
            report['fixes_applied'].extend(weather_report.get('fixes_applied', []))
        
        if 'telemetry' in session_data:
            validated['telemetry'], tel_report = self._validate_telemetry(session_data['telemetry'])
            report['fixes_applied'].extend(tel_report.get('fixes_applied', []))
        
        return validated, report
    
    def _validate_laps_data(self, laps_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Validate lap data and fix common issues.
        
        Applies validation per Feature Contract:
        - LapTime: 60-300 seconds range
        - GridPosition: 0 or NaN -> 25 (pit start)
        
        Args:
            laps_df: DataFrame containing lap data.
            
        Returns:
            Tuple of (validated DataFrame, report dict).
        """
        df = laps_df.copy()
        report = {'fixes_applied': []}
        
        # Fix pit lane starts (GridPosition validation per Feature Contract)
        if 'GridPosition' in df.columns:
            pit_starts = (df['GridPosition'] == 0.0) | (df['GridPosition'].isna())
            if pit_starts.any():
                df.loc[pit_starts, 'GridPosition'] = 25
                report['fixes_applied'].append(f"Fixed {pit_starts.sum()} pit starts")
        
        # Remove invalid lap times (LapTime validation per Feature Contract)
        if 'LapTime' in df.columns:
            try:
                lap_times = pd.to_timedelta(df['LapTime']).dt.total_seconds()
            except (ValueError, TypeError):
                lap_times = df['LapTime']
            
            limits = self.validation_rules['lap_time_limits']
            invalid = (lap_times < limits['min']) | (lap_times > limits['max'])
            if invalid.any():
                df = df[~invalid].copy()
                report['fixes_applied'].append(f"Removed {invalid.sum()} invalid lap times")
        
        return df, report
    
    def _validate_weather(self, weather_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Validate and fix weather data.
        
        Applies validation per Feature Contract:
        - AirTemp: -10 to 60°C, invalid -> NaN -> median fill
        - TrackTemp: 0 to 80°C, invalid -> NaN -> median fill
        
        Args:
            weather_df: DataFrame containing weather data.
            
        Returns:
            Tuple of (validated DataFrame, report dict).
        """
        df = weather_df.copy()
        report = {'fixes_applied': []}
        
        for col in ['AirTemp', 'TrackTemp']:
            if col not in df.columns:
                continue
            
            limit_key = 'ambient' if 'Air' in col else 'track'
            min_temp, max_temp = self.validation_rules['temperature_limits'][limit_key]
            
            invalid = (df[col] < min_temp) | (df[col] > max_temp)
            if invalid.any():
                df.loc[invalid, col] = np.nan
                report['fixes_applied'].append(f"Cleaned {invalid.sum()} {col} values")
            
            # Fill missing with median or reasonable default
            if df[col].isna().any():
                fill_value = df[col].median() if not df[col].isna().all() else (25.0 if 'Air' in col else 35.0)
                df[col] = df[col].fillna(fill_value)
        
        return df, report
    
    def _validate_telemetry(self, telemetry_dict: Dict) -> Tuple[Dict, Dict]:
        """Validate telemetry data for all drivers.
        
        Applies validation per Feature Contract:
        - Speed: 0-400 km/h range, invalid rows removed
        - Minimum 100 data points per lap
        
        Args:
            telemetry_dict: Dictionary of driver -> telemetry DataFrame.
            
        Returns:
            Tuple of (validated dict, report dict).
        """
        validated = {}
        report = {'fixes_applied': []}
        
        for driver, tel_df in telemetry_dict.items():
            if tel_df is None or tel_df.empty:
                continue
            
            # Speed validation per Feature Contract
            if 'Speed' in tel_df.columns:
                limits = self.validation_rules['speed_limits']
                invalid = (tel_df['Speed'] < limits['min']) | (tel_df['Speed'] > limits['max'])
                if invalid.any():
                    tel_df = tel_df[~invalid].copy()
                    report['fixes_applied'].append(f"{driver}: removed {invalid.sum()} invalid speeds")
            
            # Minimum telemetry points per Feature Contract
            if len(tel_df) >= 100:
                validated[driver] = tel_df
        
        return validated, report


class F1DataCleaner:
    """High-level data cleaning orchestrator."""
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.logger = logging.getLogger(__name__)
    
    def clean_session_data(self, raw_session_data: Dict) -> Dict:
        """Clean all sessions in dataset.
        
        Args:
            raw_session_data: Raw session data dictionary.
            
        Returns:
            Dict: Cleaned session data.
        """
        cleaned = {}
        validator = F1DataValidator()
        
        for session_key, session_data in raw_session_data.items():
            try:
                validated, report = validator.validate_session_data(session_data)
                cleaned[session_key] = self._apply_domain_cleaning(validated)
                
                if report['fixes_applied']:
                    self.logger.info(f"{session_key}: {len(report['fixes_applied'])} fixes")
            except Exception as e:
                self.logger.error(f"Cleaning failed for {session_key}: {e}")
        
        return cleaned
    
    def _apply_domain_cleaning(self, session_data: Dict) -> Dict:
        """Apply F1-specific cleaning rules.
        
        Normalizes Compound names per Feature Contract:
        - S -> SOFT, M -> MEDIUM, H -> HARD
        - I -> INTERMEDIATE, W -> WET
        
        Args:
            session_data: Validated session data.
            
        Returns:
            Dict: Cleaned session data with normalized compounds.
        """
        if 'laps' in session_data and session_data['laps'] is not None:
            if 'Compound' in session_data['laps'].columns:
                # Standardize compound names per Feature Contract
                compound_map = {
                    'SOFT': 'SOFT', 'MEDIUM': 'MEDIUM', 'HARD': 'HARD',
                    'INTERMEDIATE': 'INTERMEDIATE', 'WET': 'WET',
                    'S': 'SOFT', 'M': 'MEDIUM', 'H': 'HARD',
                    'I': 'INTERMEDIATE', 'W': 'WET'
                }
                session_data['laps']['Compound'] = (
                    session_data['laps']['Compound']
                    .map(compound_map)
                    .fillna('MEDIUM')
                )
        
        return session_data
