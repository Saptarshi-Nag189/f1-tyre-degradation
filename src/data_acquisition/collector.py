"""F1 data collection from FastF1 API."""
import fastf1
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class F1DataCollector:
    """Collects and structures F1 session data from FastF1.
    
    Attributes:
        years: List of years to collect data for.
        sessions: List of session types to collect.
        logger: Logger instance for this class.
    """
    
    def __init__(self, years: List[int], sessions: List[str] = None):
        """Initialize the data collector.
        
        Args:
            years: List of years or single year to collect data for.
            sessions: List of session types. Defaults to ['Practice 3', 'Qualifying', 'Race'].
        """
        self.years = years if isinstance(years, list) else [years]
        self.sessions = sessions or ['Practice 3', 'Qualifying', 'Race']
        self.logger = logging.getLogger(__name__)
        
    def collect_all_data(self) -> Dict:
        """Collect data for all specified years and sessions.
        
        Returns:
            Dict: Nested dictionary with year -> event -> session -> data structure.
        """
        all_data = {}
        for year in self.years:
            self.logger.info(f"Collecting {year} season data")
            try:
                all_data[year] = self.collect_season_data(year)
            except ValueError as e:
                # Handle schedule loading failures (FastF1 API issue)
                self.logger.error(f"Failed to load schedule for {year}: {e}")
                self.logger.warning(f"Skipping {year} season, continuing with other years")
                continue
        return all_data
    
    def collect_season_data(self, year: int) -> Dict:
        """Collect all race data for a season.
        
        Args:
            year: The year to collect data for.
            
        Returns:
            Dict: Dictionary with event name -> session data.
        """
        season_data = {}
        schedule = fastf1.get_event_schedule(year)
        race_events = schedule[schedule['EventFormat'].isin([
            'conventional', 'sprint_shootout', 'sprint_qualifying', 'sprint'
        ])]
        
        for _, event in tqdm(race_events.iterrows(), desc=f"{year}", total=len(race_events)):
            if 'test' in event['EventName'].lower():
                continue
            
            event_data = self.collect_event_data(year, event['EventName'])
            if event_data:
                season_data[event['EventName']] = event_data
        
        return season_data
    
    def collect_event_data(self, year: int, event_name: str) -> Dict:
        """Collect data for a single event across multiple sessions.
        
        Args:
            year: The year of the event.
            event_name: Name of the event.
            
        Returns:
            Dict: Dictionary with session type -> session data.
        """
        event_data = {}
        
        for session_type in self.sessions:
            try:
                session = fastf1.get_session(year, event_name, session_type)
                session.load()
                
                if not self._validate_session(session):
                    continue
                
                event_data[session_type] = {
                    'laps': session.laps,
                    'results': session.results,
                    'weather': session.weather_data,
                    'telemetry': self._extract_telemetry_sample(session),
                    'session_info': {
                        'year': year,
                        'event': event_name,
                        'session': session_type,
                        'date': session.date
                    }
                }
            except Exception as e:
                self.logger.error(f"Failed: {year} {event_name} {session_type} - {e}")
        
        return event_data
    
    def _validate_session(self, session) -> bool:
        """Check session has minimum required data.
        
        Args:
            session: FastF1 session object.
            
        Returns:
            bool: True if session has at least 10 valid laps, False otherwise.
        """
        return session.laps is not None and not session.laps.empty and len(session.laps) >= 10
    
    def _extract_telemetry_sample(self, session) -> Dict:
        """Extract representative telemetry samples to reduce memory usage.
        
        Samples every 3rd valid lap per driver to reduce data volume.
        
        Args:
            session: FastF1 session object.
            
        Returns:
            Dict: Dictionary with driver code -> telemetry DataFrame.
        """
        telemetry_data = {}
        
        for driver in session.laps['Driver'].unique():
            try:
                driver_laps = session.laps.pick_driver(driver)
                valid_laps = driver_laps[
                    (driver_laps['PitOutTime'].isna()) &
                    (driver_laps['PitInTime'].isna()) &
                    (driver_laps['LapTime'].notna())
                ]
                
                if len(valid_laps) < 3:
                    continue
                
                # Sample every 3rd lap to reduce data volume
                sampled_laps = valid_laps.iloc[::3]
                tel_data = []
                
                for _, lap in sampled_laps.iterrows():
                    try:
                        tel = lap.get_telemetry().add_distance()
                        tel['LapNumber'] = lap['LapNumber']
                        tel['Driver'] = driver
                        if lap['LapTime'] is not None:
                            tel['LapTime'] = lap['LapTime'].total_seconds()
                        tel['Compound'] = lap['Compound']
                        tel['TyreLife'] = lap['TyreLife']
                        tel_data.append(tel)
                    except Exception:
                        continue
                
                if tel_data:
                    telemetry_data[driver] = pd.concat(tel_data, ignore_index=True)
            except Exception:
                continue
        
        return telemetry_data
