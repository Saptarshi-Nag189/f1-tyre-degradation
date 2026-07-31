"""Data collection script — incremental per-event processing to handle API rate limits."""
import sys
import os
from pathlib import Path
import gc

# Anchor all paths to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import fastf1
from data_acquisition.collector import F1DataCollector
from utils.setup import setup_environment, load_config
from utils.data_validation import F1DataCleaner
import pickle
import logging
import time

def main():
    """Main data collection pipeline.
    
    Processes one EVENT at a time to handle API rate limits (500 calls/hr)
    and MemoryErrors gracefully. FastF1 cache will allow resuming instantly.
    """
    print("F1 Data Collection Pipeline (Per-Event Incremental)")
    print("=" * 50)
    
    setup_environment(project_root=PROJECT_ROOT)
    
    config = load_config(str(PROJECT_ROOT / 'config' / 'config.yaml'))
    if not config:
        print("❌ Configuration not found")
        sys.exit(1)
    
    processed_dir = PROJECT_ROOT / 'data' / 'processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    cleaned_path = processed_dir / 'f1_cleaned_data.pkl'
    if cleaned_path.exists():
        with open(cleaned_path, 'rb') as f:
            all_cleaned = pickle.load(f)
        print(f"✓ Loaded existing data containing years: {list(all_cleaned.keys())}")
    else:
        all_cleaned = {}
    
    cleaner = F1DataCleaner()
    collector = F1DataCollector(years=config['data']['years'], sessions=config['data']['sessions'])
    years_to_collect = config['data']['years']
    
    for year in years_to_collect:
        if year not in all_cleaned:
            all_cleaned[year] = {}
            
        print(f"\n{'='*40}")
        print(f"Checking {year} season...")
        print(f"{'='*40}")
        
        try:
            schedule = fastf1.get_event_schedule(year)
            race_events = schedule[schedule['EventFormat'].isin([
                'conventional', 'sprint_shootout', 'sprint_qualifying', 'sprint'
            ])]
        except Exception as e:
            logging.error(f"Failed to load schedule for {year}: {e}")
            continue
            
        events_to_fetch = [e['EventName'] for _, e in race_events.iterrows() 
                          if 'test' not in e['EventName'].lower() and e['EventName'] not in all_cleaned[year]]
        
        if not events_to_fetch:
            print(f"✓ All events for {year} already collected.")
            continue
            
        print(f"Need to collect {len(events_to_fetch)} remaining events for {year}.")
        
        for event_name in events_to_fetch:
            print(f"  -> Collecting {event_name}")
            try:
                event_data = collector.collect_event_data(year, event_name)
                
                if event_data:
                    cleaned_event = cleaner.clean_session_data(event_data)
                    all_cleaned[year][event_name] = cleaned_event
                    
                    # Save immediately after each event
                    with open(cleaned_path, 'wb') as f:
                        pickle.dump(all_cleaned, f)
                    print(f"     ✓ Saved {event_name}")
                else:
                    print(f"     ⚠ No usable data for {event_name}")
                    
                gc.collect()
                time.sleep(0.5) # small delay
                
            except Exception as e:
                if 'RateLimitExceeded' in str(type(e).__name__) or 'RateLimitExceeded' in str(e):
                    print(f"\n❌ FASTF1 API RATE LIMIT EXCEEDED (500 calls/hr).")
                    print(f"Progress has been saved up to {year} - {event_name}.")
                    print(f"Please wait until the rate limit resets (up to 1 hour) and run again.")
                    sys.exit(1)
                else:
                    logging.error(f"Failed to collect {year} {event_name}: {e}")
                    continue

    _print_summary(all_cleaned)
    print(f"\nNext: .venv\\Scripts\\python scripts\\gpu_enhanced_training.py")

def _print_summary(cleaned_data):
    total_laps = 0
    for year_data in cleaned_data.values():
        for event_data in year_data.values():
            for session_data in event_data.values():
                if isinstance(session_data, dict) and 'laps' in session_data:
                    if session_data['laps'] is not None and hasattr(session_data['laps'], '__len__'):
                        total_laps += len(session_data['laps'])
    
    print(f"\n{'='*40}")
    print(f"Data Summary:")
    print(f"  Total laps: {total_laps:,}")
    for year in sorted(cleaned_data.keys()):
        events = len(cleaned_data[year])
        print(f"  {year}: {events} events")

if __name__ == "__main__":
    main()
