## Gets summaries for each 10 second window created in preprocessing.py. ## These summaries will be used as features for the ML model.

import pandas as pd
from preprocessing import load_and_window_data

def extract_features(window_size=10):
    """Extract behavioral features per window."""
    
    # Load and window the data
    df = load_and_window_data(window_size=window_size)
    
    if df.empty:
        return pd.DataFrame()
    
    # Group by (session_id, window_id) and count event types
    features_list = []
    
    for (session_id, window_id), group in df.groupby(["session_id", "window_id"]):
        
        # Count each event type
        key_presses = len(group[group["event"] == "key_press"])
        key_releases = len(group[group["event"] == "key_release"])
        mouse_moves = len(group[group["event"] == "mouse_move"])
        mouse_clicks = len(group[group["event"] == "mouse_click"])
        scroll_ups = len(group[group["event"] == "scroll_up"])
        scroll_downs = len(group[group["event"] == "scroll_down"])
        
        # Derived metrics
        typing_speed = key_presses / window_size if window_size > 0 else 0
        mouse_activity = mouse_moves + mouse_clicks + scroll_ups + scroll_downs
        total_events = len(group)
        
        # Create feature row
        feature_row = {
            "session_id": session_id,
            "window_id": window_id,
            "key_presses": key_presses,
            "key_releases": key_releases,
            "mouse_moves": mouse_moves,
            "mouse_clicks": mouse_clicks,
            "scroll_ups": scroll_ups,
            "scroll_downs": scroll_downs,
            "typing_speed": typing_speed,
            "mouse_activity": mouse_activity,
            "total_events": total_events,
        }
        
        features_list.append(feature_row)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    return features_df

if __name__ == "__main__":
    features = extract_features()
    print(f"Extracted {len(features)} windows")
    print(features.head())