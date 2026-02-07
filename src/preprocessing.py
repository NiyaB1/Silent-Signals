import pandas as pd

def load_and_window_data(csv_path="data/interaction_log.csv", window_size=10):
    """Load interaction log and assign events to time windows."""
    
    # Load CSV with session_id column
    df = pd.read_csv(
        csv_path,
        names=["timestamp", "event", "source", "session_id"]
    )
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Compute relative time (seconds since start)
    start_time = df["timestamp"].min()
    df["relative_time"] = df["timestamp"] - start_time
    
    # Assign to windows (10-second buckets)
    df["window_id"] = (df["relative_time"] // window_size).astype(int)
    
    return df

if __name__ == "__main__":
    data = load_and_window_data()
    print(f"Total events: {len(data)}")
    print(f"Total windows: {data['window_id'].nunique()}")
    print(data.head())