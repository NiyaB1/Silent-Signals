import pandas as pd
import numpy as np

# 1. Load the CSV (no header in file, so we define it)
df = pd.read_csv(
    "data/interaction_log.csv",
    names=["timestamp", "event", "source"]
)

# 2. Sort events by time (important for correctness)
df = df.sort_values("timestamp").reset_index(drop=True)

# 3. Convert absolute timestamps to relative time (seconds since start)
start_time = df["timestamp"].min()
df["relative_time"] = df["timestamp"] - start_time

# 4. Define window size (in seconds)
WINDOW_SIZE = 10

# 5. Assign each event to a window
df["window_id"] = (df["relative_time"] // WINDOW_SIZE).astype(int)

# 6. Inspect result
print("Total windows:", df["window_id"].nunique())
print("\nSample windowed data:")
print(df.head())