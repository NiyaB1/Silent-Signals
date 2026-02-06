import pandas as pd

df = pd.read_csv(
    "data/interaction_log.csv",
    names=["timestamp", "event", "source"]
)

print("Total events:", len(df))

print("\nEvent counts:")
print(df["event"].value_counts())

print("\nFirst 5 rows:")
print(df.head())