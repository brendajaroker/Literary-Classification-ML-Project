import pandas as pd

df = pd.read_csv("data.csv")

df = df[df["Movement"].notna()]

# Count occurrences of each unique movement
movement_counts = df["Movement"].value_counts().sort_index()

print("Distinct Movements and Counts:")
for movement, count in movement_counts.items():
    print(f"- {movement}: {count}")
