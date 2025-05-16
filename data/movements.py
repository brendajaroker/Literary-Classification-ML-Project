import pandas as pd

# Load the CSV
df = pd.read_csv("data.csv")

# Drop rows with missing movement labels
df = df[df["Movement"].notna()]

# Count occurrences of each unique movement
movement_counts = df["Movement"].value_counts().sort_index()

# Print the results
print("Distinct Movements and Counts:")
for movement, count in movement_counts.items():
    print(f"- {movement}: {count}")
