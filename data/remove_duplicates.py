import pandas as pd

# Define the filename
filename = "data.csv"  # Replace with your actual CSV file name

# Load the CSV
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"Error: {filename} not found. Please ensure the file exists.")
    exit(1)

print(f"Initial row count: {len(df)}")

# Drop rows with missing or empty GutenbergID
df = df[df["GutenbergID"].notna() & (df["GutenbergID"].astype(str).str.strip() != "")]
print(f"Row count after dropping missing/empty GutenbergID: {len(df)}")

# Identify and print duplicates based on GutenbergID
duplicate_gutenberg_ids = df[df.duplicated(subset="GutenbergID", keep="first")]
if not duplicate_gutenberg_ids.empty:
    print("\nDropping the following rows due to duplicate GutenbergID:")
    for idx, row in duplicate_gutenberg_ids.iterrows():
        print(f"GutenbergID: {row['GutenbergID']}, Title: {row['Title']}")
df = df.drop_duplicates(subset="GutenbergID", keep="first")
print(f"Row count after dropping duplicate GutenbergID: {len(df)}")

# Identify and print duplicates based on Title
duplicate_titles = df[df.duplicated(subset="Title", keep="first")]
if not duplicate_titles.empty:
    print("\nDropping the following rows due to duplicate Title:")
    for idx, row in duplicate_titles.iterrows():
        print(f"GutenbergID: {row['GutenbergID']}, Title: {row['Title']}")
df = df.drop_duplicates(subset="Title", keep="first")
print(f"Row count after dropping duplicate Title: {len(df)}")

# Overwrite the original file
df.to_csv(filename, index=False)

print(f"\n{filename} has been updated with unique GutenbergIDs and Titles.")
print(f"Final row count: {len(df)}")