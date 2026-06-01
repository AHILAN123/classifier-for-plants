import pandas as pd
from configs.config import METADATA_CSV

# Load original metadata
df = pd.read_csv(METADATA_CSV)

# Keep only the required tomato classes
required_classes = [
    "Apple___Apple_scab",
    "Apple___Apple_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]
# Filter dataframe
tomato_df = df[
    df["Label"].isin(required_classes)
].copy()

# Save filtered metadata
output_path = "data/apple_metadata.csv"

tomato_df.to_csv(
    output_path,
    index=False
)

print("=" * 50)
print("TOMATO DATASET CREATED")
print("=" * 50)
print(f"Original Samples : {len(df)}")
print(f"Tomato Samples   : {len(tomato_df)}")
print(f"Saved To         : {output_path}")
print("=" * 50)
