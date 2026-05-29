import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from configs.config import(
    METADATA_CSV,
    TRAIN_CSV,
    TEST_CSV,
    RANDOM_SEED
)

df = pd.read_csv(METADATA_CSV) #read the metadata

df["binary_label"] = df["Label"].apply(
    lambda x: 0 if "healthy" in x.lower() else 1
)#healthy = 0, diseased = 1

metadata_columns=[
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

#Normalize
scaler = StandardScaler()

df[metadata_columns] = scaler.fit_transform(
    df[metadata_columns]
)

processed_df = df[metadata_columns +
    ["Image Path", "binary_label"]
]

#split for train and test 
train_df, test_df = train_test_split(
    processed_df,
    test_size=0.2,
    stratify=processed_df["binary_label"],
    random_state=RANDOM_SEED
)
train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)


# ==========================================
# PRINT SUMMARY
# ==========================================

print("========== PREPROCESSING COMPLETE ==========")

print(f"Total Samples : {len(processed_df)}")

print(f"Train Samples : {len(train_df)}")

print(f"Test Samples  : {len(test_df)}")

print("============================================")