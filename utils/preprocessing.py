import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from configs.config import(
    METADATA_CSV,
    TRAIN_CSV,
    TEST_CSV,
    RANDOM_SEED
)

df = pd.read_csv(METADATA_CSV)


class_to_idx = {
    "Apple___Apple_scab" : 0,
    "Apple___Apple_rot" : 1,
    "Apple___Cedar_apple_rust" : 2,
    "Apple___healthy" : 3,
} #classes created.


df["label"] = df["Label"].map(class_to_idx)
if df["label"].isnull().any():
    raise ValueError("Some labels were not found in class_to_idx")

metadata_columns=[
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]



processed_df = df[
    metadata_columns +
    ["Image Path", "label"]
]



#split for train and test 
train_df, test_df = train_test_split(
    processed_df,
    test_size=0.2,
    stratify=processed_df["label"],
    random_state=RANDOM_SEED
)



#standardize after splitting(on the training data)
scaler = StandardScaler()
scaler.fit(train_df[metadata_columns])
train_df[metadata_columns] = scaler.transform(
    train_df[metadata_columns]
)
test_df[metadata_columns] = scaler.transform(
    test_df[metadata_columns]
)


train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print(f"{'='*10}PREPROCESSING{'='*10}")
print(f"Total Samples : {len(processed_df)}")
print(f"Train Samples : {len(train_df)}")
print(f"Test Samples  : {len(test_df)}")
print("="*20)
