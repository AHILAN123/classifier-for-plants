import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.transforms import (
    train_transforms,
    val_transforms
)
from configs.config import IMAGE_DIR

class PlantDiseaseDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        # Load processed CSV
        self.dataframe = dataframe
        # Image transforms
        self.transform = transform
        # Metadata feature columns
        self.metadata_columns =[
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]
    #total num of samples. 
    def __len__(self):
        return len(self.dataframe)
  
   #get a sample.
    def __getitem__(self, idx):
        #get row
        row = self.dataframe.iloc[idx]

        #load image
        image_path = row["Image Path"]

        # Convert path separators for compatibility
        image_path = image_path.replace("\\", "/")

        # Full image path
        full_image_path = os.path.join(
            IMAGE_DIR,
            os.path.basename(os.path.dirname(image_path)),
            os.path.basename(image_path)
        )

        # Open image
        image = Image.open(full_image_path).convert("RGB")

        #apply transforms : _ 
        if self.transform:
            image = self.transform(image)

        #load metadata
        metadata = row[self.metadata_columns].values.astype("float32")
        metadata = torch.tensor(metadata)


        #load labels 
        label = torch.tensor(
            row["label"],
            dtype=torch.long
        )

        #the tensor sample return 
        return image, metadata, label

# HELPER FUNCTIONS

def get_train_dataset(train_df):

    return PlantDiseaseDataset(
        dataframe=train_df,
        transform=train_transforms
    )

def get_val_dataset(val_df):

    return PlantDiseaseDataset(
        dataframe=val_df,
        transform=val_transforms
    )

print("========== DATASET CLASS READY ==========")
