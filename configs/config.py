import torch
import os

#project paths : - 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "color")
METADATA_CSV = os.path.join(DATA_DIR, "metadata.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ==========================================
# IMAGE SETTINGS
# ==========================================

IMAGE_SIZE = 224

IMAGE_CHANNELS = 3

# Training Settings : - 
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

#K-FOLD Settings : - 
NUM_FOLDS = 5

SHUFFLE_DATASET = True

RANDOM_SEED = 42

#Model settings :- 
NUM_CLASSES = 2

METADATA_INPUT_DIM = 7

# Device configurations : - 
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

else:
    DEVICE = torch.device("cpu")


# ==========================================
# NORMALIZATION VALUES
# (ImageNet pretrained normalization)
# ==========================================

IMAGE_MEAN = [0.485, 0.456, 0.406]

IMAGE_STD = [0.229, 0.224, 0.225]

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


print("========== CONFIG LOADED ==========")

print(f"Device: {DEVICE}")

print(f"Image Size: {IMAGE_SIZE}")

print(f"Batch Size: {BATCH_SIZE}")

print(f"Learning Rate: {LEARNING_RATE}")

print(f"Number of Folds: {NUM_FOLDS}")

print("===================================")