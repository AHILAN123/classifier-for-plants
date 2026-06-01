#would be ensembling the 4 folds and as well as calculating the f1 precision recalls
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from configs.config import (
    DEVICE,
    BATCH_SIZE,
    CHECKPOINT_DIR
)
from models.multimodal_model import MultimodalModel
from utils.dataset import get_val_dataset

#load data
test_df = pd.read_csv("data/test.csv")
test_dataset = get_val_dataset(test_df)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Test Samples: {len(test_df)}")


#load fold models created in checkpoint/
models = []
for fold in range(1, 3):

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"fold_{fold}.pth"
    )

    model = MultimodalModel()

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    models.append(model)

    print(f"Loaded: {checkpoint_path}")

print(f"\nTotal Models Loaded: {len(models)}")


#ensemble
all_labels = []
all_predictions = []

print("\nRunning Ensemble Evaluation...\n")

with torch.no_grad():
    for images,metadata,labels in test_loader:
        images = images.to(DEVICE)
        metadata = metadata.to(DEVICE)
        labels = labels.to(DEVICE)
        ensemble_outputs = None

        for model in models:

            outputs = model(
                images,
                metadata
            )
            if ensemble_outputs is None:
                ensemble_outputs = outputs
            else:
                ensemble_outputs += outputs
        ensemble_outputs /= len(models) #average out
        predictions = torch.argmax(
            ensemble_outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )



accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

precision = precision_score(
    all_labels,
    all_predictions,
    average = "weighted"
)

recall = recall_score(
    all_labels,
    all_predictions,
    average = "weighted"
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average = "weighted"
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("_______ENSEMBLE TEST RESULTS_______")

print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels = [0,1,2,3],
        target_names=[
            "Apple scab",
            "rot",
            "Cedar apple rust",
            "healthy",
        ]
    )
)
print("--"*10)
