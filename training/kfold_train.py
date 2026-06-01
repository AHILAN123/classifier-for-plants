import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from configs.config import (
    TRAIN_CSV,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    NUM_FOLDS,
    RANDOM_SEED,
    CHECKPOINT_DIR,
    DEVICE
)
from utils.dataset import (
    get_train_dataset,
    get_val_dataset
)
from models.multimodal_model import MultimodalModel
from training.train import train_one_epoch
from training.validate import validate_one_epoch

#data loading
df = pd.read_csv(TRAIN_CSV)

print(f"Dataset Size: {len(df)}")


#kfold setup 
kfold = StratifiedKFold(
    n_splits=NUM_FOLDS,
    shuffle=True,
    random_state=RANDOM_SEED
)


#kfold train
for fold, (train_idx, val_idx) in enumerate(
    kfold.split(
        df,
        df["label"]
    )
):
    print("\n" + "=" * 50)
    print(f"FOLD {fold + 1}/{NUM_FOLDS}")
    print("=" * 50)

    
    #data split
    train_df = df.iloc[train_idx].reset_index(drop=True)

    val_df = df.iloc[val_idx].reset_index(drop=True)

    print(f"Train Samples: {len(train_df)}")
    print(f"Val Samples:   {len(val_df)}")

    #datasets
    train_dataset = get_train_dataset(
        train_df
    )

    val_dataset = get_val_dataset(
        val_df
    )

    #dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    #model
    model = MultimodalModel()

    model = model.to(DEVICE)

    #loss func
    criterion = nn.CrossEntropyLoss()

    #optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------
    # BEST VALIDATION ACCURACY
    # --------------------------------------

    best_val_acc = 0.0


    # ======================================
    # EPOCH LOOP
    # ======================================

    for epoch in range(NUM_EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )


        # ----------------------------------
        # SAVE BEST MODEL
        # ----------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"fold_{fold+1}.pth"
            )

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                f"Saved Best Model -> {checkpoint_path}"
            )

    print(
        f"Best Validation Accuracy: "
        f"{best_val_acc:.2f}%"
    )


print("\nTraining Complete!")
