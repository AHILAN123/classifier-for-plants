#K-folds validation set

import torch
from configs.config import DEVICE

def validate_one_epoch(
    model,
    dataloader,
    criterion
):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0


    with torch.no_grad():
        for images,metadata,labels in dataloader:

            images = images.to(DEVICE)
            metadata = metadata.to(DEVICE)
            labels = labels.to(DEVICE)
            #forward pass
            outputs = model(
                images,
                metadata
            )
            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = (
        running_loss /
        len(dataloader)
    )

    epoch_accuracy = (
        100.0 *
        correct /
        total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )
