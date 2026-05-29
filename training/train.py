import torch
from configs.config import DEVICE

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer
):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    #loop batches
    for images,metadata,labels in dataloader:

        #move to gpu
        images = images.to(DEVICE)
        metadata = metadata.to(DEVICE)
        labels = labels.to(DEVICE)

        #change gradients
        optimizer.zero_grad()

        #forward pass
        outputs = model(
            images,
            metadata
        )

        #loss
        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

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