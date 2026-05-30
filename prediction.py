import os
import argparse
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from configs.config import (
    DEVICE,
    CHECKPOINT_DIR
)
from models.multimodal_model import MultimodalModel

transform = transforms.Compose([ # resizing the image before passing. 
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


#arguments 
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--N", type=float, required=True)
parser.add_argument("--P", type=float, required=True)
parser.add_argument("--K", type=float, required=True)
parser.add_argument("--temperature", type=float, required=True)
parser.add_argument("--humidity", type=float, required=True)
parser.add_argument("--ph", type=float, required=True)
parser.add_argument("--rainfall", type=float, required=True)
args = parser.parse_args()



#image oopen
image = Image.open(args.image).convert("RGB")
image = transform(image)
image = image.unsqueeze(0)
image = image.to(DEVICE)


#metadata load
metadata = torch.tensor(
    [[
        args.N,
        args.P,
        args.K,
        args.temperature,
        args.humidity,
        args.ph,
        args.rainfall
    ]],
    dtype=torch.float32
).to(DEVICE)

#run folds
models = []
for fold in range(1, 5):
    model = MultimodalModel()
    path = os.path.join(
        CHECKPOINT_DIR,
        f"fold_{fold}.pth"
    )
    model.load_state_dict(
        torch.load(
            path,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()
    models.append(model)

with torch.no_grad():
    ensemble_output = None
    for model in models:
        output = model(
            image,
            metadata
        )
        if ensemble_output is None:
            ensemble_output = output
        else:
            ensemble_output += output
    ensemble_output /= len(models)

    probabilities = torch.softmax(
        ensemble_output,
        dim=1
    )

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )

prediction = prediction.item()
confidence = confidence.item()

#labellingg
label_map = {
    0: "Peach Healthy",
    1: "Peach Bacterial Spot"
}

print("\n====================")
print("Prediction:", label_map[prediction])
print(f"Confidence: {confidence*100:.2f}%")
print("====================")