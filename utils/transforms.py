from torchvision import transforms
from configs.config import (IMAGE_SIZE,IMAGE_MEAN,IMAGE_STD)

#train(data augmentation takes place here)
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),#resize
    transforms.ToTensor(), #tensor convert
    transforms.Normalize( #224*224 standard size convert
        mean=IMAGE_MEAN,
        std=IMAGE_STD
    )
])

#validation for this
val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGE_MEAN,
        std=IMAGE_STD
    )
])
print("TRANSFORMS LOADED")
