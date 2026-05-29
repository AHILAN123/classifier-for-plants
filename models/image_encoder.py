import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    resnet34,
    ResNet34_Weights
)

class ImageEncoder(nn.Module):
    def __init__(self):

        super(ImageEncoder, self).__init__()
        #pretrained resnet34
        resnet = resnet34(
        weights=None
        )

        #remove finall classfication layer i just want features the network would learn them 
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1]
        )
        #output dimension
        self.output_dim = 512


    #forward pass
    def forward(self, x):
        #extracting the feature
        features = self.feature_extractor(x)
        #flattening the features
        features = torch.flatten(features, 1)
        return features


print("IMAGE ENCODER READY ")
