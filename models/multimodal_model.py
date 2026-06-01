#here the fusion takes place basically, 
import torch 
import torch.nn as nn
from models.image_encoder import ImageEncoder
from models.metadata_encoder import MetadataEncoder
from configs.config import NUM_CLASSES

class MultimodalModel(nn.Module):
    def __init__(self):

        super(MultimodalModel, self).__init__()
        #image encoder
        self.image_encoder = ImageEncoder()
        #metadataencoder
        self.metadata_encoder = MetadataEncoder()


       #fusion dimensions
        fusion_dim = (
            self.image_encoder.output_dim +
            self.metadata_encoder.output_dim
        )

        #classifier comes to play 
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )


    #forward pass
    def forward(self, image, metadata):
        #image features
        image_features = self.image_encoder(image)
        #metadatafeatures
        metadata_features = self.metadata_encoder(metadata)


        #fudsion features
        fused_features = torch.cat( #concatenate the things.
            [image_features, metadata_features],
            dim=1
        )

        #final classsification 
        output = self.classifier(fused_features)

        return output


print("*******MULTIMODAL MODEL READY *******")
