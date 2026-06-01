import torch
import torch.nn as nn

#FNN model
class MetadataEncoder(nn.Module):

    def __init__(self):
        super(MetadataEncoder, self).__init__()
        
        #fully connected network 
        self.network = nn.Sequential(
            # Input Layer
            nn.Linear(7, 64),
            nn.ReLU(),
            # Hidden Layer
            nn.Linear(64, 128),
            nn.ReLU()
        )
       #output embedding dimension 
        self.output_dim = 128

   #forward pass
    def forward(self, x):
        embedding = self.network(x)
        return embedding

print("*******METADATA ENCODER READY********")
