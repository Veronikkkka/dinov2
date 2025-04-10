import torch
import torch.nn as nn
from dinov2.eval.segmentation.models.decode_heads import BNHead  # Adjust this import based on your actual model structure

# Assuming DINO is your model (replace with the actual DINO model import)
class DINOWithSegmentation(nn.Module):
    def __init__(self, pretrained_dino_model, in_channels, out_channels, resize_factors=None):
        super(DINOWithSegmentation, self).__init__()
        self.dino_model = pretrained_dino_model  # Pretrained DINO model
        self.segmentation_head = BNHead(in_channels=in_channels, out_channels=out_channels, resize_factors=resize_factors)

    def forward(self, x):
        # Extract features from DINO model
        features = self.dino_model(x)  # Replace this with actual forward pass of DINO
        
        # Forward pass through the segmentation head
        segmentation_output = self.segmentation_head(features)
        
        return segmentation_output


pretrained_dino = torch.load("/home/paperspace/Documents/nika_space/dinov2/new_encoder/model_0005999.rank_0.pth") # Load your pretrained DINO model here
model = DINOWithSegmentation(pretrained_dino_model=pretrained_dino, in_channels=256, out_channels=21)  # 21 classes for segmentation

# Assume you have an image to process
input_image = torch.randn(1, 3, 256, 384)  # Example image tensor (batch_size, channels, height, width)

# Forward pass
output = model(input_image)
