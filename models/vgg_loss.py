import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.feature_layers = nn.Sequential(
            vgg16.features[:4],  # relu1_2
            vgg16.features[4:9], # relu2_2
            vgg16.features[9:16] # relu3_3
        )
        # Freeze the VGG model
        for param in self.feature_layers.parameters():
            param.requires_grad = False

    def forward(self, img1, img2):
        # Extract features
        features1 = self.feature_layers(img1)
        features2 = self.feature_layers(img2)

        # Compute loss for each layer
        loss = 0
        for feat1, feat2 in zip(features1, features2):
            loss += F.l1_loss(feat1, feat2)

        return loss