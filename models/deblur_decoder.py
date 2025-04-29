import torch
from torch import nn

class DeblurDecoder(nn.Module):
    def __init__(self, clip_ch):
        """
        clip_ch: dict with channel dimensions of CLIP's ResNet-50 layers.
        """
        super().__init__()
        self.clip_ch = clip_ch
        
        # Define the decoder layers
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(clip_ch['layer4'], 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),


            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),           
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),           
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),           
            nn.ReLU(inplace=True),
            
            # Final convolution to predict the residual
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, blurry, feats):
        """
        Forward pass of the residual decoder.
        """
        x4 = feats['layer4']
        residual = self.decoder(x4)
        
        return residual

    def get_trainable_parameters(self):
        return self.parameters()
