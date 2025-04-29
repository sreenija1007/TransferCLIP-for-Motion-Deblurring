import clip
import torch
from torch import nn

class CLIPFeatureExtractor(nn.Module):
    def __init__(self, backbone='RN50', device='cpu', unfreeze_layer4=False):
        super().__init__()
        # Load CLIP’s ModifiedResNet visual encoder
        self.clip_model, _ = clip.load(backbone, device=device)
        # Freeze it
        for p in self.clip_model.visual.parameters():
            p.requires_grad = False

        # Layers we’ll hook
        self.taps = {
            'layer1': self.clip_model.visual.layer1,
            'layer2': self.clip_model.visual.layer2,
            'layer3': self.clip_model.visual.layer3,
            'layer4': self.clip_model.visual.layer4,
        }

        if unfreeze_layer4:
            for p in self.clip_model.visual.layer4.parameters():
                p.requires_grad = True
        self.unfreeze_layer4 = unfreeze_layer4
        self.features = {}
        for name, layer in self.taps.items():
            layer.register_forward_hook(self._hook(name))

    def get_trainable_parameters(self):
        trainable_parameters = []
        if self.unfreeze_layer4:
            for p in self.clip_model.visual.layer4.parameters():
                if p.requires_grad:
                    trainable_parameters.append(p)
        return trainable_parameters

    def reset_parameters(self):
        if self.unfreeze_layer4:
            for layer in self.clip_model.visual.layer4.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def _hook(self, name):
        def hook(module, inp, out):
            self.features[name] = out
        return hook

    def forward(self, x):        
        # Stem: conv1→bn1→relu1→conv2→bn2→relu2→conv3→bn3→relu3→avgpool
        x = self.clip_model.visual.conv1(x)
        x = self.clip_model.visual.bn1(x)
        x = self.clip_model.visual.relu1(x)
        x = self.clip_model.visual.conv2(x)
        x = self.clip_model.visual.bn2(x)        
        x = self.clip_model.visual.relu2(x)        
        x = self.clip_model.visual.conv3(x)        
        x = self.clip_model.visual.bn3(x)        
        x = self.clip_model.visual.relu3(x)        
        x = self.clip_model.visual.avgpool(x)

        # Four ResNet blocks (hooks capture their outputs)
        _ = self.clip_model.visual.layer1(x)
        _ = self.clip_model.visual.layer2(_)
        _ = self.clip_model.visual.layer3(_)
        _ = self.clip_model.visual.layer4(_)

        # Return a dict of {layer1:…, layer2:…, layer3:…, layer4:…}
        return self.features
