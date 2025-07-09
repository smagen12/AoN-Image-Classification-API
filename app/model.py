import torch
import torch.nn as nn
from torchvision import models

class AnimeCartoonResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(AnimeCartoonResNet50, self).__init__()
        self.model = models.resnet50(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model(weights_path: str) -> AnimeCartoonResNet50:
    model = AnimeCartoonResNet50()
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model
