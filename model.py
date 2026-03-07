import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path):

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 7)

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    return model