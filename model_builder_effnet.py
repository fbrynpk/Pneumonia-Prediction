# Functionize the model creation process

import torch
import torchvision
import helper_functions

from torch import nn

# Setup device agnostic code
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# Create an EffNetB0 features extractor
def create_effnet(
    pretrained_weights: torchvision.models.Weights,
    model: torchvision.models,
    in_features: int,
    dropout: int,
    out_features: int,
):
    # Get the weights and setup the model
    model = model(weights=pretrained_weights).to(device)

    # Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Change the classifier head
    helper_functions.set_seeds()
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features, out_features=out_features),
    ).to(device)

    return model
