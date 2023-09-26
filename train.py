"""
Trains a PyTorch image classification model for covid detection using device-agnostic code.
"""

import os
import torch
import torchvision

from torch import nn
from torchvision import transforms
from timeit import default_timer as timer
from functions import (
    data_setup,
    engine,
    model_builder,
    utils,
    model_builder_effnet,
    helper_functions,
)

# Setup Hyperparameters
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 1e-3
PYTORCH_ENABLE_MPS_FALLBACK = 1

# Setup directories
train_dir = "data/xray_dataset_covid19/train"
test_dir = "data/xray_dataset_covid19/test"

# Setup device agnostic code
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Write a transform for image
vgg_train_transform = transforms.Compose(
    [
        # Resize our images to 64x64
        transforms.Resize(size=(64, 64)),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5),
        # Turns image into grayscale
        transforms.Grayscale(num_output_channels=3),
        # Trivial Augment
        transforms.TrivialAugmentWide(num_magnitude_bins=10),
        # Turn the image into a torch.Tensor
        transforms.ToTensor()
        # Permute the channel height and width
    ]
)

vgg_test_transform = transforms.Compose(
    [
        # Resize our images to 64x64
        transforms.Resize(size=(64, 64)),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5),
        # Turns image into grayscale
        transforms.Grayscale(num_output_channels=3),
        # Turn the image into a torch.Tensor
        transforms.ToTensor()
        # Permute the channel height and width
    ]
)

# Create DataLoader's and get class_names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transform=vgg_train_transform,
    test_transform=vgg_test_transform,
    batch_size=BATCH_SIZE,
)

# Set the pretrained weights of Efficient Net B0
pretrained_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

# Make sure to transform the data exactly how the model was previously trained on
auto_test_transform = pretrained_weights.transforms()

auto_train_transform = transforms.Compose(
    [
        transforms.TrivialAugmentWide(num_magnitude_bins=10),
        auto_test_transform,
    ]
)

# Create DataLoader's and get class_names
(
    train_dataloader_auto,
    test_dataloader_auto,
    class_names,
) = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transform=auto_train_transform,
    test_transform=auto_test_transform,
    batch_size=BATCH_SIZE,
)

# Start the timer
start_time = timer()

# Set seeds
helper_functions.set_seeds(seed=42)

# Keep track of experiment numberes
exp_number = 0

# Set Number of Epochs
NUM_EPOCHS = [5, 10]

# Create Models list (need to create a new model for each experiment)
models = ["EffNetB0", "TinyVGG"]

# Create DataLoaders Dictionary
train_dataloaders = {
    "data_auto": train_dataloader_auto,
    "data_manual": train_dataloader,
}

test_dataloaders = {
    "data_test_auto": test_dataloader_auto,
    "data_test_manual": test_dataloader,
}


# Loop through the epochs
for epochs in NUM_EPOCHS:
    # Loop through each name and create a new model instance
    for model_name in models:
        # Print out info
        exp_number += 1
        print(f"[INFO] Experiment Number: {exp_number}")
        print(f"[INFO] Model: {model_name}")
        print(f"[INFO] Number of Epochs: {epochs}")
        # Select and create the model
        if model_name == "EffNetB0":
            train_dataloader = train_dataloaders["data_auto"]
            test_dataloader = test_dataloaders["data_test_auto"]
            dataloader_name = list(train_dataloaders.keys())[
                list(train_dataloaders.values()).index(train_dataloader)
            ]
            print(f"[INFO] DataLoader: {dataloader_name}")
            model, transforms = model_builder_effnet.create_effnet(
                pretrained_weights=pretrained_weights,
                model=torchvision.models.efficientnet_b0,
                in_features=1280,
                dropout=0.2,
                out_features=len(class_names),
            )
        else:
            train_dataloader = train_dataloaders["data_manual"]
            test_dataloader = test_dataloaders["data_test_manual"]
            dataloader_name = list(train_dataloaders.keys())[
                list(train_dataloaders.values()).index(train_dataloader)
            ]
            print(f"[INFO] DataLoader: {dataloader_name}")
            # Create TinyVGG Model
            model = model_builder.TinyVGG(
                input=3, hidden_units=HIDDEN_UNITS, output=len(class_names)
            ).to(device)
        # Create a new loss and optimizer for every model
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
        # Train target model with target dataloader and track experiments
        engine.train(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=epochs,
            device=device,
        )
        # Save model to a file so we can import it later if need be
        save_filepath = f"{model_name}_{dataloader_name}_{epochs}_epochs.pth"
        utils.save_model(model=model, target_dir="models", model_name=save_filepath)
        print("-" * 50 + "\n")

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO]Total Training Time: {end_time - start_time:.3f} seconds")
