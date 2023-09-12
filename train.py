"""
Trains a PyTorch image classification model for covid detection using device-agnostic code.
"""

import os
import torch
from torchvision import transforms
from timeit import default_timer as timer
import data_setup, engine, model_builder, utils

# Setup Hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 1e-3

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

# Create transforms
# Write a transform for image
data_transform = transforms.Compose(
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
    transform=data_transform,
    batch_size=BATCH_SIZE,
)

# Create Model
model = model_builder.TinyVGG(
    input=3, hidden_units=HIDDEN_UNITS, output=len(class_names)
).to(device)

# Setup Loss Function and Optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Start the timer
start_time = timer()

# Start the training with help from engine.py
engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device,
)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO]Total Training Time: {end_time - start_time:.3f} seconds")

# Save the model
utils.save_model(model=model, target_dir="models", model_name="TinyVGGV1.pth")
