# DNN TRAINING ON EDGE DEVICES & CHARACTERIZING PERFORMANCE - PRACTICALS ❀

## Experiment Setup:

### Power Mode and Cooling:

+ Default power mode: MAXN for AGX and Nano, 15W for NX.
+ DVFS (Dynamic Voltage and Frequency Scaling) turned off.
+ Fan speed set to maximum to prevent resource throttling due to overheating.
### Storage Configuration:

+ Training data stored on SSD for AGX and NX, on SD card for Nano.
+ If uniform storage media type is needed across all devices, HDD over USB used as it's present on all.
### Data Loading Optimization:

+ Prefetch factor in PyTorch DataLoader set to default value of 2.
+ Number of worker processes in DataLoader set to 4.
### Mini-batch Size and Training Setup:

+ Mini-batch size set to 16 images.
+ Learning rate: 0.01, Momentum: 0.9.
+ Optimizer: Stochastic Gradient Descent (SGD).
+ Loss function: Cross-entropy.
### Experimental Procedure:

+ Page cache cleared at the start of every experiment run to avoid cross-experiment effects. 
+ However, it's retained across epochs within a single training run.
+ Each experiment trains DNN models for 6 epochs (approximately 15 hours).
+ Results averaged over epochs 1–5 (90 minutes for each epoch) since epoch 0 incurs bootstrapping overheads.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torchvision.models as models

# Step 1: Data Loading and Preprocessing
# Define data transformations
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),         # Random crop with padding
    transforms.RandomHorizontalFlip(),           # Randomly flip images horizontally
    transforms.ToTensor(),                        # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with mean and std
])

# Load CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
batch_size = 16  # Mini-batch size
num_workers = 4  # Number of worker processes in DataLoader

# Create DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Step 2: Model Definition
# Define VGG11 model
model = models.vgg11(pretrained=False, num_classes=10)  # 10 output classes for CIFAR-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model.to(device)

# Step 3: Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD optimizer

# Step 4: Iterative Training
num_epochs = 6  # Train for 6 epochs
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0

print("Training finished.")

```
## Explanation:

### Data Loading and Preprocessing:

+ We define a series of transformations to be applied to each input image in the dataset using `torchvision.transforms.Compose`.
+ The CIFAR-10 dataset is loaded using `torchvision.datasets.CIFAR10`. We specify the root directory to save the dataset, set train=True for the training split, and apply the defined transformations.
+ DataLoader is used to efficiently load and preprocess data in mini-batches. We specify the batch size, shuffle the dataset, and set the number of worker processes for parallel data loading and preprocessing.
### Model Definition:

+ We define the VGG11 model using torchvision.models.vgg11. The pretrained argument is set to False to train the model from scratch, and num_classes is set to 10 for CIFAR-10.
+ We check for the availability of GPU using `torch.cuda.is_available()` and move the model to the GPU device if available.
### Loss Function and Optimizer:

+ We define the loss function as cross-entropy using `torch.nn.CrossEntropyLoss()`.
+ The optimizer is defined as SGD (Stochastic Gradient Descent) using `torch.optim.SGD`. We specify a learning rate of 0.01 and momentum of 0.9.
### Iterative Training:

+ We iterate over the dataset for a specified number of epochs using a nested loop.
+ Within each epoch, we set the model to training mode using `model.train()`.
+ We iterate over mini-batches of data loaded by the DataLoader. Each mini-batch consists of input data (data) and corresponding labels (targets).
+ We move the data and labels to the GPU device if available.
+ We perform the forward pass through the model, calculate the loss, perform backward pass, and update model parameters using the optimizer.
+ Running loss is accumulated and printed at regular intervals for monitoring training progress.
After completing all epochs, we print a message indicating that training has finished.

