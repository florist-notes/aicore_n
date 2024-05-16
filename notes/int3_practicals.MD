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

# `Performance Metrics | Experimental Setup`:

### Monitoring Setup:
+ We'll use the `jtop` Python module to monitor CPU, GPU, and RAM utilization, as well as average and instantaneous power.
+ The tegrastats utility from NVIDIA is used internally by jtop.
+ Power measurements are from on-board sensors in the Jetson devices, capturing the power load from the module (not the carrier board and peripherals).
+ Sampling occurs approximately every 1 second, but may deviate by up to 200 ms due to monitoring harness delays.

### Energy Calculation:
+ Total energy for training in a duration 𝑇 is calculated as the sum of the instantaneous power measured at each time point 𝑡_𝑖, weighted by the duration between successive samples (𝑡_𝑖 − 𝑡_{𝑖−1}).
+ This calculation approximates the integral of power over time.

### IO Metrics:
+ Read IOPS (Input/Output Operations Per Second) and bytes read per second (throughput) are measured using `iostat`.

### Disk Cache Measurement:
+ The fraction of the dataset present in the Linux (in-memory) disk cache is measured using `vmtouch`.

### Fetch Stall and GPU Compute Time:
+ Fetch stall time is the time taken to fetch and pre-process data, not overlapping with GPU compute time. It's calculated as max((fetch time + pre-process time − GPU compute time), 0).
GPU compute time includes kernel launch time, forward and backward passes of training.
These times are measured using `torch.cuda.event` with the synchronize option for accurate time capture.

### Epoch-level Measurements:
+ Fetch stall and GPU compute times are summed over all mini-batches in an epoch to obtain their average time per epoch.
+ End-to-End (E2E) time per epoch includes fetch stall time, GPU compute time, and any framework overheads.

```python
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import jtop  # Import jtop module for monitoring

# Step 1: Data Loading and Preprocessing
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
batch_size = 16
num_workers = 4
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Step 2: Model Definition (Assuming VGG11 model as before)

# Step 3: Loss Function and Optimizer (Assuming as before)

# Step 4: Iterative Training
num_epochs = 6

for epoch in range(num_epochs):
    # Monitoring setup
    j = jtop.Jtop()
    j.start()

    # Initialize metrics
    total_energy = 0.0
    fetch_stall_time = 0.0
    gpu_compute_time = 0.0
    e2e_time = 0.0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.cuda(), targets.cuda()

        # Start timer for fetch stall time
        fetch_start = torch.cuda.Event(enable_timing=True)
        fetch_start.record()

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # End timer for fetch stall time and calculate
        fetch_end = torch.cuda.Event(enable_timing=True)
        fetch_end.record()
        torch.cuda.synchronize()
        fetch_stall_time += fetch_start.elapsed_time(fetch_end)

        # Measure GPU compute time
        gpu_compute_time += fetch_end.elapsed_time(fetch_start)

        # Update total energy
        power = j.measure()['power']
        total_energy += power * 1.0  # Sampling interval is approximately 1 second

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')

    # End monitoring
    j.stop()

    # Calculate end-to-end time
    e2e_time = fetch_stall_time + gpu_compute_time

    # Print metrics for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Total Energy: {total_energy:.4f} J, '
          f'Fetch Stall Time: {fetch_stall_time:.4f} ms, GPU Compute Time: {gpu_compute_time:.4f} ms, '
          f'End-to-End Time: {e2e_time:.4f} ms')

print("Training finished.")
```

### Explanation:

+ `Monitoring Setup`: We import the `jtop` module for monitoring CPU, GPU, and RAM utilization, as well as power.
+ `Epoch-level Metrics`: Inside the epoch loop, we initialize metrics such as total energy, fetch stall time, GPU compute time, and end-to-end time.
+ `Monitoring Loop`: Within the training loop, we start monitoring with `j.start()` and stop it with `j.stop()` after each epoch.
+ `Metrics Calculation`: We calculate total energy by summing the instantaneous power measurements over time, and fetch stall and GPU compute times using CUDA events.
+ `Printing Metrics`: At the end of each epoch, we print the calculated metrics including total energy, fetch stall time, GPU compute time, and end-to-end time.

## `iostat` and `vmtouch` are used for additional monitoring and measurement purposes in the context of the experiment described. Here's how they are related:

## iostat:
+ iostat is used to monitor input/output (I/O) statistics for storage devices.
+ In the experiment, iostat can be used to measure read IOPS (Input/Output Operations Per Second) and bytes read per second (throughput).
+ These metrics provide insights into the storage device's performance and can be useful for understanding how data loading operations impact the overall training process.
+ Example command to measure read IOPS and throughput: ` $ iostat -d -k -x 1`
+ This command will display disk I/O statistics, including read IOPS and throughput, every 1 second.

## vmtouch:
+ vmtouch is used to query and manipulate the Linux kernel's virtual memory (page cache) subsystem.
+ In the experiment, vmtouch can be used to measure the fraction of the dataset present in the Linux (in-memory) disk cache.
+ This metric indicates how much of the dataset is currently cached in memory, which can affect data loading performance.
+ Example command to measure the fraction of dataset in the disk cache: ` $ vmtouch -m /path/to/dataset
`.
+ This command will display information about the dataset's presence in the disk cache.

In summary, `iostat` is used to monitor storage I/O performance, while `vmtouch` is used to monitor the dataset's presence in the disk cache, providing insights into how storage operations may impact training performance. These tools complement the monitoring provided by `jtop` and help capture a more comprehensive view of system behavior during training.
