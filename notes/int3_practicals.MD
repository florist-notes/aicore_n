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
+ Optimizer: Stochastic Gradient Descent (SGD) [[GD](https://youtu.be/vMh0zPT0tLI?si=-N5ifoa8w7O4e4eQ), [SGD](https://youtu.be/vMh0zPT0tLI?si=jwl9O-9t3qGnyyJ5)].
+ Loss function: Cross-entropy [[Cross Entropy](https://youtu.be/6ArSys5qHAU?si=M9lWNnIWddw0Hm2m), [Cross Entropy Derivatives and Backpropagation](https://youtu.be/xBEh66V9gZo?si=B5EQlOdZw7dsT4I7)].
  
### Experimental Procedure:
+ Page cache cleared at the start of every experiment run to avoid cross-experiment effects. 
+ However, it's retained across epochs within a single training run.
+ Each experiment trains DNN models for 6 epochs ([Image Classification with Convolutional Neural Networks (CNNs)](https://youtu.be/HGwBXDKFk9I?si=ytQFbyxcFKXDX_5H)).
+ Results averaged over epochs 1–5 since epoch 0 incurs bootstrapping overheads.

### PSEUDOCODE:

```python
# Pseudocode for Training a LeNet Model on Grayscale CIFAR-10 Dataset

# Import necessary libraries and modules
Import torch
Import neural network module from torch
Import optimizers from torch
Import DataLoader from torch.utils.data
Import transforms from torchvision
Import CIFAR10 dataset from torchvision.datasets

# Step 1: Data Loading and Preprocessing
Define transformation pipeline:
    - Convert images to grayscale with 1 output channel
    - Resize images to 32x32 pixels
    - Convert images to tensors
    - Normalize images with mean 0.5 and standard deviation 0.5

Load CIFAR-10 training dataset with defined transformations
Set batch size to 16
Set number of workers to 4 for data loading
Create DataLoader for the dataset with batch size, shuffle, and number of workers

# Step 2: Model Definition (LeNet)
Define LeNet class inheriting from nn.Module:
    Initialize layers:
        - Convolutional layer 1: input channels = 1, output channels = 6, kernel size = 5
        - Convolutional layer 2: input channels = 6, output channels = 16, kernel size = 5
        - Fully connected layer 1: input features = 16 * 5 * 5, output features = 120
        - Fully connected layer 2: input features = 120, output features = 84
        - Fully connected layer 3: input features = 84, output features = 10 (for CIFAR-10 classes)
    Define forward pass:
        - Apply ReLU to conv1, then max pooling with kernel size 2 and stride 2
        - Apply ReLU to conv2, then max pooling with kernel size 2 and stride 2
        - Flatten the tensor
        - Apply ReLU to fc1
        - Apply ReLU to fc2
        - Output from fc3

Instantiate LeNet model
Move model to GPU if available

# Step 3: Loss Function and Optimizer
Define cross-entropy loss function
Define SGD optimizer with learning rate 0.01 and momentum 0.9

# Step 4: Iterative Training
Set number of epochs to 6

For each epoch:
    For each batch in DataLoader:
        Move data and targets to GPU if available
        Zero the gradients
        Forward pass: compute model outputs from data
        Compute loss using outputs and targets
        Backward pass: compute gradients
        Update model parameters using optimizer
        If batch index is a multiple of 100:
            Print current epoch, step, and loss

Print "Training finished."
```
## Sample Code Steps: [ [LeNet script from DREAM:Lab](https://github.com/dream-lab/edge-train-bench/blob/sigmetrics-2023/exp_scripts/client_lenet_mnist.py) ]
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Data Loading and Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
    transforms.Resize((32, 32)),                 # Resize images to 32x32
    transforms.ToTensor(),                       # Convert images to tensors
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean and std
])

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
batch_size = 16
num_workers = 4
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Step 2: Model Definition (LeNet)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 output classes for CIFAR-10

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet().to(device)

# Step 3: Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Step 4: Iterative Training
num_epochs = 6

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("Training finished.")
```

## Explanation:

### Data Loading and Preprocessing:

+ We define a series of transformations to be applied to each input image in the dataset using `torchvision.transforms.Compose`.
+ The CIFAR-10 dataset is loaded using `torchvision.datasets.CIFAR10`. We specify the root directory to save the dataset, set train=True for the training split, and apply the defined transformations.
+ DataLoader is used to efficiently load and preprocess data in mini-batches. We specify the batch size, shuffle the dataset, and set the number of worker processes for parallel data loading and preprocessing.
### Model Definition:

+ We define the LeNet model.
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


## Sample Code : [ [ResNet script from DREAM:Lab](https://github.com/dream-lab/edge-train-bench/blob/sigmetrics-2023/exp_scripts/client_resnet_cifar.py) ] ← monitoring and logging (csv, jtop, multiprocessing etc)
```python
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import jtop  # Import jtop module for monitoring

# Step 1: Data Loading and Preprocessing

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

## 🌸 Experiment Bash Script : [ [bash_script /](https://github.com/dream-lab/edge-train-bench/blob/sigmetrics-2023/exp_scripts/exp_script.sh) ]

#### To drop cache : ` $ sync && echo 3 | tee /proc/sys/vm/drop_caches `.

## `iostat` and `vmtouch` are used for additional monitoring and measurement purposes in the context of the experiment described. Here's how they are related:

## → iostat: ` $ sudo apt-get install sysstat`
+ iostat is used to monitor input/output (I/O) statistics for storage devices (#sysstat for iostat).
+ In the experiment, iostat can be used to measure read IOPS (Input/Output Operations Per Second) and bytes read per second (throughput).
+ These metrics provide insights into the storage device's performance and can be useful for understanding how data loading operations impact the overall training process.
+ Example command to measure read IOPS and throughput: ` $ iostat -d -k -x 1`
+ This command will display disk I/O statistics, including read IOPS and throughput, every 1 second.

## → vmtouch: ` $ sudo apt-get install vmtouch`
+ vmtouch is used to query and manipulate the Linux kernel's virtual memory (page cache) subsystem.
+ In the experiment, vmtouch can be used to measure the fraction of the dataset present in the Linux (in-memory) disk cache.
+ This metric indicates how much of the dataset is currently cached in memory, which can affect data loading performance.
+ Example command to measure the fraction of dataset in the disk cache: ` $ vmtouch -m /path/to/dataset
`.
+ This command will display information about the dataset's presence in the disk cache.

In summary, `iostat` is used to monitor storage I/O performance, while `vmtouch` is used to monitor the dataset's presence in the disk cache, providing insights into how storage operations may impact training performance. These tools complement the monitoring provided by `jtop` and help capture a more comprehensive view of system behavior during training.
