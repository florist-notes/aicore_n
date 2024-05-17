# CODE BREAKDOWN | [ [dream-lab/client_resnet_cifar.py](https://github.com/dream-lab/edge-train-bench/blob/sigmetrics-2023/exp_scripts/client_resnet_cifar.py)  ] 🌸

### Importing Modules from the Standard Library:

```python
from collections import OrderedDict
import sys
import csv
import time
import os
import collections
import multiprocessing
import logging
```
+ `OrderedDict`: This is a dictionary subclass from the collections module that maintains the order in which keys are inserted.
+ `sys`: This module provides access to some variables used or maintained by the Python interpreter and to functions that interact strongly with the interpreter.
+ `csv`: This module implements classes to read and write tabular data in CSV format.
time: This module provides various time-related functions.
+ `os`: This module provides a way of using operating system-dependent functionality like reading or writing to the file system.
+ `collections`: This module implements specialized container datatypes providing alternatives to Python’s general-purpose built-in containers like dict, list, set, and tuple.
+ `multiprocessing`: This module supports the spawning of processes using an API similar to the threading module.
+ `logging`: This module provides a flexible framework for emitting log messages from Python programs.

### Importing Modules from External Libraries:

```python
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from jtop import jtop
import pandas as pd
```

+ `flwr (Flower)`: A library for federated learning that simplifies the orchestration of federated learning experiments.
+ `torch`: The core library of PyTorch, an open-source machine learning library based on the Torch library.
+ `torch.nn`: A submodule of PyTorch that provides various classes and functions to build neural networks.
+ `torch.nn.functional`: A submodule that contains functions used for building neural network layers (e.g., activation functions).
+ `torchvision.transforms`: This module provides common image transformations for data preprocessing.
+ `torch.utils.data.DataLoader`: A PyTorch utility that provides an iterable over a dataset with support for batching, shuffling, and multiprocess data loading.
+ `numpy`: A fundamental package for scientific computing in Python.
+ `jtop`: A library to interface with the Jetson board to monitor hardware metrics.
+ `pandas`: A data analysis and manipulation library.

### Custom Import : [ [/landmark_dataset.py](https://github.com/dream-lab/edge-train-bench/blob/sigmetrics-2023/exp_scripts/landmark_dataset.py) ]

```python
from landmark_dataset import Landmarks
```

### Re-importing from torchvision:

```python
from torchvision import datasets, transforms
```
+ `torchvision.datasets`: This module provides many pre-loaded datasets as well as tools for loading custom datasets.
+ `torchvision.transforms`: This is re-imported here and provides the same functionality as mentioned earlier.

### Device Configuration:

```python
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```
This line sets up the device to be used for PyTorch operations. If a CUDA-compatible GPU is available, it sets the device to `cuda:0` (the first GPU). Otherwise, it defaults to the CPU.

### Logging Formatter Configuration:

```python
formatter = logging.Formatter('%(message)s')
```
This line sets up a logging formatter that will format log messages as simple messages without additional metadata (like timestamps or log levels).

###  Setup Logger Function:

```python
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
```
This function initializes and returns a logger that writes to a specified log file.

`name`: The name of the logger.
`log_file`: The file to which logs will be written.
`level`: The logging level (e.g., INFO, DEBUG).

### Start Logging Function:

```python
def start_logging(filename, iostats_filename, memstats_filename, swapstats_filename, cpufreqstats_filename, gpufreqstats_filename, emcstats_filename, ramstats_filename, external_device, reference_time):
    print("jtop logging started")
```
This function starts logging system statistics using the `jtop` library. The function parameters specify the filenames for various logs and the external device to monitor.

#### 🌸 DataFrames Initialization:

```python
    output = pd.DataFrame()
    output_cpufreqstats = pd.DataFrame()
    output_gpufreqstats = pd.DataFrame()
    output_emcstats = pd.DataFrame()
    output_ramstats = pd.DataFrame()
```
These lines initialize empty pandas DataFrames to store the statistics.

#### 🌸 Initial Logging:
```python
    with jtop() as jetson:
        tegrastats_entry = jetson.stats
        tegrastats_entry['log_time'] = str(time.time() - reference_time)
        output = output.append(tegrastats_entry, ignore_index=True)
        output.to_csv(filename, index=False)

        cpufreqstats_entry = jetson.cpu
        cpufreqstats_entry['log_time'] = str(time.time() - reference_time)
        output_cpufreqstats = output_cpufreqstats.append(cpufreqstats_entry, ignore_index=True)
        output_cpufreqstats.to_csv(cpufreqstats_filename, index=False)

        gpufreqstats_entry = jetson.gpu
        gpufreqstats_entry['log_time'] = str(time.time() - reference_time)
        output_gpufreqstats = output_gpufreqstats.append(gpufreqstats_entry, ignore_index=True)
        output_gpufreqstats.to_csv(gpufreqstats_filename, index=False)

        emcstats_entry = jetson.emc
        emcstats_entry['log_time'] = str(time.time() - reference_time)
        output_emcstats = output_emcstats.append(emcstats_entry, ignore_index=True)
        output_emcstats.to_csv(emcstats_filename, index=False)

        ramstats_entry = jetson.ram
        ramstats_entry['log_time'] = str(time.time() - reference_time)
        output_ramstats = output_ramstats.append(ramstats_entry, ignore_index=True)
        output_ramstats.to_csv(ramstats_filename, index=False)
```
This block logs initial statistics for various metrics (CPU, GPU, EMC, RAM) and writes them to the corresponding CSV files.
#### 🌸 Continuous Logging:
```python
    with jtop() as jetson:
        while jetson.ok():
            tegrastats_entry = jetson.stats
            tegrastats_entry['log_time'] = str(time.time() - reference_time)
            output = output.append(tegrastats_entry, ignore_index=True)

            cpufreqstats_entry = jetson.cpu
            cpufreqstats_entry['log_time'] = str(time.time() - reference_time)
            output_cpufreqstats = output_cpufreqstats.append(cpufreqstats_entry, ignore_index=True)

            gpufreqstats_entry = jetson.gpu
            gpufreqstats_entry['log_time'] = str(time.time() - reference_time)
            output_gpufreqstats = output_gpufreqstats.append(gpufreqstats_entry, ignore_index=True)

            emcstats_entry = jetson.emc
            emcstats_entry['log_time'] = str(time.time() - reference_time)
            output_emcstats = output_emcstats.append(emcstats_entry, ignore_index=True)

            ramstats_entry = jetson.ram
            ramstats_entry['log_time'] = str(time.time() - reference_time)
            output_ramstats = output_ramstats.append(ramstats_entry, ignore_index=True)
```
This block continuously collects statistics in a loop until the `jetson.ok()` method returns `False`.
#### 🌸 IO Statistics Logging:
```python
            io_output = os.popen("iostat -xy 1 1 -d " + external_device + " | awk 'NR>3{ for (x=2; x<=16; x++) {  printf\"%s \", $x}}' | sed 's/ /,/g'| sed 's/,*$//g'")
            io_output = io_output.read() + "," + str(time.time() - reference_time)
            iostats_filename.info(io_output)
```
This block logs I/O statistics using the iostat command and writes the output to the specified log file.
#### 🌸 Memory Statistics Logging:
```python
            mem_output = os.popen("free -mh | awk 'NR==2{for (x=2;x<=7;x++){printf\"%s \", $x}}' | sed 's/ /,/g'| sed 's/,*$//g'")
            mem_output = mem_output.read() + "," + str(time.time() - reference_time)
            memstats_filename.info(mem_output)
```
This block logs memory statistics using the `free -mh` command and writes the output to the specified log file.
#### 🌸 Swap Statistics Logging:
```python
            swap_output = os.popen("free -mh | awk 'NR==3{for (x=2;x<=4;x++){printf\"%s \", $x}}' | sed 's/ /,/g'| sed 's/,*$//g'")
            swap_output = swap_output.read() + "," + str(time.time() - reference_time)
            swapstats_filename.info(swap_output)
```
This block logs swap memory statistics using the `free -mh` command and writes the output to the specified log file.
#### 🌸 Periodic CSV Writing:
```python
            output.to_csv(filename, index=False, mode='a', header=False)
            output = pd.DataFrame()

            output_cpufreqstats.to_csv(cpufreqstats_filename, index=False, mode='a', header=False)
            output_cpufreqstats = pd.DataFrame()

            output_gpufreqstats.to_csv(gpufreqstats_filename, index=False, mode='a', header=False)
            output_gpufreqstats = pd.DataFrame()

            output_emcstats.to_csv(emcstats_filename, index=False, mode='a', header=False)
            output_emcstats = pd.DataFrame()

            output_ramstats.to_csv(ramstats_filename, index=False, mode='a', header=False)
            output_ramstats = pd.DataFrame()
```
This block writes the accumulated DataFrames to the corresponding CSV files periodically, then clears the DataFrames for the next batch of data.

### Cutout Class :

The `Cutout` class implements the Cutout data augmentation technique, which involves randomly masking out square regions of an image during training.

```python
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
```

`init`: Initializes the Cutout class with the length of the square to be cut out.

`call`: When an instance of Cutout is called with an image:

+ Creates a mask of ones with the same height and width as the image.
+ Randomly selects a position in the image.
+ Calculates the boundaries of the square to be cut out.
+ Sets the corresponding region in the mask to zero.
+ Applies the mask to the image, effectively zeroing out the square region.

### Data Transforms Function :
The `_data_transforms_landmarks` function defines data transformations for training and validation datasets.

```python
def _data_transforms_landmarks():
    IMAGENET_MEAN = [0.5, 0.5, 0.5]
    IMAGENET_STD = [0.5, 0.5, 0.5]

    image_size = 224
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, valid_transform
```
+ `train_transform`: Composes several transformations for the training dataset, including random resized cropping, horizontal flipping, tensor conversion, normalization, and Cutout.
+ `valid_transform`: Composes transformations for the validation dataset, including center cropping, tensor conversion, and normalization.

### Read CSV Function :
The `_read_csv` function reads a CSV file and returns its contents as a list of dictionaries.

```python
def _read_csv(path):
    with open(path, 'r') as f:
        return list(csv.DictReader(f))
```
+ `path`: Path to the CSV file.
+ `Returns`: A list of dictionaries where each row in the CSV file is represented as a dictionary with column names as keys.

### Get Mapping Per User Function
The `get_mapping_per_user` function processes a CSV file to create mappings for user-specific data.
```python
def get_mapping_per_user(fn):
    mapping_table = _read_csv(fn)
    expected_cols = ['user_id', 'image_id', 'class']
    if not all(col in mapping_table[0].keys() for col in expected_cols):
        logger.error('%s has wrong format.', mapping_file)
        raise ValueError(
            'The mapping file must contain user_id, image_id and class columns. '
            'The existing columns are %s' % ','.join(mapping_table[0].keys()))

    data_local_num_dict = dict()
    mapping_per_user = collections.defaultdict(list)
    data_files = []
    net_dataidx_map = {}
    sum_temp = 0

    for row in mapping_table:
        user_id = row['user_id']
        mapping_per_user[user_id].append(row)
    for user_id, data in mapping_per_user.items():
        num_local = len(mapping_per_user[user_id])
        net_dataidx_map[int(user_id)] = (sum_temp, sum_temp + num_local)
        data_local_num_dict[int(user_id)] = num_local
        sum_temp += num_local
        data_files += mapping_per_user[user_id]
    assert sum_temp == len(data_files)

    return data_files, data_local_num_dict, net_dataidx_map
```
+ `mapping_table`: Reads the CSV file and stores it as a list of dictionaries.
+ `Checks for required columns`: Ensures the CSV file contains `user_id`, `image_id`, and `class` columns.
+ `Initializes dictionaries`: For storing user-specific data.
+ `Populates dictionaries`: Maps each `user_id` to its respective data entries.
+ `Returns`: A list of all data entries, a dictionary mapping user IDs to the number of local data points, and a dictionary mapping user IDs to index ranges of their data.

### Get Dataloader Function :
The `get_dataloader_Landmarks` function creates data loaders for the training and test datasets.
```python
def get_dataloader_Landmarks(datadir, train_files, test_files, train_bs, test_bs, dataidxs=None, num_w=0, pf=2):
    dl_obj = Landmarks
    transform_train, transform_test = _data_transforms_landmarks()

    train_ds = dl_obj(datadir, train_files, 0, dataidxs=None, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, test_files, 1, dataidxs=None, train=False, transform=transform_test, download=True)

    train_dl = DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False, num_workers=num_w, prefetch_factor=pf)
    test_dl = DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False, num_workers=num_w, prefetch_factor=pf)

    return train_dl, test_dl
```
+ `datadir`: Directory containing the data.
+ `train_files`: List of training files.
+ `test_files`: List of test files.
+ `train_bs`: Batch size for training.
+ `test_bs`: Batch size for testing.
+ `Transforms`: Applies data transformations for training and testing.
+ `DataLoader`: Creates data loaders for the training and test datasets.

### Load Partition Data Function :
The `load_partition_data_landmarks` function loads and partitions data for federated learning.
```python
def load_partition_data_landmarks(dataset, data_dir, fed_train_map_file, fed_test_map_file, partition_method=None, partition_alpha=None, client_number=1, batch_size=16, num_w=0, pf=2):
    train_files, data_local_num_dict, net_dataidx_map = get_mapping_per_user(fed_train_map_file)
    test_files = _read_csv(fed_test_map_file)

    class_num = len(np.unique([item['class'] for item in train_files]))
    train_data_num = len(train_files)

    train_data_global, test_data_global = get_dataloader_Landmarks(data_dir, train_files, test_files, batch_size, batch_size, None, num_w, pf)
    test_data_num = len(test_files)
    return train_data_global, test_data_global, class_num
```

+ `Parameters`: Includes dataset, data directory, file paths, partition method, alpha, client number, batch size, number of workers, and prefetch factor.
+ `Loads Data`: Reads the federated train and test map files.
+ `Class Count`: Determines the number of unique classes.
+ `Creates Data Loaders`: For global training and testing data.
+ `Returns`: The global train and test data loaders, and the number of classes.

The `get_mnist_dataset` and `get_cifar10_dataset` functions are designed to load and preprocess the MNIST and CIFAR-10 datasets, respectively. These functions create data loaders for training and testing, and normalize the data. Below is an explanation of each function:

### `get_mnist_dataset Function` :
This function sets up the MNIST dataset for training and testing.

```python
def get_mnist_dataset(data_path, bs, num_w, pf):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
    dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    
    class_names = [str(c) for c in range(num_classes)]
    
    trainloader = torch.utils.data.DataLoader(
        dst_train, batch_size=bs, shuffle=True, drop_last=False, num_workers=num_w, prefetch_factor=pf
    )
    
    testloader = torch.utils.data.DataLoader(
        dst_test, batch_size=bs, shuffle=False, drop_last=False, num_workers=num_w, prefetch_factor=pf
    )
    
    return trainloader, testloader, num_classes, channel
```

#### + Parameters:
- `data_path`: Path to store the downloaded data.
- `bs`: Batch size for data loaders.
- `num_w`: Number of workers for data loading.
- `pf`: Prefetch factor for data loading.
#### + Setup:
- `channel`: Number of image channels (1 for grayscale).
- `im_size`: Image size (28x28 for MNIST).
- `num_classes`: Number of classes (10 for MNIST).
- `mean and std`: Mean and standard deviation for normalization.
#### + Transforms:
- `transforms.Compose`: Converts images to tensors and normalizes them.
#### + Datasets:
- `dst_train`: Training dataset with transformations.
- `dst_test`: Testing dataset with transformations.
#### + Data Loaders:
- `trainloader`: Data loader for the training dataset.
- `testloader`: Data loader for the testing dataset.
- `Returns`: Train loader, test loader, number of classes, and number of channels.

### `get_cifar10_dataset` Function
This function sets up the CIFAR-10 dataset for training and testing.

```python
def get_cifar10_dataset(data_path, bs, num_w, pf):
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
    dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    
    class_names = dst_train.classes
    
    trainloader = torch.utils.data.DataLoader(
        dst_train, batch_size=bs, shuffle=True, drop_last=False, num_workers=num_w, prefetch_factor=pf
    )
    
    testloader = torch.utils.data.DataLoader(
        dst_test, batch_size=bs, shuffle=False, drop_last=False, num_workers=num_w, prefetch_factor=pf
    )
    
    return trainloader, testloader, num_classes, channel
```

#### + Parameters:
- `data_path`: Path to store the downloaded data.
- `bs`: Batch size for data loaders.
- `num_w`: Number of workers for data loading.
- `pf`: Prefetch factor for data loading.
#### + Setup:
- `channel`: Number of image channels (3 for RGB).
- `im_size`: Image size (32x32 for CIFAR-10).
- `num_classes`: Number of classes (10 for CIFAR-10).
- `mean and std`: Mean and standard deviation for normalization.
#### + Transforms:
- `transforms.Compose`: Converts images to tensors and normalizes them.
#### + Datasets:
- `dst_train`: Training dataset with transformations.
- `dst_test`: Testing dataset with transformations.
#### + Data Loaders:
- `trainloader`: Data loader for the training dataset.
- `testloader`: Data loader for the testing dataset.
- `Returns`: Train loader, test loader, number of classes, and number of channels.


### `train` Function :

```python
def train(net, trainloader,testloader, epoch_fname, fetch_fname, compute_fname, vmtouch_fname, dataset_outer_folder, reference_time, epochs):
    print('Training')
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()
    vmtouch_dir = os.path.join(dataset_outer_folder,'cifar10_data/')
    start1 = torch.cuda.Event(enable_timing=True)
    end1 = torch.cuda.Event(enable_timing=True)
    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start3 = torch.cuda.Event(enable_timing=True)
    end3 = torch.cuda.Event(enable_timing=True)
    #start_time = time.time()
    epoch_count = 0
    """ vmtouch logging """
    '''
    with open(vmtouch_fname, 'a', newline='') as file: # DEPRECATED
        vmtouch_output = os.popen(r"vmtouch -f " + vmtouch_dir + r" | sed 's/^.*://' | sed -z 's/\n/,/g' | sed 's/\s\+/,/g' | sed 's/,\{2,\}/,/g' | sed -e 's/^.\(.*\).$/\1/'")
        vmtouch_output = vmtouch_output.read()+","+str(time.time() - reference_time)
        writer = csv.writer(file)
        writer.writerow(vmtouch_output.split(","))
    '''
    vmtouch_output = os.popen(r"vmtouch -f " + vmtouch_dir + r" | sed 's/^.*://' | sed -z 's/\n/,/g' | sed 's/\s\+/,/g' | sed 's/,\{2,\}/,/g' | sed -e 's/^.\(.*\).$/\1/'")
    vmtouch_output = vmtouch_output.read()+","+str(time.time() - reference_time)
    vmtouch_fname.info(vmtouch_output)
    # with open(vmtouchlogtime_fname, 'a', newline='') as file:
    #     writer = csv.writer(file, delimiter=' ',escapechar=' ', quoting=csv.QUOTE_NONE)
    #     writer.writerow([str(time.time() - reference_time)])
    os.system('sudo ./drop_caches.sh')
    for _ in range(epochs):
        #start1.record()
        """ vmtouch logging """
        '''
        with open(vmtouch_fname, 'a', newline='') as file: # DEPRECATED
            vmtouch_output = os.popen(r"vmtouch -f " + vmtouch_dir + r" | sed 's/^.*://' | sed -z 's/\n/,/g' | sed 's/\s\+/,/g' | sed 's/,\{2,\}/,/g' | sed -e 's/^.\(.*\).$/\1/'")
            vmtouch_output = vmtouch_output.read()+","+str(time.time() - reference_time)
            writer = csv.writer(file)
            writer.writerow(vmtouch_output.split(","))
        '''
        vmtouch_output = os.popen(r"vmtouch -f " + vmtouch_dir + r" | sed 's/^.*://' | sed -z 's/\n/,/g' | sed 's/\s\+/,/g' | sed 's/,\{2,\}/,/g' | sed -e 's/^.\(.*\).$/\1/'")
        vmtouch_output = vmtouch_output.read()+","+str(time.time() - reference_time)
        vmtouch_fname.info(vmtouch_output)
        # with open(vmtouchlogtime_fname, 'a', newline='') as file:
        #     writer = csv.writer(file, delimiter=' ',escapechar=' ', quoting=csv.QUOTE_NONE)
        #     writer.writerow([str(time.time() - reference_time)])
        print('Epoch: ' + str(_) + ' Begins')
        # start_time = time.time() # DEPRECATED
        start1.record() # e2e epoch time starts
        # start2_time = time.time() # DEPRECATED
        start2.record() # fetch time per batch starts
        for batch_idx, (images, labels) in enumerate(trainloader):
            print('Batch index = '+ str(batch_idx))
            images, labels = images.to(DEVICE), labels.to(DEVICE) # won't be part of fetch + preprocess
            # end2_time = time.time() - start2_time # DEPRECATED
            end2.record() # fetch time per batch ends
            start3.record() # compute time per batch starts
            # start_time1 = time.time() # DEPRECATED
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # end_time1  = time.time() - start_time1 # DEPRECATED
            end3.record() # compute time per batch ends
            torch.cuda.synchronize()
            '''
            with open(fetch_fname, 'a', newline='') as file: # DEPRECATED
                writer = csv.writer(file)
                writer.writerow([str(_),str(batch_idx),"null",str(start2.elapsed_time(end2)),str(time.time() - reference_time)])
            '''
            fetch_fname.info(str(_) + "," + str(batch_idx) + "," + "null" + "," + str(start2.elapsed_time(end2)) + "," + str(time.time() - reference_time))
            '''
            with open(compute_fname, 'a', newline='') as file: # DEPRECATED
                writer = csv.writer(file)
                writer.writerow([str(_),str(batch_idx),"null",str(start3.elapsed_time(end3)),str(time.time() - reference_time)])
            '''
            compute_fname.info(str(_) + "," + str(batch_idx) + "," + "null" + "," + str(start3.elapsed_time(end3)) + "," + str(time.time() - reference_time))
            # start2_time = time.time() # fetch time per batch starts
            start2.record() # fetch time per batch starts
        end1.record() # e2e epoch time ends
        torch.cuda.synchronize()
        # time_end = (time.time() - start_time) # DEPRECATED
        # loss1, acc1 = test(net,testloader)
        print('Epoch: ' + str(_) + ' Ends')
        # print("Epoch: " + str(_) + ", Loss: " + str(loss) + ", Accuracy: " + str(acc1) + ", Time: " + str(time_end))
        '''
        with open(epoch_fname, 'a', newline='') as file: # DEPRECATED
            writer = csv.writer(file)
            # writer.writerow([str(_),str(time_end),str(loss),str(acc1),str(start1.elapsed_time(end1)),str(time.time() - reference_time)])
            writer.writerow([str(_),"null",str(loss),"null",str(start1.elapsed_time(end1)),str(time.time() - reference_time)])
        '''
        epoch_fname.info(str(_) + "," + "null" + ","  + "\"" + str(loss)  + "\"" + "," + "null" + "," + str(start1.elapsed_time(end1)) + "," + str(time.time() - reference_time))
        """
        if epoch_count == 0:
            os.system('sudo ./drop_caches.sh')
        """
        epoch_count +=1
    """ vmtouch logging """
    '''
    with open(vmtouch_fname, 'a', newline='') as file: # DEPRECATED
        vmtouch_output = os.popen(r"vmtouch -f " + vmtouch_dir + r" | sed 's/^.*://' | sed -z 's/\n/,/g' | sed 's/\s\+/,/g' | sed 's/,\{2,\}/,/g' | sed -e 's/^.\(.*\).$/\1/'")
        vmtouch_output = vmtouch_output.read()+","+str(time.time() - reference_time)
        writer = csv.writer(file)
        writer.writerow(vmtouch_output.split(","))
    '''
    vmtouch_output = os.popen(r"vmtouch -f " + vmtouch_dir + r" | sed 's/^.*://' | sed -z 's/\n/,/g' | sed 's/\s\+/,/g' | sed 's/,\{2,\}/,/g' | sed -e 's/^.\(.*\).$/\1/'")
    vmtouch_output = vmtouch_output.read()+","+str(time.time() - reference_time)
    vmtouch_fname.info(vmtouch_output)
    # with open(vmtouchlogtime_fname, 'a', newline='') as file:
    #     writer = csv.writer(file, delimiter=' ',escapechar=' ', quoting=csv.QUOTE_NONE)
    #     writer.writerow([str(time.time() - reference_time)])
```

#### + Parameters:
 - net: The neural network model.
 - trainloader: DataLoader for the training dataset.
 - testloader: DataLoader for the testing dataset.
 - epoch_fname: File handler or logger for epoch timing.
 - fetch_fname: File handler or logger for fetch timing.
 - compute_fname: File handler or logger for compute timing.
 - vmtouch_fname: File handler or logger for vmtouch output.
 - dataset_outer_folder: Directory containing the dataset.
 - reference_time: Start time for logging.
 - epochs: Number of epochs to train the model.

#### + Training Loop:
 - Sets up criterion (loss function) and optimizer.
 - Logs vmtouch output to track memory page cache status.
 - Drops caches before starting training.
 - For each epoch:
  
         + Records timing using CUDA events.
         + Logs vmtouch output.
         + For each batch:
            +  Measures fetch and compute times.
            +  Performs forward and backward passes and updates weights.
         + Logs fetch and compute times.
         + Logs epoch time.
 - Optionally drops caches after the first epoch.
 - Logs final vmtouch output.