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

#### 🌸 
