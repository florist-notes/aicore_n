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