# PyTorch 🔥

PyTorch is a powerful, yet easy-to-use deep learning library for Python, mainly used for applications such as computer vision and natural language processing.

While TensorFlow was developed by Google, PyTorch was developed by Facebook's AI Research Group, which has recently shifted management of the framework to the newly created PyTorch Foundation, which is under the supervision of the Linux Foundation.

The flexibility of PyTorch allows easy integration of new data types and algorithms, and the framework is also efficient and scalable, since it was designed to minimize the number of computations required and to be compatible with a variety of hardware architectures.

`Components of PyTorch` :

+ `Tensors`:

    Tensors are the building blocks of PyTorch, similar to arrays in NumPy but with the added ability to run on GPUs (Graphics Processing Units).
    A tensor is a multi-dimensional array, and PyTorch provides many operations to manipulate these tensors, such as adding, multiplying, reshaping, etc.

+ `Autograd (Automatic Differentiation)`:

    PyTorch’s `autograd` feature automatically calculates gradients (derivatives) needed for backpropagation.
    This is crucial for training neural networks as it allows the model to update its weights based on the loss function.
    When you perform operations on tensors, PyTorch builds a computational graph, and autograd helps you backtrack through this graph to compute gradients.

+ `Neural Network Module (torch.nn)`:

    The `torch.nn` module provides a wide range of pre-built layers, loss functions, and utilities to build neural networks.
    It includes layers like convolutional layers for image processing, recurrent layers for sequence data, and linear layers for fully connected networks.

 + `Optimizers (torch.optim)`:

    Optimizers in PyTorch help in updating the model’s parameters based on the gradients computed by autograd.
    Common optimizers include Stochastic Gradient Descent (SGD), Adam, and RMSprop, each with different strategies for adjusting the learning rate and momentum during training.

+ `DataLoader`:

    The DataLoader is a utility that allows you to load and preprocess data in batches.
    It handles tasks like shuffling data, applying transformations, and managing batch sizes, making it easier to feed data into a model during training or inference.

+ `PyTorch Hub`:

    PyTorch Hub is a repository of pre-trained models that can be easily loaded and fine-tuned for specific tasks.
    These models can save time and computational resources, especially when working with limited edge devices.

+ `TorchScript`:

    TorchScript is a way to convert your PyTorch models into a format that can be run independently from Python.
    This is particularly useful for deploying models to edge devices where Python may not be available.
    TorchScript allows you to optimize models and make them more efficient for inference on resource-constrained devices.

+ `ONNX (Open Neural Network Exchange)`:

    ONNX is an open format for representing machine learning models.
    PyTorch models can be exported to ONNX format, which allows them to be used in different environments, including edge devices that may use frameworks like TensorFlow or Caffe2.
    ONNX is important for interoperability and for running models on different hardware and platforms.

[ [[web](https://pytorch.org/)], [[documentation](https://pytorch.org/docs/stable/index.html)], [[github](https://github.com/pytorch/pytorch)], [[tutorial](https://pytorch.org/tutorials/)], [[machinelearningmastery.](https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/)], [[hub.docker](https://hub.docker.com/r/pytorch/pytorch/tags)], [[learnpytorch.io](https://www.learnpytorch.io/01_pytorch_workflow/)], [[paper](https://arxiv.org/abs/1912.01703)], [[pytorch-2.0](https://pytorch.org/get-started/pytorch-2.0/)] ]

<img src="./img/pyt.png" width=30%><a> </a><img src="./img/pytorch-ecosystem.png" width=60%>

[ [torchvision](https://github.com/pytorch/vision), [pytorch-geometric](https://github.com/pyg-team/pytorch_geometric), [ignite](https://github.com/pytorch/ignite), [torch-rl](https://github.com/pytorch/rl), [libMTL](https://github.com/median-research-group/LibMTL), [botorch](https://github.com/pytorch/botorch), [torchdyn](https://github.com/DiffEqML/torchdyn), [OpenMined](https://github.com/OpenMined), [skorch](https://github.com/skorch-dev/skorch), [advertorch](https://github.com/BorealisAI/advertorch) ]

PyTorch 🔥 Tensors:

```python
    import torch

    x = torch.empty(3) #1d vector with 3 empty elements
    x = torch.empty(2, 3) #2d matrix with empty values
    x = torch.empty(2, 2, 3) #3d matrix

    y = torch.rand(2, 2) #2d matrix with random values
    z = torch.zeros(2, 3)
    z = torch.ones(2, 2) 
    z = torch.ones(2, 2, dtype=torch.int)  
    z = torch.ones(2, 2, dtype=torch.float16) 
    
    print(x.dtype)
    print(z.size())

    u = torch.tensor([2.5, 0.1]) #tensor from list

    x1 = torch.rand(2, 2)
    x2 = torch.rand(2, 2)
    x3 = x1 + x2 #element wise addition ( ' -, *, / ' )
    x3 = torch.add(x1, x2) #same operation ( torch.sub(x1, x2), torch.mul(x1, x2), torch.div(x1, x2) )

    x2.add_(x1) #inline addition, ' _ ' for inplace operations in pytorch

    x2[1, 1].item() #value of the tensor with 1 item

    x4 = torch.tensor(4, 4)
    x4.view(16) #view in 1d
    x4.view(-1, 8) #pytorch will determinee the right dimension with '-1'
```
numpy to torch tensor and vice versa:

```python
    import torch
    import numpy as np

    if torch.cuda.is_available():
        device = torch.device("cuda")
        x = torch.ones(5, device=device) #tensor on 'GPU'
        y = torch.ones(5)
        z = x + y
        z = z.to("cpu") #move back to 'CPU'

    a = torch.ones(5)
    b = a.numpy()
    a = a.to(device) #move the tensor operation to 'GPU'

    c = np.ones(5)
    d = torch.from_numpy(c)

    a.add_(1) #'b' gets updated as well as they both point to the same memory location

    e = np.ones(5, requires_grad=True) 
    print(d.shape)
    print(d.dtype)
    print(d.device)

    new_t = torch.rand_like(d, dtype=torch.float)

    my_shape = (3, 3)
    rand_t = torch.rand(my_shape) #create tensor from shape
```

Autograd (calculating gradients):

```python
    import torch
    x = torch.rand(3, requires_grad=True) #False by default and creates a computational graph when True

    y = x + 2
    z = y*y*2
    z = z.mean()

    z.backward() #calculates dz/dx
    print(x.grad) #stores tensor gradients 


    v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
    z.backward(v) #needs arguments in case of vector tensor

    x.requires_grad_(False) #stop pytorch tracking history
    x.detach() #creates new tensor that sheds the gradients
    with torch.no_grad(): #wrapping around will not track history


    import torch
    weights = torch.ones(4, requires_grad=True)

    for epoch in range(3):
        model_output = (weights*3).sum()
        model_output.backward()
        print(weights.grad)
        weights.grad.zero_()
```

optimizers in torch:

```python
    optimizer = torch.optim.SGD(weights, lr=0.01)
    optimizer.step()
    optimizer.zero_grad()
```

`requires_grad=True` tells pytorch that we will have to calculate gradient of tensor in later optimization steps.

### PREDICTION: 

Prediction : (PyTorch Model) ;
Gradients Computation : (Autograd) ;
Loss Computation : (PyTorch Loss) ;
Parameter Updates : (PyTorch Optimizer).

```python
    import numpy as np

    X = np.array([1,2,3,4], dtype=np.float32)
    Y = np.array([2,4,6,8], dtype=np.float32)

    w = 0.0

    #model prediction
    def forward(x):
        return w * x

    #loss
    def loss(y, y_predicted):
        return ((y_pred-y)**2).mean() 

    #gradient
    def gradient(x,y , y_predicted):
        retrun np.dot(2*x, y_ypred - y).mean()

    print(f'Prediction before training: f(5) = {forward(5):,3f}')

    learning_rate = 0.01
    n_iters = 10

    for epoch in range(n_iters): # prediction
        y_pred = forward(X) #forward pass
        l = loss(Y. y_pred) #loss
        dw = gradient(X,Y,y_pred) #gradients
        w -= learning_rate * dw #update weights
        if epoch % 1 == 0:
            print(f'epoch {epoch*1}: w = {w:.3f}, loss = {l:.8f}')


    print(f'Prediction after training: f(5) = {forward(5):,3f}')
```

If done with PyTorch:

```python
    import torch
    X = torch.tensor([1,2,3,4], dtype=torch.float32)
    Y = torch.tensor([2,4,6,8], dtype=torch.float32)

    w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)



    for epoch in range(n_iters): # prediction
        y_pred = forward(X) #forward pass
        l = loss(Y. y_pred) #loss
        l.backward()
        with torch.no_grad():
            w -= leaning_rate * w.grad
        w.grad.zero_()
    
```
### PyTorch Tensor Operations:

```python

    zeros_tensor = torch.zeros((2, 3))
    transposed = zeros_tensor.T

#loading dataset

    from torchvision import datasets
    from torchvision.transforms import ToTensor
    import matplotlib.pyplot as plt

```

### PyTorch CNN:

```python
from torch.autograd import Variable
import torch.nn.functional as F


class SimpleCNN(torch.nn.Module):
   def __init__(self):
      super(SimpleCNN, self).__init__()
      #Input channels = 3, output channels = 18
      self.conv1 = torch.nn.Conv2d(3, 18, kernel_size = 3, stride = 1, padding = 1)
      self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
      #4608 input features, 64 output features (see sizing flow below)
      self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
      #64 input features, 10 output features for our 10 defined classes
      self.fc2 = torch.nn.Linear(64, 10)


def forward(self, x):
   x = F.relu(self.conv1(x))
   x = self.pool(x)
   x = x.view(-1, 18 * 16 *16)
   x = F.relu(self.fc1(x))
   #Computes the second fully connected layer (activation applied later)
   #Size changes from (1, 64) to (1, 10)
   x = self.fc2(x)
   return(x)     
```

## [ [YOLO TRAIN](./code/yolo_train/), [ResNet TRAIN](./code/resnet_train/), [BERT TRAIN](./code/BERT_train/), [LSTM TRAIN](./code/LSTM_train/), [MobileNet Train](./code/mobnet_train/) ]

### Efficient Data Loading with `DataLoader`:
+ Use `num_workers` in `DataLoader` to parallelize data loading and augment data on-the-fly.
+ `pin_memory=True` can speed up data transfer to GPU when using `DataLoader`.
+ `persistent_workers=True` helps to avoid re-initializing workers between epochs, reducing overhead.

### Custom Autograd Functions:
+ Define custom gradients by subclassing `torch.autograd.Function`. This allows for highly efficient and flexible backpropagation, useful in scenarios like implementing novel layers or loss functions.

### Mixed Precision Training:
+ Use `torch.cuda.amp` for mixed precision training, which can significantly reduce memory usage and speed up training, especially on GPUs with Tensor Cores (e.g., NVIDIA V100, A100).

### Model Checkpointing:
+ Save and load model states with `torch.save()` and `torch.load()`, but be cautious to save the state_dict instead of the entire model object to avoid issues with architecture changes.
Periodically save checkpoints during training, not just the final model, to recover from crashes.

### Gradient Accumulation:
+ For large batch training that doesn't fit into GPU memory, accumulate gradients across multiple mini-batches using `loss.backward()` without stepping the optimizer after each batch. Instead, step the optimizer every few batches.

### Efficient Memory Management:
+ Use `torch.no_grad()` to disable gradient computation for inference, reducing memory consumption and speeding up computations.
Clear gradients manually with `optimizer.zero_grad()` before the next optimization step to avoid accumulating gradients.


resources :  [Implementing ConvNext in PyTorch](https://towardsdatascience.com/implementing-convnext-in-pytorch-7e37a67abba6), [transformers tutorial](https://github.com/NielsRogge/Transformers-Tutorials), [python-docathon](https://pytorch.org/blog/announcing-docathon/), [pytorch-youtube](https://www.youtube.com/@PyTorch/videos), [incredible pytorch](https://github.com/ritchieng/the-incredible-pytorch), [awesome pytorch list](https://github.com/bharathgs/Awesome-pytorch-list), [pytorch deep dive tutorial](https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/), [pytorch tutorials](https://pytorch.org/tutorials/), @github/[pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial), [learnpytorch.io](https://www.learnpytorch.io/), [PyTorch for DL & ML](https://youtu.be/V_xro1bcAuA?si=kEmbxC1CxNsd2oqP), [The StatQuest Introduction to PyTorch](https://youtu.be/FHdlXe1bSe4?si=V_1fU23kGzMttG6P), [Learn PyTorch for deep learning in a day. Literally.](https://youtu.be/Z_ikDlimN6A?si=NAgcfskOQ0YQLheE), [Build an AI/ML Tennis Analysis system with YOLO, PyTorch, and Key Point Extraction](https://youtu.be/L23oIHZE14w?si=xXDJHrPNN1TBhz27), [Deep Learning with PyTorch](https://www.youtube.com/watch?v=kY14KfZQ1TI&list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1).