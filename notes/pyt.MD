# PyTorch

[[web](https://pytorch.org/)], [[documentation](https://pytorch.org/docs/stable/index.html)], [[github](https://github.com/pytorch/pytorch)], [[tutorial](https://pytorch.org/tutorials/)], [[machinelearningmastery.](https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/)], [[hub.docker](https://hub.docker.com/r/pytorch/pytorch/tags)], [[learnpytorch.io](https://www.learnpytorch.io/01_pytorch_workflow/)], [[paper](https://arxiv.org/abs/1912.01703)], [[pytorch-2.0](https://pytorch.org/get-started/pytorch-2.0/)]

<img src="./img/pyt.png" width=30%><a> </a><img src="./img/pytorch-ecosystem.png" width=60%>

[torchvision](https://github.com/pytorch/vision), [pytorch-geometric](https://github.com/pyg-team/pytorch_geometric), [ignite](https://github.com/pytorch/ignite), [torch-rl](https://github.com/pytorch/rl), [libMTL](https://github.com/median-research-group/LibMTL), [botorch](https://github.com/pytorch/botorch), [torchdyn](https://github.com/DiffEqML/torchdyn), [OpenMined](https://github.com/OpenMined), [skorch](https://github.com/skorch-dev/skorch), [advertorch](https://github.com/BorealisAI/advertorch).

tensors:

```python
    import torch

    x = torch.empty(3) #1d vector with 3 empty elements
    x = torch.empty(2, 3) #2d matrix with empty values
    x = torch.empty(2, 2, 3) #3d matrix

    y = torch.rand)(2, 2) #2d matrix with random values
    z = torch.zeros(2, 3)
    z = torch.ones(2, 2) 
    z = torch.ones(2, 2, dtype=torch.int)  
    z = torch.ones(2, 2, dtype=torch.float16) 
    
    print(x.dtype)
    print(z.size())

    u = torch.tensor([2.5, 0.1]) #tensor from list

    x1 = torch.rand(2, 2)
    x2 = torch.rand(2, 2)
    x3 = x1 + x2 #element wise addition ( ' -, *, / ')
    x3 = torch.add(x1, x2) #same operation ( torch.sub(x1, x2), torch.mul(x1, x2), torch.div(x1, x2) )

    x2.add_(x1) #inline addition , ' _ ' for inplace operations in pytorch

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
```

autograd (calculating gradients):

```python
    import torch
    x = torch.rand(3, requires_grad=True) #False by default and creates a computational graph when True

    y = x + 2
    z = y*y*2
    z = z.mean()

    z.backward() #calculates dz/dx
    print(x.grad) #stores tensor gradients 


    v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
    z.backward(v) #neds arguments in case of vector tensor

    x.requires_grad_(False) #stop pytorh tracking history
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

STEPS:

PREDICTION: PyTorch Model > Gradients Computation : Autograd > Loss Computation : PyTorch Loss, Parameter Updates : PyTorch Optimizer

```python
    import numpy as np

```






resources :  [Implementing ConvNext in PyTorch](https://towardsdatascience.com/implementing-convnext-in-pytorch-7e37a67abba6), [transformers tutorial](https://github.com/NielsRogge/Transformers-Tutorials), [python-docathon](https://pytorch.org/blog/announcing-docathon/), [pytorch-youtube](https://www.youtube.com/@PyTorch/videos), [incredible pytorch](https://github.com/ritchieng/the-incredible-pytorch), [awesome pytorch list](https://github.com/bharathgs/Awesome-pytorch-list).