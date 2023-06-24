# PyTorch

[[web](https://pytorch.org/)], [[documentation](https://pytorch.org/docs/stable/index.html)], [[github](https://github.com/pytorch/pytorch)], [[tutorial](https://pytorch.org/tutorials/)], [[machinelearningmastery.](https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/)], [[hub.docker](https://hub.docker.com/r/pytorch/pytorch/tags)], [[learnpytorch.io](https://www.learnpytorch.io/01_pytorch_workflow/)], [[paper](https://arxiv.org/abs/1912.01703)], [[pytorch-2.0](https://pytorch.org/get-started/pytorch-2.0/)]

<img src="./img/pyt.png" width=30%><a> </a><img src="./img/pytorch-ecosystem.png" width=60%>

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

`requires_grad=True` tells pytorch that we will have to calculate gradient of tensor in later optimization steps.







resources :  [Implementing ConvNext in PyTorch](https://towardsdatascience.com/implementing-convnext-in-pytorch-7e37a67abba6), [transformers tutorial](https://github.com/NielsRogge/Transformers-Tutorials), [python-docathon](https://pytorch.org/blog/announcing-docathon/), [pytorch-youtube](https://www.youtube.com/@PyTorch/videos)