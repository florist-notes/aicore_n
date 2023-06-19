# Artificial intelligence : s-AI-kia

here is my mini resource guide on AI > ML > DL:

<center><img src="./img/st.png" width=100%></center>



## Artificial Intelligence [[RESOURCES](./notes/airesource.MD)]

```latex
    AI ~ θ|1

```

## Deep Learning [[theory](./notes/dltheory.MD)] 


## Deep Learning [[code](./notes/dlcode.MD)] 

```python
    import gymnasium as gym
    import math
    import random
    import matplotlib
    import matplotlib.pyplot as plt
    from collections import namedtuple, deque
    from itertools import count

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F

    env = gym.make("CartPole-v1")

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

```