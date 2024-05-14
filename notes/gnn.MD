# Graph Neural Networks:
Graphs are versatile data structures that represent relationships between entities. Many real-world datasets such as social networks, citation networks, and molecular structures can be naturally modeled as graphs. Traditional neural networks are not well-suited for processing graph-structured data due to their fixed-size inputs and lack of inherent mechanisms to handle irregular graph structures. Graph Neural Networks (GNNs) address these limitations by leveraging the graph topology to perform node or graph-level tasks.

<img src="./img/gnn.png" width=100%>

Graph Neural Networks apply the predictive power of deep learning to rich data structures that depict objects and their relationships as points connected by lines in a graph. In GNNs, data points are called nodes, which are linked by lines — called edges — with elements expressed mathematically so machine learning algorithms can make useful predictions at the level of nodes, edges or entire graphs.

Graph Neural Networks (GNNs) have emerged as a groundbreaking paradigm in the realm of machine learning, enabling the analysis and modeling of complex data structured as graphs. 

GNNs are founded on several key concepts:
+ <b>`Graph Representation`:</b> GNNs operate on graph-structured data, which consists of nodes (entities) connected by edges (relationships). This flexible representation is suitable for a wide range of applications, including social networks, recommendation systems, and molecular chemistry.
+ <b>`Neighborhood Aggregation`:</b> GNNs leverage information from neighboring nodes to update a node's representation. This recursive process allows GNNs to capture local and global graph patterns.
+ <b>`Learnable Parameters`:</b> GNNs employ learnable parameters, such as neural network layers, to adaptively transform node features during message passing.
+ <b>`Node and Graph-level Tasks`:</b> GNNs can be tailored to perform node-level tasks (e.g., node classification) or graph-level tasks (e.g., graph classification) by aggregating and processing information accordingly.

### Architecture of Graph Neural Networks:
GNNs operate on graph-structured data by recursively aggregating information from neighboring nodes. The core components of a GNN architecture include:

+ `Node Embedding`: Each node in the graph is associated with an initial feature vector representing its attributes.
+ `Message Passing`: GNNs propagate information between neighboring nodes by passing messages along the edges of the graph. This process allows nodes to aggregate information from their local neighborhoods.
+ `Aggregation Function`: Aggregated information from neighboring nodes is combined to update the representation of each node.
+ `Readout Function`: For graph-level tasks, a readout function aggregates node representations to produce a graph-level embedding.

Training GNNs typically involves optimizing a loss function that measures the discrepancy between predicted and ground-truth labels or values.

GNN architecture for node classification on a graph dataset:
```py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolutionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolutionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, adjacency_matrix, node_features):
        # adjacency_matrix: [N, N], node_features: [N, input_dim]
        aggregate = torch.matmul(adjacency_matrix, node_features)  # Aggregate neighbor features
        output = self.linear(aggregate)  # Linear transformation
        return output

class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNeuralNetwork, self).__init__()
        self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
        self.gc2 = GraphConvolutionLayer(hidden_dim, output_dim)

    def forward(self, adjacency_matrix, node_features):
        h1 = F.relu(self.gc1(adjacency_matrix, node_features))  # First Graph Convolution
        h2 = self.gc2(adjacency_matrix, h1)  # Second Graph Convolution
        return F.log_softmax(h2, dim=1)  # Log Softmax for classification

# Example usage:
# Define the adjacency matrix and node features
adjacency_matrix = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)
node_features = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)

# Define the model
input_dim = 2  # Dimensionality of input features
hidden_dim = 16  # Dimensionality of hidden layer
output_dim = 2  # Number of classes (for classification)
model = GraphNeuralNetwork(input_dim, hidden_dim, output_dim)

# Forward pass
output = model(adjacency_matrix, node_features)
print("Output probabilities:", output)
```

In this code:
+ `GraphConvolutionLayer`: Implements a single graph convolutional layer. It applies a linear transformation to the aggregated features of neighboring nodes.
+ `GraphNeuralNetwork`: Defines a two-layer Graph Neural Network using `GraphConvolutionLayer`.
+ The `forward` method of `GraphNeuralNetwork` performs two graph convolutional operations and applies log softmax for node classification.
+ The example usage section demonstrates how to define an adjacency matrix and node features and perform a forward pass through the GNN model.

PyG (PyTorch Geometric) [[repo](https://pytorch-geometric.readthedocs.io/en/latest/)] ( Geometric Computer Vision) | @github/[pytorch_geometric](https://github.com/pyg-team/pytorch_geometric), @github/[GNNpapers](https://github.com/thunlp/GNNPapers) | [Learning on Graphs (LoG)](https://logconference.org/)

resources: [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/), [Understanding Convolutions on Graphs](https://distill.pub/2021/understanding-gnns/),  [Graph neural networks: A review of methods and applications](https://arxiv.org/ftp/arxiv/papers/1812/1812.08434.pdf), [Theoretical Foundations of Graph Neural Networks](https://youtu.be/uF53xsT7mjc), Deep Learning on Graphs : [(book)], [The basics of spatio-temporal graph neural networks](https://youtu.be/RRMU8kJH60Q), [Stanford CS224W: Machine Learning with Graphs](https://www.youtube.com/watch?v=JDW82csukhE&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=58), [Microsoft Research : An Introduction to Graph Neural Networks: Models and Applications](https://youtu.be/zCEYiCxrL_0), [Graph Neural Networks and Diffusion PDEs | Benjamin Chamberlain & James Rowbottom](https://www.youtube.com/watch?v=9SMbH18nMUg), [The ultimate intro to Graph Neural Networks. Maybe.](https://www.youtube.com/watch?v=me3UsMm9QEs), [DeepFindr : GNN](https://www.youtube.com/watch?v=fOctJB4kVlM&list=PLV8yxwGOxvvoNkzPfCx2i8an--Tkt7O8Z), [TensorFlow : Intro to graph neural networks (ML Tech Talks)](https://youtu.be/8owQBFAHw7E), [Theoretical Foundations of Graph Neural Networks](https://www.youtube.com/watch?v=uF53xsT7mjc), [WelcomeAIOverlords : Graph Neural Networks](https://www.youtube.com/watch?v=OI0Jo-5d190&list=PLSgGvve8UweGx4_6hhrF3n4wpHf_RV76_&index=1), [Tutorial: Graph Neural Networks in TensorFlow: A Practical Guide](https://youtu.be/JqWROPYeqjA), @github/[gbdll](https://github.com/naganandy/graph-based-deep-learning-literature), @github/[dgl](https://github.com/dmlc/dgl), [gnn-blog](https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial), [gnn-blog-2](https://hhaji.github.io/Deep-Learning/Graph-Neural-Networks/), @github/[graph-neural-network](https://github.com/topics/graph-neural-network), @github/[GNNs-Recipe](https://github.com/dair-ai/GNNs-Recipe), @github/[graph-neural-networks](https://github.com/topics/graph-neural-networks).