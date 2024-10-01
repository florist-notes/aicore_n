# Deep Learning Theory

I took AEC : CS 872 ( Machine Learning ) and AEC : CS771 ( Artificial Intelligence ) classes at ug and few related courses via online programs : { [classes 1](https://twitter.com/florist_notes/status/1641001006581383168), [classes 2](https://twitter.com/florist_notes/status/1641003166744313856), [classes 3](https://twitter.com/florist_notes/status/1659689325376983040) } along with few noteworthy ML projects { [proj 1](https://twitter.com/florist_notes/status/1640149371936276482), [proj 2](https://twitter.com/florist_notes/status/1639864034244657155), [proj 3](https://twitter.com/florist_notes/status/1652428726032904192) }. 

A Deep Neural Network (DNN) is a type of machine learning model designed to mimic the way the human brain processes information. It's called "deep" because it has multiple layers of interconnected "neurons" (or nodes) that work together to learn patterns from data.

`Components of a Deep Neural Network
Neurons (Nodes)`:

Think of neurons as tiny decision-makers. Each neuron receives some input, processes it, and then passes an output to the next layer of neurons.
Neurons are organized into layers: an input layer (where the data enters), one or more hidden layers (where the processing happens), and an output layer (which gives the final result).

+ `Layers`:

    + `Input Layer`: This is the first layer where data is fed into the network. Each neuron in this layer represents one feature of the input data (like pixels in an image or words in a sentence).
    + `Hidden Layers`: These are the middle layers where most of the computation occurs. The "deep" in deep learning comes from having multiple hidden layers.
    + `Output Layer`: The final layer that produces the result, like classifying an image as a cat or dog.

+ `Weights`:

    Each connection between neurons has a weight. A weight determines how much influence the output of one neuron has on the neuron it's connected to in the next layer.
    During training, the network adjusts these weights to improve its predictions.

+ `Bias`:

    Bias is an additional parameter in each neuron that allows the network to shift the activation function to the left or right. It helps the model fit the data better.

+ `Activation Function`:

    After a neuron processes its input, it passes the result through an activation function. This function decides whether the neuron should be "activated" (i.e., send a signal to the next layer) or not.

    Common activation functions include:
    + `Sigmoid`: Outputs a value between 0 and 1, useful for binary classification.
    + `ReLU (Rectified Linear Unit)`: Outputs the input directly if it's positive; otherwise, it outputs zero. It's widely used because it helps networks train faster.
    + `Tanh`: Outputs values between -1 and 1, which helps center the data.

+ `Loss Function`:

    The loss function measures how well the network's predictions match the actual data. It's a way to evaluate how far off the predictions are from the correct answers.
    Examples of loss functions:
    + `Mean Squared Error (MSE)`: Used for regression tasks where the network predicts a continuous value. It calculates the average squared difference between the predicted and actual values.
    + `Cross-Entropy Loss`: Used for classification tasks, it measures how different the predicted probabilities are from the actual class labels.

+ `Gradient Descent`:

    Gradient Descent is the optimization method used to update the weights in the network to minimize the loss function.
    Imagine the loss function as a mountain, and gradient descent is like finding the fastest way down to the lowest point (the minimum loss).
    It works by calculating the gradient (slope) of the loss function with respect to each weight and then adjusting the weights in the direction that reduces the loss.

+ `Backpropagation`:

    Backpropagation is the process of updating the weights in the network. After making a prediction, the network calculates the loss and then works backward, layer by layer, adjusting the weights to reduce the loss.

    It ensures that each neuron learns from the errors it made in the previous prediction.


<table>
<tr>
<th>



[Computer Vision](./cv.MD)
</th>
<th>

[Deep Reinforcement Learning](./drl.MD)

</th>
<th>

[Graph Neural Networks](./gnn.MD)

</th>
</tr>

<tr>
<td>

[SLAM](https://github.com/florist-notes/vSLAM)
</td>
<td>

[Meta Learning](#)</td>
</tr>
</table>

<img src="./img/dl.png" width=100%>

notes on tools: [[PyTorch](./pyt.MD)], [[TensorFlow](./tf.MD)], [[INT 1](./int1.MD)], [[INT 2](./int2.MD)], [[INT 3](./int3.MD)], [[INT 4](./int4.MD)], [[INT 5](./int5.MD)], [[INT 6](./int6.MD)], [[INT 7](./int7.MD)].

