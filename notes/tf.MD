# TensorFlow 🍊
[ [tensorflow.org](https://www.tensorflow.org/), [github](https://github.com/tensorflow/tensorflow), [docker](https://hub.docker.com/r/tensorflow/tensorflow/), [tensorflow-tfx](https://www.tensorflow.org/tfx), [tf-youtube](https://www.youtube.com/@TensorFlow), [tf-dev-certificate](https://www.tensorflow.org/certificate) ]

<img src="./img/tf.png" width=100%>

```python
    $ pip install tensorflow
```

TensorFlow 2.0 represents a significant evolution of the popular open-source machine learning framework. Released in September 2019, TensorFlow 2.0 aims to simplify the development and deployment of machine learning models while providing a more intuitive and flexible user experience. Here, we delve into the key features, components, and advancements in TensorFlow 2.0:


+ <b>`Eager Execution`</b> (Dynamic Computation): TensorFlow 2.0 adopts eager execution by default, allowing developers to evaluate operations immediately, much like Python, making debugging and model development more intuitive.

+ <b>`Keras Integration`</b> (Keras as a High-Level API): TensorFlow 2.0 integrates Keras as its high-level API for building and training deep learning models. Keras provides an easy-to-use interface for creating neural networks.

+ <b>`TensorFlow Data`</b> (Efficient Data Input Pipeline): TensorFlow 2.0 introduces the tf.data API, which simplifies and optimizes data input pipelines, enhancing training efficiency.

+ <b>`Model Subclassing`</b> (Customizable Models): TensorFlow 2.0 enables model subclassing, allowing developers to create custom, highly flexible neural network architectures.

+ <b>`SavedModel Format`</b> (Model Serialization): TensorFlow 2.0 introduces the SavedModel format as a more efficient way to serialize and save machine learning models, facilitating model deployment and serving.
  
+ <b>`TensorFlow Serving`</b> (Simplified Deployment): TensorFlow Serving, an official component of TensorFlow, makes it easier to deploy machine learning models in production environments.

+ <b>`GPU and TPU Acceleration`</b> (Enhanced Hardware Support): TensorFlow 2.0 offers seamless GPU and TPU (Tensor Processing Unit) acceleration, enabling faster model training and inference.

+ <b>`Distribution Strategies`</b> (Distributed Training): TensorFlow 2.0 provides distribution strategies that simplify the process of training models on multiple GPUs or TPUs.

+ <b>`TensorFlow Hub`</b> (Model Sharing): TensorFlow Hub offers a repository for sharing pre-trained models and model components, streamlining the development process.

+ <b>`TensorFlow Lite`</b> (Edge Devices): TensorFlow 2.0 includes TensorFlow Lite, which allows developers to deploy machine learning models on edge devices like mobile phones and IoT devices.

+ <b>`TensorBoard`</b> (Visualization Tools): TensorBoard, the visualization toolkit for TensorFlow, has been improved, making it easier to monitor and analyze training progress.

+ <b>`Community and Ecosystem`</b> (Active Development): TensorFlow boasts a vibrant community and ecosystem, with an extensive collection of libraries, tools, and resources that enhance its capabilities.

+ <b>`TensorFlow Extended`</b> (TFX- End-to-End ML Pipelines): TFX, an end-to-end machine learning platform, is integrated with TensorFlow 2.0, providing solutions for deploying machine learning pipelines at scale.

+ <b>`TensorFlow.js and TensorFlow Lite for Web ( Web Integration)`</b>: TensorFlow 2.0 supports TensorFlow.js for training and deploying models in the browser, as well as TensorFlow Lite for deploying models on mobile and embedded devices.

+ <b>`Future Directions (Ongoing Advancements)`</b> : TensorFlow continues to evolve, with ongoing research and development in areas such as automated machine learning (AutoML), reinforcement learning, and quantum machine learning.


```python

>>> import tensorflow as tf
>>> tf.add(1, 2).numpy()
3
>>> hello = tf.constant('Hello, TensorFlow! 🌸')
>>> hello.numpy()
b'Hello, TensorFlow! 🌸'

```

<b>Tensor :</b> Tensor forms the core framework of TensorFlow. All the computations in TensorFlow involve tensors. It is a matrix of n-dimensions that represents multiple types of data. A tensor can be the result of a computation or it can originate from the input data.


```python

    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0


    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])

    predictions = model(x_train[:1]).numpy()
    predictions

    tf.nn.softmax(predictions).numpy()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    loss_fn(y_train[:1], predictions).numpy()

    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test,  y_test, verbose=2)

    probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
    ])


```

## Semantic Segmentation with TensorFlow 2.0:

Semantic segmentation, a fundamental task in computer vision, involves assigning a class label to each pixel in an image. DeepLab, a prominent Convolutional Neural Network (CNN) architecture, has been at the forefront of semantic segmentation due to its ability to capture intricate details and contextual information. In this abstract, we delve into the mathematical foundations of DeepLab and provide practical TensorFlow 2.0 code examples for semantic segmentation.

Mathematical Foundations: DeepLab's success lies in its architectural innovations, notably atrous (dilated) convolutions and the spatial pyramid pooling module.

### Atrous Convolutions:

Atrous convolutions enable the network to have an effectively larger receptive field without increasing the number of parameters.
The mathematical concept involves introducing gaps (dilation rate) in the convolutional kernel to sample pixels at larger intervals.
Mathematically, the output of an atrous convolution with a dilation rate "d" can be represented as:

```
    Y[i] = ∑(k=0 to K) X[i + d * k] * W[k]
```

### Spatial Pyramid Pooling (SPP):
SPP captures multi-scale information from different regions of the input feature map.
Mathematically, it involves dividing the input feature map into grids of different sizes and pooling the information from each grid independently.

```py
import tensorflow as tf
from tensorflow.keras import layers

# Define the DeepLab model
def DeepLabV3Plus(input_shape, num_classes):
    # Encoder (backbone)
    backbone = tf.keras.applications.Xception(input_shape=input_shape, include_top=False)
    
    # Atrous Spatial Pyramid Pooling (ASPP)
    aspp = layers.Conv2D(256, (1, 1), activation='relu')(backbone.output)
    aspp = layers.Conv2D(256, (3, 3), dilation_rate=6, activation='relu', padding='same')(aspp)
    aspp = layers.Conv2D(256, (3, 3), dilation_rate=12, activation='relu', padding='same')(aspp)
    aspp = layers.Conv2D(256, (3, 3), dilation_rate=18, activation='relu', padding='same')(aspp)
    aspp = layers.GlobalAveragePooling2D()(aspp)
    aspp = layers.Reshape((1, 1, 256))(aspp)
    aspp = layers.Conv2D(256, (1, 1), activation='relu')(aspp)
    aspp = layers.UpSampling2D(size=(4, 4))(aspp)
    
    # Decoder
    decoder = layers.UpSampling2D(size=(4, 4))(backbone.output)
    decoder = layers.Concatenate()([decoder, aspp])
    decoder = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(decoder)
    decoder = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(decoder)
    
    # Output layer
    output = layers.Conv2D(num_classes, (1, 1), activation='softmax')(decoder)
    
    model = tf.keras.Model(inputs=backbone.input, outputs=output)
    return model

# Create the DeepLab model
model = DeepLabV3Plus(input_shape=(256, 256, 3), num_classes=21)
```


resources: [udemy-tensorflow-developer-certificate](https://www.udemy.com/course/tensorflow-developer-certificate-machine-learning-zero-to-mastery/), [tf2-machine-learning-mastery](https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/), [tf-quick-start-guide](https://www.tensorflow.org/tutorials/quickstart/beginner), [tf-models](https://github.com/tensorflow/models), [computer vision with tf2](https://youtu.be/IA3WxTTPXqQ), [tensorflow-c++/API](https://www.tensorflow.org/api_docs/cc), [TensorFlow 2.0 Complete Course](https://youtu.be/tPYj3fFJGjk?si=qOGMA0lRvyMpGnGF), [Deep Learning for Computer Vision with Python and TensorFlow](https://youtu.be/IA3WxTTPXqQ?si=7oi12Gb93Y6jNxF3), [TensorFlow 2.0](https://www.youtube.com/watch?v=5Ym-dOS9ssA&list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb), [Build a Deep Learning Model that can LIP READ using Python and Tensorflow](https://youtu.be/uKyojQjbx4c?si=81cAVpNyRlsEzBXF).