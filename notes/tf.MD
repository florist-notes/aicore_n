# TensorFlow

[tensorflow.org](https://www.tensorflow.org/), [github](https://github.com/tensorflow/tensorflow), [docker](https://hub.docker.com/r/tensorflow/tensorflow/), [tensorflow-tfx](https://www.tensorflow.org/tfx), [tf-youtube](https://www.youtube.com/@TensorFlow), [tf-dev-certificate](https://www.tensorflow.org/certificate)

<img src="./img/tf.png" width=100%>

```python

    $ pip install tensorflow

```

TensorFlow was first made public in 2015, while the first stable version appeared on February 11, 2017. It is created and maintained by Google. Ever since then, it has become one of the most popular frameworks for deep learning and machine learning projects. It has a vast library for large-scale machine learning and numerical computation. Here are some more milestones of TensorFlow:

+ In December 2017, Kubeflow is released for the operation and deployment of TensorFlow on Kubernetes.
+ In March 2018, TensorFlow 1.0 was released for machine learning in JavaScript.
+ In Jan 2019, TensorFlow 2.0 is released which adds a number of components to TensorFlow.
+ In May 2019, TensorFlow Graphic was released for deep learning in computer graphics.


TensorFlow allows you to create dataflow graphs that describe how data moves through a graph. The graph consists of nodes that represent a mathematical operation. A connection or edge between nodes is a multidimensional data array. It takes inputs as a multi-dimensional array where you can construct a flowchart of operations that can be performed on these inputs.

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

resources: [udemy-tensorflow-developer-certificate](https://www.udemy.com/course/tensorflow-developer-certificate-machine-learning-zero-to-mastery/), [tf2-machine-learning-mastery](https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/), [tf-quick-start-guide](https://www.tensorflow.org/tutorials/quickstart/beginner), [tf-models](https://github.com/tensorflow/models), [computer vision with tf2](https://youtu.be/IA3WxTTPXqQ), [tensorflow-c++/API](https://www.tensorflow.org/api_docs/cc)