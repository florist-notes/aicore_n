#include <iostream>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

using namespace tensorflow;

int main() {
    // Create a TensorFlow session
    Scope root = Scope::NewRootScope();
    ClientSession session(root);

    // Define input dimensions
    int batch_size = 1;
    int input_height = 28;
    int input_width = 28;
    int input_channels = 1;

    // Define input placeholder
    auto input = ops::Placeholder(root, DT_FLOAT);
    auto reshaped_input = ops::Reshape(root, input, {batch_size, input_height, input_width, input_channels});

    // Define convolutional layer
    int num_filters = 32;
    int filter_size = 3;
    auto conv1 = ops::Conv2D(root, reshaped_input, ops::Const(root, {filter_size, filter_size, input_channels, num_filters}), {1, 1, 1, 1}, "SAME");

    // Define max-pooling layer
    auto pool1 = ops::MaxPool(root, conv1, ops::Const(root, {1, 2, 2, 1}), {1, 2, 2, 1}, "SAME");

    // Define fully connected layer
    int num_hidden_units = 128;
    auto pool1_flat = ops::Reshape(root, pool1, {batch_size, -1});
    auto fc1 = ops::MatMul(root, pool1_flat, ops::Const(root, {{pool1_flat.shape().dim_size(1), num_hidden_units}}));
    auto relu1 = ops::Relu(root, fc1);

    // Define output layer
    int num_classes = 10;
    auto logits = ops::MatMul(root, relu1, ops::Const(root, {{num_hidden_units, num_classes}}));

    // Initialize variables
    std::vector<Tensor> outputs;
    session.Run({{input, Tensor(DataType::DT_FLOAT, TensorShape({batch_size, input_height, input_width, input_channels}))}},
                {logits}, {}, &outputs);

    // Print the output
    std::cout << "Output Tensor: " << outputs[0].matrix<float>() << std::endl;

    return 0;
}
