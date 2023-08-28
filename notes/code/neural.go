// This code creates a neural network to solve the XOR problem, which is a simple binary classification task. It uses a feedforward architecture with one hidden layer, sigmoid activation functions, and gradient descent for training.

package main

import (
    "fmt"
    "math"
    "math/rand"
    "time"

    "gonum.org/v1/gonum/mat"
)

func sigmoid(x float64) float64 {
    return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
    return x * (1.0 - x)
}

func main() {
    rand.Seed(time.Now().UnixNano())

    // Define neural network parameters
    inputSize := 2
    hiddenSize := 3
    outputSize := 1
    learningRate := 0.1
    epochs := 10000

    // Initialize weights and biases with random values
    inputHiddenWeights := mat.NewDense(hiddenSize, inputSize, nil)
    inputHiddenWeights.Apply(func(_, _ int, _ float64) float64 {
        return rand.Float64()
    }, inputHiddenWeights)

    hiddenOutputWeights := mat.NewDense(outputSize, hiddenSize, nil)
    hiddenOutputWeights.Apply(func(_, _ int, _ float64) float64 {
        return rand.Float64()
    }, hiddenOutputWeights)

    hiddenBiases := mat.NewDense(hiddenSize, 1, nil)
    hiddenBiases.Apply(func(_, _ int, _ float64) float64 {
        return rand.Float64()
    }, hiddenBiases)

    outputBiases := mat.NewDense(outputSize, 1, nil)
    outputBiases.Apply(func(_, _ int, _ float64) float64 {
        return rand.Float64()
    }, outputBiases)

    // Define training data (XOR problem)
    X := mat.NewDense(4, inputSize, []float64{0, 0, 0, 1, 1, 0, 1, 1})
    Y := mat.NewDense(4, outputSize, []float64{0, 1, 1, 0})

    // Training loop
    for epoch := 0; epoch < epochs; epoch++ {
        // Forward propagation
        var hiddenOutput mat.Dense
        hiddenOutput.Mul(X, inputHiddenWeights.T())
        hiddenOutput.Apply(func(_, _ int, v float64) float64 {
            return sigmoid(v + hiddenBiases.At(0, 0))
        }, &hiddenOutput)

        var finalOutput mat.Dense
        finalOutput.Mul(&hiddenOutput, hiddenOutputWeights.T())
        finalOutput.Apply(func(_, _ int, v float64) float64 {
            return sigmoid(v + outputBiases.At(0, 0))
        }, &finalOutput)

        // Backpropagation
        var outputError mat.Dense
        outputError.Sub(Y, &finalOutput)

        var outputDelta mat.Dense
        outputDelta.MulElem(&outputError, finalOutput.Apply(sigmoidDerivative, finalOutput))

        var hiddenError mat.Dense
        hiddenError.Mul(&outputDelta, hiddenOutputWeights)

        var hiddenDelta mat.Dense
        hiddenDelta.MulElem(&hiddenError, hiddenOutput.Apply(sigmoidDerivative, hiddenOutput))

        // Update weights and biases
        hiddenOutputWeights.Add(hiddenOutputWeights, hiddenOutput.T().Mul(&outputDelta).Scale(learningRate))
        inputHiddenWeights.Add(inputHiddenWeights, X.T().Mul(&hiddenDelta).Scale(learningRate))
        outputBiases.Add(outputBiases, outputDelta.ColView(0).Sum().Scale(learningRate))
        hiddenBiases.Add(hiddenBiases, hiddenDelta.ColView(0).Sum().Scale(learningRate))
    }

    // Test the trained network
    var testInput mat.Dense
    testInput.Mul(X, inputHiddenWeights.T())
    testInput.Apply(func(_, _ int, v float64) float64 {
        return sigmoid(v + hiddenBiases.At(0, 0))
    }, &testInput)

    var testOutput mat.Dense
    testOutput.Mul(&testInput, hiddenOutputWeights.T())
    testOutput.Apply(func(_, _ int, v float64) float64 {
        return sigmoid(v + outputBiases.At(0, 0))
    }, &testOutput)

    fmt.Println("Predicted output:")
    mat.Print(testOutput)
}
