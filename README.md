# Perceptron
A perceptron is a fundamental building block of artificial intelligence and machine learning. Think of it as a simplified model of a neuron in our brain. It takes inputs, multiplies them by their respective weights, adds them up, and applies a threshold to determine its output.

This output is then compared to the desired output (target), and the perceptron adjusts its weights using a process called gradient descent to minimize the difference between the actual and desired outputs. In essence, the perceptron learns from data to make predictions or classify new examples, just as we, humans, learn to make decisions in our daily lives.

## single-layer-perceptron
A single-layer perceptron can learn linear relationships between inputs and a single output. With 3 input signals, it would have 3 input nodes, and one output node. It is suitable for tasks where the data can be separated by a straight line (or hyperplane in higher dimensions). If the data is not linearly separable, a more complex network structure is needed. A multi-layer perceptron (MLP) can learn complex, non-linear relationships. It consists of multiple layers of neurons, including hidden layers, which allow it to model intricate patterns.

This, minimal single-layer perceptron example is a great fit for educating ML from ground-up and serves as a foundation for more complex models.

## Key Features
- Represents a single neuron with 3 inputs
- Trains using a simplified rule (gradient update without full error propagation)
- Uses sigmod activation function
- Trained on synthetic training data and test evaluation
- Goal - To learn 'x1 AND x2 (ignoring x3)'

## How to build
gcc perceptron.c -o perceptron -lm

## Example run
```bash
./perceptron
Training ...
Training completed!
Trained neuron - weights: [5.875, 5.875, -3.996], Bias: -4.940
Testing ...
Input: [0.0, 0.0, 0.0] => Predicted: 0.007 (Expected: 0.0)
Input: [0.0, 1.0, 1.0] => Predicted: 0.045 (Expected: 0.0)
Input: [1.0, 0.0, 1.0] => Predicted: 0.045 (Expected: 0.0)
Input: [1.0, 1.0, 0.0] => Predicted: 0.999 (Expected: 1.0)
Input: [1.0, 1.0, 1.0] => Predicted: 0.943 (Expected: 1.0
```
