# Perceptron
A perceptron is a fundamental building block of artificial intelligence and machine learning. Think of it as a simplified model of a neuron in our brain. It takes inputs, multiplies them by their respective weights, adds them up, and applies a threshold to determine its output.

This output is then compared to the desired output (target), and the perceptron adjusts its weights using a process called gradient descent to minimize the difference between the actual and desired outputs. In essence, the perceptron learns from data to make predictions or classify new examples, just as we, humans, learn to make decisions in our daily lives.

## Single-layer perceptron
A single-layer perceptron can learn linear relationships between inputs and a single output. With 3 input signals, it would have 3 input nodes, and one output node. It is suitable for tasks where the data can be separated by a straight line (or hyperplane in higher dimensions). If the data is not linearly separable, a more complex network structure is needed. A multi-layer perceptron (MLP) can learn complex, non-linear relationships. It consists of multiple layers of neurons, including hidden layers, which allow it to model intricate patterns.

This, minimal single-layer perceptron example is a great fit for educating ML from ground-up and serves as a foundation for more complex models.

## Multi-layer perceptron
### Non-Linearity
A linear method, like a single perceptron or logistic regression, tries to solve problems by separating data with a single straight line (or a flat plane in more than two dimensions). However, the non-linerar problems can't be solved with a use of single-layer perceptron.
e.g. XOR
```bash
Input1   Input2   Output
-----    ------   ------
0        0        0
0        1        1
1        0        1
1        1        0

  Y-axis
  1 |  X         O
    |
  0 |  O         X
    +---------------
      0         1
        X-axis
```
You can clearly see the linear challenge. Try to draw one single straight line to put both Xs on one side and both Os on the other.

-    A horizontal line? Fails.
-    A vertical line? Fails.
-    A diagonal line? Fails.

It's impossible. This property is called being "not linearly separable."

A linear model's entire worldview is based on finding that one perfect line. Its mathematical formula is effectively the equation of a line (w1*x1 + w2*x2 + b = 0). Since no such line exists for the XOR problem, the model is fundamentally incapable of solving it. It will try its best, but its error rate will never go to zero because its core assumption (that a line can solve the problem) is wrong.

### Once can ask, why XOR is Simple in Digital Computing?
Digital computing and hardware design don't "learn" from data. They implement explicit logical rules using physical components (transistors).

The reason XOR is simple in digital is that it isn't a fundamental, indivisible operation. It is constructed from other, simpler gates that are fundamental.

The Boolean logic for A XOR B can be expressed using simpler AND, OR, and NOT gates:

A XOR B = (A OR B) AND (NOT (A AND B))

To break this down:
-    Gate 1 (OR): Calculate A OR B. This is simple.
-    Gate 2 (NAND): Calculate NOT (A AND B). This is also simple.
-    Gate 3 (AND): Take the results of the first two gates and AND them together.

### Why can't we do the same for perceptron based learning?
Yes, we can do. Thats called Multi-Level Perceptron (MLP).
The reason an MLP can solve XOR is that its hidden layer effectively learns to become the simpler logic gates needed to build XOR.

When you train an MLP on the XOR data:

-    Hidden Neuron 1 might learn to fire like an OR gate.
-    Hidden Neuron 2 might learn to fire like a NAND (NOT AND) gate.
-    The Output Neuron then learns to AND the signals from those two hidden neurons.

The MLP, through the process of backpropagation, discovers on its own that the best way to solve the problem is to decompose it into simpler, linearly separable parts, mimicking the very same logic used by a digital logic. It learns to create a non-linear decision boundary by combining multiple linear boundaries.


## Key Features
- Represents a single neuron with 3 inputs
- Trains using a simplified rule (gradient update with full error propagation)
- Uses sigmod activation function
- Trained on synthetic training data and test evaluation
- Goal for slp - To learn 'x1 AND x2 (ignoring x3)'
- Goal for mlp - To learn 'x1 XOR x2'

## How to build
gcc single-layer-perceptron.c -o slp -lm
gcc multi-layer-perceptron.c  -o mlp -lm

## Example run
```bash
./slp
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
