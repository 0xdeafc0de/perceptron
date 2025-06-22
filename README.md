# Perceptron & a Mini Model for machine learning
A perceptron is a fundamental building block of artificial intelligence and machine learning. Think of it as a simplified model of a neuron in the brain. It:
    - Takes multiple inputs
    - Multiplies them by their respective weights
    - Adds them up
    - Applies an activation function (like a threshold or sigmoid) to produce an output

The output is compared with the target, and the perceptron adjusts its weights using a process like gradient descent to minimize the error. This is how the perceptron "learns" to make predictions, much like how humans improve decision-making through feedback.


## Single-layer perceptron (SLP)
A single-layer perceptron can learn linear relationships between inputs and outputs. For example, with 3 input features and 1 output, it forms a model like:

```bash
output = activation(w1·x1 + w2·x2 + w3·x3 + bias)
```
It works well when the data is linearly separable (i.e., can be split with a straight line or hyperplane). If not, it fails, as seen in classic problems like XOR.

This minimal SLP is a great educational tool and a stepping stone to more powerful models.

## Multi-layer perceptron & Non-Linearity
Linear models like logistic regression or a single perceptron can only separate data with straight lines. But real-world problems often require non-linear decision boundaries, like the XOR problem.
Here is XOR truth table
```bash
Input1   Input2   Output
-----    ------   ------
0        0        0
0        1        1
1        0        1
1        1        0
```
If you try to separate the 1s and 0s using a single line (in 2D), you'll fail:

```bash
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

### But XOR is Simple in Digital logic. Why?
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
- Represents multi-layer model for XOR learning (with 1 hidden layer).
- Trains using a simplified rule (gradient update with full error prop`agation)
- Uses sigmod activation function
- Trained on synthetic training data and test evaluation
- Goal for slp - To learn 'x1 AND x2 (ignoring x3)'
- Goal for mlp - To learn 'x1 XOR x2'

## How to build
```bash
$ gcc single-layer-perceptron.c -o slp -lm
$ gcc multi-layer-perceptron.c  -o mlp -lm
```

## Example run
```bash
$ ./slp
Training ...
Training completed!
Trained neuron - weights: [5.875, 5.875, -3.996], Bias: -4.940
Testing ...
Input: [0.0, 0.0, 0.0] => Predicted: 0.007 (Expected: 0.0)
Input: [0.0, 1.0, 1.0] => Predicted: 0.045 (Expected: 0.0)
Input: [1.0, 0.0, 1.0] => Predicted: 0.045 (Expected: 0.0)
Input: [1.0, 1.0, 0.0] => Predicted: 0.999 (Expected: 1.0)
Input: [1.0, 1.0, 1.0] => Predicted: 0.943 (Expected: 1.0

$ ./mlp 
Epoch 1000/15000, MSE: 0.212942
Epoch 2000/15000, MSE: 0.004943
Epoch 3000/15000, MSE: 0.001831
Epoch 4000/15000, MSE: 0.001095
Epoch 5000/15000, MSE: 0.000774
Epoch 6000/15000, MSE: 0.000596
Epoch 7000/15000, MSE: 0.000484
Epoch 8000/15000, MSE: 0.000406
Epoch 9000/15000, MSE: 0.000350
Epoch 10000/15000, MSE: 0.000307
Epoch 11000/15000, MSE: 0.000273
Epoch 12000/15000, MSE: 0.000246
Epoch 13000/15000, MSE: 0.000224
Epoch 14000/15000, MSE: 0.000205
Epoch 15000/15000, MSE: 0.000190

--- Testing Trained MLP on XOR data ---
Input | Target | Prediction
----------------------------------
0.0, 0.0 |  0.0   | 0.0193
0.0, 1.0 |  1.0   | 0.9871
1.0, 0.0 |  1.0   | 0.9868
1.0, 1.0 |  0.0   | 0.0068

```

## Mini Model
### mnist database
Our training data consists of total 60000 images of handwritten digits in 28 by 28 pixel format. Each row represents 1 such image with total 789 comumns, first being the target variable andi the remaining 784 (28x28) are pixel values. The input pixels are greyscale, with a value of 0.0 representing white, a value of 1.0 representing black, and in between values representing gradually darkening shades of grey.


### Model
The model is minimal and has just 3 layers - input, hidden and output. The input layer contains neurons encoding the values of the input pixels.
As our training data consists of 28 by 28 pixel images, so the input layer contains 784=28×28 neurons.

The second layer of the network is a hidden layer and contains just 15 neurons.

The output layer of the network contains 10 neurons. If the first neuron fires, i.e., has an output ≈1
, then that will indicate that the network thinks the digit is a 0. If the second neuron fires then that will indicate that the network thinks the digit is a 1. And so on.

### model diagram
<img src="https://github.com/user-attachments/assets/1bb510c8-4254-49aa-9868-499f842a9226">

_For simplicity not all 728 inputs are shown.
_

### Run
Running the binary will train the model and save it in a file.
```bash
$ ./mini_model
Training on 60000 samples.... Number of iteration = 10
Iteration....0
Iteration....1
Iteration....2
Iteration....3
Iteration....4
Iteration....5
Iteration....6
Iteration....7
Iteration....8
Iteration....9
Training complete. Saving model to model.bin
```

- Test Example - test 1001th row from the mnist_test.csv file
- The test uses the earlier trained (and saved) model.

```bash
$ ./mini_model test mnist_test.csv 1001
Model successfully loaded from model.bin
Doing some testing

Prediction for test sample 1001 (label=0):
Class 0: 0.996
Class 1: 0.000
Class 2: 0.000
Class 3: 0.000
Class 4: 0.000
Class 5: 0.004
Class 6: 0.000
Class 7: 0.000
Class 8: 0.000
Class 9: 0.000
```
