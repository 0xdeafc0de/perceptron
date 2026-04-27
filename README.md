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
- Trains using a simplified rule (gradient update with full error propagation)
- Uses sigmoid activation function (SLP/MLP) and ReLU + softmax (mini model)
- Trained on synthetic training data and test evaluation
- Goal for slp - To learn 'x1 AND x2 (ignoring x3)'
- Goal for mlp - To learn 'x1 XOR x2'

## Enhancements
The following improvements were applied over the original student implementation:

| # | Change | Files |
|---|--------|-------|
| 1 | **Xavier/He weight initialization** — Xavier uniform for sigmoid layers, He uniform for ReLU layers. Replaces naive `[-1, 1]` uniform init which causes slow/unstable training. | all |
| 2 | **Fisher-Yates shuffle each epoch** — Randomises sample order before every training pass, preventing gradient bias from fixed ordering. | all |
| 3 | **MSE loss logging** — Reports mean squared error every 1000 epochs so convergence is visible during training. | SLP, MLP |
| 4 | **Time-based learning rate decay** — `lr = lr₀ / (1 + decay × epoch)`. Helps fine-tune weights in later epochs without manual tuning. | all |
| 5 | **`N_FEATURES` define** — Eliminates the mismatch between the hardcoded struct array size (`weights[3]`) and the runtime `n_inp` argument. | SLP |
| 6 | **Refactor duplicate forward pass** — `train()` called `forward()` instead of re-implementing the same logic inline. | mini_model |
| 7 | **Memory safety fix** — `X` initialised to `NULL`; cleanup guarded to prevent crash when running `info` mode (no data loaded). | mini_model |
| 8 | **`eval` command** — Batch accuracy evaluation over an entire CSV dataset with per-class breakdown. | mini_model |

## How to build

A `setup.sh` script handles downloading the MNIST dataset, converting it to CSV, and building all binaries.

```bash
# Full setup: download MNIST data + build all binaries
$ ./setup.sh

# Or step by step:
$ ./setup.sh --data     # download & convert MNIST CSVs
$ ./setup.sh --build    # compile all three binaries
$ ./setup.sh --train    # train mini_model on mnist_train.csv
$ ./setup.sh --test     # run smoke tests on all models
```

To build manually without the script:
```bash
$ gcc -Wall -O2 single-layer-perceptron.c -o slp -lm
$ gcc -Wall -O2 multi-layer-perceptron.c  -o mlp -lm
$ gcc -Wall -O2 mini_model.c              -o mini_model -lm
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
Running the binary will train the model on `mnist_train.csv` and save the weights to `model.bin`.

```bash
# Train (requires mnist_train.csv — run ./setup.sh --data first)
$ ./mini_model
Training on 60000 samples.... Number of iteration = 10
Iteration....0 (lr=0.001000)
Iteration....1 (lr=0.000999)
...
Iteration....9 (lr=0.000991)
Training complete. Saving model to model.bin
```

- **Single sample test** — predict the digit for a specific row in a CSV:

```bash
$ ./mini_model test mnist_test.csv 1001
Loading model from file model.bin...
Model successfully loaded from model.bin
testing the model model.bin with test data from file mnist_test.csv at row 1001

Prediction for test sample 1001 (label=0):
Class 0: 0.996
Class 1: 0.000
...
```

- **Batch accuracy evaluation** — run against an entire dataset:

```bash
$ ./mini_model eval mnist_test.csv
Loading model from file model.bin...
Model successfully loaded from model.bin
Evaluating model on 10000 samples from mnist_test.csv...

Overall accuracy: 9407 / 10000 = 94.07%

Per-class accuracy:
  Class 0:  969 /  980 = 98.88%
  Class 1: 1115 / 1135 = 98.24%
  Class 2:  941 / 1032 = 91.18%
  Class 3:  937 / 1010 = 92.77%
  Class 4:  914 /  982 = 93.08%
  Class 5:  826 /  892 = 92.60%
  Class 6:  903 /  958 = 94.26%
  Class 7:  916 / 1028 = 89.11%
  Class 8:  923 /  974 = 94.76%
  Class 9:  963 / 1009 = 95.44%
```

> **94.07% accuracy** on 10,000 test samples with only 11,935 parameters (784→15→10 network).

- **Model info**:

```bash
$ ./mini_model info
Model parameters from model.bin:
Input Size: 784
Hidden Units: 15
Output Classes: 10
Total Parameters: 11935
```
