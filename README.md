# Neural Networks: From Perceptron to Deep Learning

A hands-on exploration of neural network fundamentals, from single neurons to multi-layer architectures. This project implements three models of increasing complexity:

| Model | Purpose | Architecture | Task |
|-------|---------|--------------|------|
| **Single-Layer Perceptron (SLP)** | Learn basic neuron | 3 inputs → 1 output | Binary classification (AND-like) |
| **Multi-Layer Perceptron (MLP)** | Overcome linearity limits | 2 inputs → 3 hidden → 1 output | XOR (non-linear problem) |
| **Mini Model (MNIST)** | Real-world classification | 784 inputs → 32 hidden → 10 outputs | Handwritten digit recognition (28×28px images) |

Each model teaches core concepts: weight initialization, backpropagation, activation functions, and learning rate scheduling.

## Quick Start

```bash
# Full setup: download MNIST data + build all binaries
$ ./setup.sh

# Or step by step:
$ ./setup.sh --data     # download & convert MNIST CSVs
$ ./setup.sh --build    # compile all three binaries
$ ./setup.sh --train    # train mini_model on mnist_train.csv (≈1-2 min)
$ ./setup.sh --test     # run smoke tests on all models
```

## Key Features

**Educational Models:**
- Simple, readable C implementations (~500-700 lines each)
- Clear backpropagation and forward pass logic
- Demonstrates gradient descent, momentum concepts

**Performance Improvements:**
- Xavier/He weight initialization (not naive uniform)
- Fisher-Yates shuffling per epoch
- Time-based learning rate decay
- MSE loss logging for convergence tracking

**Real-World Validation:**
- Mini Model: **96.68% accuracy on MNIST test set (10K samples)**
- Confusion matrix and per-class metrics
- Model card display (like HuggingFace)

---

## Model 1: Single-Layer Perceptron (SLP)

A single-layer perceptron can learn linear relationships between inputs and outputs. For example, with 3 input features and 1 output, it forms a model like:

```bash
output = activation(w1·x1 + w2·x2 + w3·x3 + bias)
```
It works well when the data is linearly separable (i.e., can be split with a straight line or hyperplane). If not, it fails, as seen in classic problems like XOR.

This minimal SLP is a great educational tool and a stepping stone to more powerful models.

## Model 2: Multi-Layer Perceptron (MLP)
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

| # | Change | Impact | Files |
|---|--------|--------|-------|
| 1 | **Xavier/He weight initialization** | Faster convergence, stable training | all |
| 2 | **Fisher-Yates shuffle per epoch** | Prevents gradient bias, better generalization | all |
| 3 | **Time-based learning rate decay** | Fine-tuning in later epochs | all |
| 4 | **MSE loss logging** | Visible convergence tracking | SLP, MLP |
| 5 | **Memory safety** | Prevents crash in info mode | mini_model |
| 6 | **Refactored forward pass** | DRY principle, easier maintenance | mini_model |
| 7 | **`eval` command** | Batch accuracy + per-class metrics | mini_model |
| 8 | **Model card display** | LLM-style test output | mini_model |
| 9 | **Confusion matrix** | Visual error analysis | mini_model |
| 10 | **Increased model capacity** | 96.68% accuracy (vs 94.07%) | mini_model |

## Building Manually

If you prefer not to use `setup.sh`, compile each binary directly:

```bash
$ gcc -Wall -O2 single-layer-perceptron.c -o slp -lm
$ gcc -Wall -O2 multi-layer-perceptron.c  -o mlp -lm
$ gcc -Wall -O2 mini_model.c              -o mini_model -lm
```

## Testing Locally

Once built, you can run each model:

```bash
# Model 1: Single-Layer Perceptron (learns AND-like pattern)
$ ./slp

# Model 2: Multi-Layer Perceptron (learns XOR)
$ ./mlp

# Model 3: MNIST (requires setup.sh --data first)
$ ./mini_model                    # train
$ ./mini_model test mnist_test.csv 50     # test 50 samples
$ ./mini_model eval mnist_test.csv        # full evaluation
$ ./mini_model info                       # show architecture
```

---

## Implementation Details

## Model 3: MNIST Digit Recognition (Mini Model)

### Overview

A practical neural network that classifies handwritten digits (0-9) from the MNIST dataset. The model learns to extract visual features from 28×28 pixel grayscale images and output a 10-dimensional probability distribution.

**Performance:** 96.68% accuracy on 10,000 test samples | 25,450 parameters (784→32→10)

### Usage

**Train the model** on 60,000 MNIST training images:

```bash
$ ./mini_model
Training on 60000 samples.... Number of iteration = 50
Iteration....0 (lr=0.001000)
Iteration....1 (lr=0.000999)
...
Iteration....49 (lr=0.000980)
Training complete. Saving model to model.bin
```

**Test on a sample** of N images and see confusion matrix:

```bash
$ ./mini_model test mnist_test.csv 100

╔════════════════════════════════════════════════════════════════╗
║                    MINI MODEL — Test Report                    ║
╠════════════════════════════════════════════════════════════════╣
║ Architecture:  784 → 32 → 10  (Fully Connected, ReLU+Softmax)  ║
║ Parameters:    25450 total  (W1:25088 b1:32 | W2:320 b2:10)   ║
║ Training:      50 iterations, LR=0.0010 (with decay)           ║
║ Dataset:       MNIST (28×784px grayscale images, 10 classes)   ║
║ Inference:     100 test samples from mnist_test.csv            ║
╚════════════════════════════════════════════════════════════════╝

Testing 100 sample(s)...

  Sample    0 | actual=7  predicted=7  OK   (100.0%)
  Sample    1 | actual=2  predicted=2  OK   (100.0%)
  ...
  
  Confusion Matrix (rows=actual, cols=predicted):
  actual \ pred      0    1    2    3    4    5    6    7    8    9
  class 0            8    .    .    .    .    .    .    .    .    .
  class 1            .    9    .    .    .    .    .    .    .    .
  ...

╔════════════════════════════════════════════════════════════════╗
║                      Test Summary                              ║
╠════════════════════════════════════════════════════════════════╣
║ Correct:       98 / 100  (98.00%)                              ║
║ Errors:        2 / 100  (2.00%)                                ║
║ Weakest class: 7 (92.00%)                                      ║
╚════════════════════════════════════════════════════════════════╝
```

**Batch accuracy evaluation** on entire test/train set:

```bash
$ ./mini_model eval mnist_test.csv

Evaluating model on 10000 samples from mnist_test.csv...

Overall accuracy: 9668 / 10000 = 96.68%

Per-class accuracy:
  Class 0:  964 /  980 = 98.37%
  Class 1: 1124 / 1135 = 99.03%
  Class 2: 1003 / 1032 = 97.19%
  ...
  Class 9:  966 / 1009 = 95.74%
```

**Model info** (architecture and parameter breakdown):

```bash
$ ./mini_model info

Loading model from file model.bin...
Model successfully loaded from model.bin
Model parameters from model.bin:
Input Size: 784
Hidden Units: 32
Output Classes: 10
Training Iterations (default): 50
Learning Rate (default): 0.0010

Parameter Counts:
  W1 (Input -> Hidden): 25088
  b1 (Hidden biases):      32
  W2 (Hidden-> Output):    320
  b2 (Output biases):       10
  TOTAL Parameters:      25450
```

---
