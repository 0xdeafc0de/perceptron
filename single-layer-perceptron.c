#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// A Neuron
typedef struct {
    double weights[3]; // 3 Input  features {x1, x2, x3}
    double bias;
    double output;     // current output value
} Neuron;

// Sigmoid activation function
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// Derivative of sigmoid - used in learning
double sigmoid_derivative(double z) {
    double s = sigmoid(z);
    return s * (1 - s);
}

// Initialize perceptron weights and bias randomly
void init_perceptron(Neuron *n) {
    int i;
    for (i = 0; i < 3; i++) {
        n->weights[i] = ((double)rand() / RAND_MAX) * 2 - 1; // [-1, 1]
    }
    n->bias = ((double)rand() / RAND_MAX) * 2 - 1;
}

