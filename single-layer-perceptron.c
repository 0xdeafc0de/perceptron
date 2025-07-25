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

// calculate neuron output for a given set of input
double calc_output(Neuron *n, double *inp, int n_inp) {
    double z = 0.0;
    int i;
    for (i = 0; i < n_inp; i++) {
        z += n->weights[i] * inp[i];
    }
    z += n->bias;
    n->output = sigmoid(z);
    return n->output;
}

///////// Training function////////
// Inputs
// n   ->   pointer to neuron/perceptron
// inp ->   pointer to input features
// n_inp -> number of input features
// tgt ->   target value
// lr  ->   learning rate
void train_perceptron(Neuron *n, double *inp, int n_inp, double tgt, double lr) {
    double z = 0.0;
    int i;
    // op = x1.w1 + x2.w2 + ... + b
    for (i = 0; i < n_inp; i++) {
        z += n->weights[i] * inp[i];
    }
    z += n->bias;

    double pred = sigmoid(z);
    double err = tgt - pred;
    double delta = err * sigmoid_derivative(z); //gradient

    // adjust
    for (i = 0; i < n_inp; i++) {
        n->weights[i] += lr * delta * inp[i];
    }
    n->bias += lr * delta;
}

void print_neuron(Neuron *n) {
    printf("weights: [%.3f, %.3f, %.3f], Bias: %.3f\n",
        n->weights[0], n->weights[1], n->weights[2], n->bias);
}

int main() {
    srand(time(NULL));

    Neuron neuron;
    init_perceptron(&neuron);

    // Sample dataset: [x1, x2, x3], target
    // Goal -> Learn simple pattern like x1 AND x2 (ignoring x3)
    double inputs[5][3] = {
        {0, 0, 0},  // expect 0
        {0, 1, 1},  // expect 0
        {1, 0, 1},  // expect 0
        {1, 1, 0},  // expect 1
        {1, 1, 1}   // expect 1
    };
    double targets[5] = {0, 0, 0, 1, 1};

    int num_epoch = 10 * 1000;
    double lr = 0.1; // learning rate

    // Training loop
    printf("Training ...\n");
    int epoch;
    for (epoch = 0; epoch < num_epoch; epoch++) {
        int i;
        for (i = 0; i < 5; i++) {
            train_perceptron(&neuron, inputs[i], 3, targets[i], lr);
        }
    }
    printf("Training completed!\n");

    // output trained weights and bias
    printf("Trained neuron - ");
    print_neuron(&neuron);

    // Testing
    printf("Testing ...\n");
    int i;
    for (i = 0; i < 5; i++) {
            double out = calc_output(&neuron, inputs[i], 3);
            printf("Input: [%.1f, %.1f, %.1f] => Predicted: %.3f (Expected: %.1f)\n",
                inputs[i][0], inputs[i][1], inputs[i][2], out, targets[i]);
    }

    return 0;
}
