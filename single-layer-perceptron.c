#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_FEATURES 3 // number of input features

// A Neuron
typedef struct {
    double weights[N_FEATURES]; // Input features {x1, x2, x3}
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

// Initialize perceptron weights using Xavier uniform init (for sigmoid activation)
// Xavier: weights ~ Uniform[-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out))]
void init_perceptron(Neuron *n) {
    int i;
    double limit = sqrt(6.0 / (N_FEATURES + 1)); // fan_in=N_FEATURES, fan_out=1 output
    for (i = 0; i < N_FEATURES; i++) {
        n->weights[i] = ((double)rand() / RAND_MAX) * 2.0 * limit - limit;
    }
    n->bias = 0.0; // biases initialized to zero
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
    int i;
    printf("weights: [");
    for (i = 0; i < N_FEATURES; i++) {
        printf("%.3f%s", n->weights[i], i < N_FEATURES - 1 ? ", " : "");
    }
    printf("], Bias: %.3f\n", n->bias);
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
    double lr_initial = 0.1; // initial learning rate
    double lr_decay   = 1e-4; // time-based decay: lr = lr_initial / (1 + decay * epoch)

    // Training loop
    printf("Training ...\n");
    int indices[5] = {0, 1, 2, 3, 4};
    int epoch;
    for (epoch = 0; epoch < num_epoch; epoch++) {
        double lr = lr_initial / (1.0 + lr_decay * epoch); // decayed learning rate
        // Shuffle training order each epoch (Fisher-Yates)
        int i;
        for (i = 4; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }
        for (i = 0; i < 5; i++) {
            train_perceptron(&neuron, inputs[indices[i]], N_FEATURES, targets[indices[i]], lr);
        }
        // Log MSE every 1000 epochs
        if ((epoch + 1) % 1000 == 0) {
            double mse = 0.0;
            int k;
            for (k = 0; k < 5; k++) {
                double out = calc_output(&neuron, inputs[k], N_FEATURES);
                double err = targets[k] - out;
                mse += err * err;
            }
            printf("Epoch %d/%d, MSE: %.6f\n", epoch + 1, num_epoch, mse / 5);
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
            double out = calc_output(&neuron, inputs[i], N_FEATURES);
            printf("Input: [%.1f, %.1f, %.1f] => Predicted: %.3f (Expected: %.1f)\n",
                inputs[i][0], inputs[i][1], inputs[i][2], out, targets[i]);
    }

    return 0;
}
