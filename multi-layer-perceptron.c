#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// --- Configuration ---
#define NUM_INPUTS 2
#define NUM_HIDDEN_NEURONS 3 // 2 is sufficient, 3-4 can be more stable
#define NUM_OUTPUTS 1
#define NUM_TRAINING_SAMPLES 4
#define EPOCHS 15000
#define LEARNING_RATE 0.5

// --- Structures ---
typedef struct {
    double weights_ih[NUM_INPUTS][NUM_HIDDEN_NEURONS];
    double bias_h[NUM_HIDDEN_NEURONS];
    double weights_ho[NUM_HIDDEN_NEURONS][NUM_OUTPUTS];
    double bias_o[NUM_OUTPUTS];
} MLP;

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Initialize MLP with random weights and biases
void init_mlp(MLP *mlp) {
    srand(time(NULL));
    for (int i = 0; i < NUM_INPUTS; i++) {
        for (int j = 0; j < NUM_HIDDEN_NEURONS; j++) {
            mlp->weights_ih[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
    for (int i = 0; i < NUM_HIDDEN_NEURONS; i++) {
        mlp->bias_h[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            mlp->weights_ho[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        mlp->bias_o[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}


void forward_pass(MLP *mlp, const double inputs[], double *hidden_activations, double *final_output) {
    // Hidden Layer
    for (int j = 0; j < NUM_HIDDEN_NEURONS; j++) {
        double weighted_sum = mlp->bias_h[j];
        for (int i = 0; i < NUM_INPUTS; i++) {
            weighted_sum += inputs[i] * mlp->weights_ih[i][j];
        }
        hidden_activations[j] = sigmoid(weighted_sum);
    }

    // Output Layer
    for (int k = 0; k < NUM_OUTPUTS; k++) {
        double weighted_sum = mlp->bias_o[k];
        for (int j = 0; j < NUM_HIDDEN_NEURONS; j++) {
            weighted_sum += hidden_activations[j] * mlp->weights_ho[j][k];
        }
        final_output[k] = sigmoid(weighted_sum);
    }
}

void backward_pass(MLP *mlp, const double inputs[], double target, double *hidden_activations, double *final_output) {
    // 1. Calculate output layer error and delta
    double output_errors[NUM_OUTPUTS];
    double output_deltas[NUM_OUTPUTS];
    for (int k = 0; k < NUM_OUTPUTS; k++) {
        output_errors[k] = target - final_output[k];
        output_deltas[k] = output_errors[k] * sigmoid_derivative(final_output[k]);
    }

    // 2. Calculate hidden layer error and delta
    double hidden_errors[NUM_HIDDEN_NEURONS];
    double hidden_deltas[NUM_HIDDEN_NEURONS];
    for (int j = 0; j < NUM_HIDDEN_NEURONS; j++) {
        hidden_errors[j] = 0.0;
        for (int k = 0; k < NUM_OUTPUTS; k++) {
            hidden_errors[j] += output_deltas[k] * mlp->weights_ho[j][k];
        }
        hidden_deltas[j] = hidden_errors[j] * sigmoid_derivative(hidden_activations[j]);
    }

    // 3. Update weights and biases
    for (int k = 0; k < NUM_OUTPUTS; k++) {
        mlp->bias_o[k] += output_deltas[k] * LEARNING_RATE;
        for (int j = 0; j < NUM_HIDDEN_NEURONS; j++) {
            mlp->weights_ho[j][k] += hidden_activations[j] * output_deltas[k] * LEARNING_RATE;
        }
    }
    for (int j = 0; j < NUM_HIDDEN_NEURONS; j++) {
        mlp->bias_h[j] += hidden_deltas[j] * LEARNING_RATE;
        for (int i = 0; i < NUM_INPUTS; i++) {
            mlp->weights_ih[i][j] += inputs[i] * hidden_deltas[j] * LEARNING_RATE;
        }
    }
}


int main() {
    MLP mlp;
    init_mlp(&mlp);

    // XOR Training Data
    double training_inputs[NUM_TRAINING_SAMPLES][NUM_INPUTS] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    double training_targets[NUM_TRAINING_SAMPLES][NUM_OUTPUTS] = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // Training Loop
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_error = 0;
        for (int i = 0; i < NUM_TRAINING_SAMPLES; i++) {
            double *inputs = training_inputs[i];
            double target = training_targets[i][0];

            double hidden_activations[NUM_HIDDEN_NEURONS];
            double final_output[NUM_OUTPUTS];

            forward_pass(&mlp, inputs, hidden_activations, final_output);
            backward_pass(&mlp, inputs, target, hidden_activations, final_output);
            total_error += (target - final_output[0]) * (target - final_output[0]);
        }
        if ((epoch + 1) % 1000 == 0) {
            printf("Epoch %d/%d, MSE: %f\n", epoch + 1, EPOCHS, total_error / NUM_TRAINING_SAMPLES);
        }
    }

    // Testing the Trained Network
    printf("\n--- Testing Trained MLP on XOR data ---\n");
    printf("Input | Target | Prediction\n");
    printf("----------------------------------\n");
    for (int i = 0; i < NUM_TRAINING_SAMPLES; i++) {
        double *inputs = training_inputs[i];
        double target = training_targets[i][0];
        double hidden_activations[NUM_HIDDEN_NEURONS];
        double prediction[NUM_OUTPUTS];

        forward_pass(&mlp, inputs, hidden_activations, prediction);

        printf("%.1f, %.1f |  %.1f   | %.4f\n", inputs[0], inputs[1], target, prediction[0]);
    }

    return 0;
}
