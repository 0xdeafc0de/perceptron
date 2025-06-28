#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Training configuration
// Set the number of iterations
#define NUM_ITERATIONS 10
// Set the base learning rate
#define LEARNING_RATE 0.001

// Size of hidden layer
#define HIDDEN_UNITS 15

#define INPUT_SIZE 784
#define NUM_CLASSES 10
#define MAX_SAMPLES 60000
#define MODEL_FILE "model.bin"
#define DEF_TRAINING_FILE "mnist_train.csv"
#define DEF_TESTING_FILE  "mnist_test.csv"

// Model Definition
typedef struct {
    double W1[INPUT_SIZE][HIDDEN_UNITS];  // Input-to-hidden weights
    double b1[HIDDEN_UNITS];              // Hidden layer biases
    double W2[HIDDEN_UNITS][NUM_CLASSES]; // Hidden-to-output weights
    double b2[NUM_CLASSES];               // Output layer biases
} MiniModel;

// ReLU activation and its derivative
double relu(double x) { return x > 0 ? x : 0; }
double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }

// Applies softmax activation to logits for multi-class probability output
void softmax(double* input, int size, double* output) {
    double max = input[0];
    for (int i = 1; i < size; i++)
        if (input[i] > max) max = input[i];

    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        double val = exp(input[i] - max);
        if (isnan(val) || isinf(val)) val = 1e-9;  // clamp if needed
        output[i] = val;
        sum += val;
    }

    if (sum == 0.0) sum = 1e-9;  // avoid division by zero

    for (int i = 0; i < size; i++)
        output[i] /= sum;
}

// Generates a random weight in range [-1, 1]
double rand_weight() {
    return ((double)rand() / RAND_MAX) * 0.1 - 0.05;
}

// Serializes model to file
void save_model(MiniModel* m, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) { perror("fopen"); exit(1); }
    fwrite(m, sizeof(MiniModel), 1, f);
    fclose(f);
}

// Deserializes model from file
MiniModel* load_model(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        return NULL;
    }

    MiniModel* m = (MiniModel*)malloc(sizeof(MiniModel));
    fread(m, sizeof(MiniModel), 1, f);
    fclose(f);

    return m;
}

// Create and initialize a new MiniModel with random weights
MiniModel* NN() {
    MiniModel* m = (MiniModel*)malloc(sizeof(MiniModel));
    for (int i = 0; i < INPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_UNITS; j++)
            m->W1[i][j] = rand_weight();

    for (int j = 0; j < HIDDEN_UNITS; j++)
        m->b1[j] = rand_weight();

    for (int i = 0; i < HIDDEN_UNITS; i++)
        for (int j = 0; j < NUM_CLASSES; j++)
            m->W2[i][j] = rand_weight();

    for (int j = 0; j < NUM_CLASSES; j++)
        m->b2[j] = rand_weight();

    return m;
}

// Forward pass of the network: computes probabilities from input image
void forward(MiniModel* m, unsigned char input[INPUT_SIZE], double* out_probs, double* hidden_out) {
    for (int j = 0; j < HIDDEN_UNITS; j++) {
        hidden_out[j] = m->b1[j];
        for (int i = 0; i < INPUT_SIZE; i++)
            hidden_out[j] += input[i] / 255.0 * m->W1[i][j];
        hidden_out[j] = relu(hidden_out[j]);
    }

    double logits[NUM_CLASSES];
    for (int j = 0; j < NUM_CLASSES; j++) {
        logits[j] = m->b2[j];
        for (int i = 0; i < HIDDEN_UNITS; i++)
            logits[j] += hidden_out[i] * m->W2[i][j];
    }
    softmax(logits, NUM_CLASSES, out_probs);
}

// Trains the model using cross-entropy loss and gradient descent
void train(MiniModel* m, unsigned char** X, int Y[], int samples, int iterations, double alpha) {
    for (int iter = 0; iter < iterations; iter++) {
        //printf("\rIteration....%d", iter);
        printf("Iteration....%d\n", iter);
        fflush(stdout);
        for (int n = 0; n < samples; n++) {
            unsigned char* input = X[n];
            int label = Y[n];

            // Forward pass (hidden layer)
            double hidden[HIDDEN_UNITS];
            double d_hidden[HIDDEN_UNITS];
            for (int j = 0; j < HIDDEN_UNITS; j++) {
                hidden[j] = m->b1[j];
                for (int i = 0; i < INPUT_SIZE; i++)
                    hidden[j] += input[i] / 255.0 * m->W1[i][j];
                hidden[j] = relu(hidden[j]);
            }

            // Forward pass (output layer)
            double logits[NUM_CLASSES];
            for (int j = 0; j < NUM_CLASSES; j++) {
                logits[j] = m->b2[j];
                for (int i = 0; i < HIDDEN_UNITS; i++)
                    logits[j] += hidden[i] * m->W2[i][j];
            }

            double probs[NUM_CLASSES];
            softmax(logits, NUM_CLASSES, probs);

            // Gradient of loss w.r.t logits
            double d_logits[NUM_CLASSES];
            for (int j = 0; j < NUM_CLASSES; j++)
                d_logits[j] = probs[j] - (j == label ? 1.0 : 0.0);

            // Backpropagation: output to hidden
            for (int i = 0; i < HIDDEN_UNITS; i++)
                for (int j = 0; j < NUM_CLASSES; j++)
                    m->W2[i][j] -= alpha * d_logits[j] * hidden[i];
            for (int j = 0; j < NUM_CLASSES; j++)
                m->b2[j] -= alpha * d_logits[j];

            // Backpropagation: hidden to input
            for (int i = 0; i < HIDDEN_UNITS; i++) {
                double grad = 0.0;
                for (int j = 0; j < NUM_CLASSES; j++)
                    grad += d_logits[j] * m->W2[i][j];
                d_hidden[i] = grad * relu_derivative(hidden[i]);
            }

            for (int i = 0; i < INPUT_SIZE; i++)
                for (int j = 0; j < HIDDEN_UNITS; j++)
                    m->W1[i][j] -= alpha * d_hidden[j] * input[i] / 255.0;
            for (int j = 0; j < HIDDEN_UNITS; j++)
                m->b1[j] -= alpha * d_hidden[j];
        }
    }
}

// Loads MNIST CSV data (label + pixels) into arrays
int load_csv(const char* filename, unsigned char*** X_ptr, int Y[], int max_samples) {
    FILE* f = fopen(filename, "r");
    if (!f) { perror("fopen"); return 0; }

    unsigned char** X = (unsigned char**)malloc(max_samples * sizeof(unsigned char*));
    for (int i = 0; i < max_samples; i++)
        X[i] = (unsigned char*)malloc(INPUT_SIZE);

    int samples = 0;
    char line[8192];
    while (fgets(line, sizeof(line), f) && samples < max_samples) {
        char* tok = strtok(line, ",");
        if (!tok) continue;
        Y[samples] = atoi(tok);
        for (int i = 0; i < INPUT_SIZE; i++) {
            tok = strtok(NULL, ",");
            if (!tok) break;
            X[samples][i] = (unsigned char)atoi(tok);
        }
        samples++;
    }
    fclose(f);
    *X_ptr = X;
    return samples;
}

void print_model_parameters(MiniModel *model) {
    printf("Model paramers from %s:\n", MODEL_FILE);
    printf("Input Size: %d\n", INPUT_SIZE);
    printf("Hidden Units: %d\n", HIDDEN_UNITS);
    printf("Output Classes: %d\n", NUM_CLASSES);
    printf("Training Iterations (default): %d\n", NUM_ITERATIONS);
    printf("Learning Rate (default): %.4f\n", LEARNING_RATE);

#if 0
    printf("\nW1 (Input -> Hidden):\n");
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_UNITS; j++) {
            printf("%.4f ", model->W1[i][j]);
        }
    }
    printf("\n");
#endif
    int w1_params = (INPUT_SIZE * HIDDEN_UNITS);
    int b1_params = HIDDEN_UNITS;
    int w2_params = (HIDDEN_UNITS * NUM_CLASSES);
    int b2_params = NUM_CLASSES;
    int total_params = w1_params + b1_params + w2_params + b2_params;
    printf("\nParameter Counts:\n");
    printf("  W1 (Input -> Hidden): %d\n", w1_params);
    printf("  b1 (Hidden biases):   %d\n", b1_params);
    printf("  W2 (Hidden-> Output): %d\n", w2_params);
    printf("  b2 (Output biases):   %d\n", b2_params);
    printf("  TOTAL Parameters:     %d\n", total_params);
}

int main(int argc, char** argv) {
    srand(time(NULL));
    char* csv_file = NULL;
    int samples;

    if (argc == 1) {
        printf("No arguments provided... doing training on %s\n", DEF_TRAINING_FILE);
	printf("Other options - \n");
	printf("%s test [%s %s]\n", argv[0], "test_data_csv_file", "n");
	printf("\t\t where n is the row index on the csv file\n");
    }

    int test_index = -1;
    if (argc >= 3 && strcmp(argv[1], "test") == 0) {
        csv_file = argv[2];
        if (argc >= 4) test_index = atoi(argv[3]);
    }

    unsigned char** X;
    int Y[MAX_SAMPLES];

    MiniModel* model = NULL;
    int mode = 0; // 0 = train, 1 = test only, 2 = Info only

    if (argc >= 2) {
        if ((strcmp(argv[1], "test") == 0) || (strcmp(argv[1], "info") == 0)) {
            printf("Loading model from file %s...\n", MODEL_FILE);
            model = load_model(MODEL_FILE);
            if (!model) {
                fprintf(stderr, "Could not load model for testing.\n");
                return -1;
            }
            printf("Model successfully loaded from %s\n", MODEL_FILE);
            mode = 1;
            if (strcmp(argv[1], "info") == 0) {
                mode = 2;
                print_model_parameters(model);
            }
        }
    } else {
        // Load dataset
        if (csv_file == NULL) {
            csv_file = DEF_TRAINING_FILE;
	}
        samples = load_csv(csv_file, &X, Y, MAX_SAMPLES);
        if (samples == 0) {
            fprintf(stderr, "No data loaded.\n");
            return 1;
        }
        model = NN();
        printf("Training on %d samples.... Number of iteration = %d\n", samples, NUM_ITERATIONS);

        train(model, X, Y, samples, NUM_ITERATIONS, LEARNING_RATE);
        printf("Training complete. Saving model to %s\n\n", MODEL_FILE);
        save_model(model, MODEL_FILE);
    }

    // If in test mode, make prediction for a sample
    if (mode == 1) {
        if (test_index < 0) {
            test_index = 0; // use first row from data file
        }
        if (csv_file == NULL) {
            csv_file = DEF_TESTING_FILE;
	}

        samples = load_csv(csv_file, &X, Y, MAX_SAMPLES);
        if (test_index < 0 || test_index >= samples) {
            test_index = samples - 1;
        }
        printf("testing the model %s with test data from file %s at row %d\n",
            MODEL_FILE, csv_file, test_index);

        double out[NUM_CLASSES], h[HIDDEN_UNITS];
        forward(model, X[test_index], out, h);
        printf("\nPrediction for test sample %d (label=%d):\n", test_index, Y[test_index]);
        for (int i = 0; i < NUM_CLASSES; i++) {
            printf("Class %d: %.3f\n", i, out[i]);
        }
    }

    // Cleanup
    for (int i = 0; i < MAX_SAMPLES; i++) {
        free(X[i]);
    }
    free(X);
    free(model);

    return 0;
}
