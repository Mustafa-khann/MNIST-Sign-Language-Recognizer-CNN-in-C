#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cnn.h"
#include "utils.h"

#define NUM_CLASSES 24
#define LEARNING_RATE 0.0001
#define NUM_EPOCHS 50
#define BATCH_SIZE 32
#define TRAIN_SIZE 27455
#define TEST_SIZE 7172

// Function prototypes
float* forwardPass(ConvLayer* conv1, PoolingLayer* pool1, ConvLayer* conv2, PoolingLayer* pool2, FCLayer* fc1, FCLayer* fc2, float* input);
void backwardPass(ConvLayer* conv1, PoolingLayer* pool1, ConvLayer* conv2, PoolingLayer* pool2, FCLayer* fc1, FCLayer* fc2,
                  float* input, float* conv1Output, float* pool1Output, float* conv2Output, float* pool2Output, float* fc1Output, float* fc2Output,
                  int label, float learningRate);

int main() {
    srand(time(NULL));
    printf("Debug: Initialized random seed\n");


    // Load MNIST sign language dataset
    double** trainImages = (double**)malloc(TRAIN_SIZE * sizeof(double*));
    if (trainImages == NULL) {
        fprintf(stderr, "Failed to allocate memory for trainImages\n");
        exit(1);
    }
    printf("Debug: Allocated memory for trainImages\n");

    for (int i = 0; i < TRAIN_SIZE; i++) {
        trainImages[i] = (double*)malloc(784 * sizeof(double));
        if (trainImages[i] == NULL) {
            fprintf(stderr, "Failed to allocate memory for trainImages[%d]\n", i);
            exit(1);
        }
    }
    printf("Debug: Allocated memory for individual trainImages\n");

    int* trainLabels = (int*)malloc(TRAIN_SIZE * sizeof(int));
    if (trainLabels == NULL) {
        fprintf(stderr, "Failed to allocate memory for trainLabels\n");
        exit(1);
    }
    printf("Debug: Allocated memory for trainLabels\n");

    double** testImages = (double**)malloc(TEST_SIZE * sizeof(double*));
    if (testImages == NULL) {
        fprintf(stderr, "Failed to allocate memory for testImages\n");
        exit(1);
    }
    printf("Debug: Allocated memory for testImages\n");

    for (int i = 0; i < TEST_SIZE; i++) {
        testImages[i] = (double*)malloc(784 * sizeof(double));
        if (testImages[i] == NULL) {
            fprintf(stderr, "Failed to allocate memory for testImages[%d]\n", i);
            exit(1);
        }
    }
    printf("Debug: Allocated memory for individual testImages\n");

    int* testLabels = (int*)malloc(TEST_SIZE * sizeof(int));
    if (testLabels == NULL) {
        fprintf(stderr, "Failed to allocate memory for testLabels\n");
        exit(1);
    }
    printf("Debug: Allocated memory for testLabels\n");

    printf("Debug: About to load MNIST data\n");
    loadMNISTData("dataset/sign_mnist_train.csv", "dataset/sign_mnist_test.csv", trainImages, trainLabels, testImages, testLabels, TRAIN_SIZE, TEST_SIZE);
    printf("Debug: MNIST data loaded successfully\n");

    // Create CNN layers
    ConvLayer* conv1 = createConvLayer(28, 28, 1, 3, 32);
    PoolingLayer* pool1 = createPoolingLayer(26, 26, 32, 2);
    ConvLayer* conv2 = createConvLayer(13, 13, 32, 3, 64);
    PoolingLayer* pool2 = createPoolingLayer(11, 11, 64, 2);
    FCLayer* fc1 = createFCLayer(5 * 5 * 64, 128);
    FCLayer* fc2 = createFCLayer(128, NUM_CLASSES);

    // Training loop
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        printf("Starting epoch %d\n", epoch + 1);
        printf("Debug: About to shuffle data\n");
        shuffleData(trainImages, trainLabels, TRAIN_SIZE);
        printf("Debug: Data shuffled successfully\n");

        for (int i = 0; i < TRAIN_SIZE; i += BATCH_SIZE) {
            printf("Debug: Processing batch starting at index %d\n", i);
            for (int j = i; j < i + BATCH_SIZE && j < TRAIN_SIZE; j++) {
                printf("Debug: Processing image %d\n", j);
                float* input = (float*)malloc(784 * sizeof(float));
                if (input == NULL) {
                    fprintf(stderr, "Failed to allocate memory for input\n");
                    exit(1);
                }
                for (int k = 0; k < 784; k++) {
                    input[k] = (float)trainImages[j][k];
                }
                printf("Debug: Input converted to float\n");

                int label = trainLabels[j];

                float* conv1Output = (float*)malloc(26 * 26 * 32 * sizeof(float));
                if (conv1Output == NULL) {
                    fprintf(stderr, "Failed to allocate memory for conv1Output\n");
                    exit(1);
                }
                printf("Debug: conv1Output allocated\n");

                // Forward pass
                printf("Debug: Starting forward pass\n");
                forwardConv(conv1, input, conv1Output);
                printf("Debug: forwardConv completed\n");

                // Free memory
                free(input);
                free(conv1Output);
            }
        }

        // Evaluate on test set
        int correctPredictions = 0;
        for (int i = 0; i < TEST_SIZE; i++) {
            float* input = (float*)testImages[i];
            float* output = forwardPass(conv1, pool1, conv2, pool2, fc1, fc2, input);
            int prediction = argmax(output, NUM_CLASSES);
            if (prediction == testLabels[i]) {
                correctPredictions++;
            }
            free(output);
        }

        float accuracy = (float)correctPredictions / TEST_SIZE;
        printf("Epoch %d, Test Accuracy: %.2f%%\n", epoch + 1, accuracy * 100);
    }

    // Free memory
    freeConvLayer(conv1);
    freePoolingLayer(pool1);
    freeConvLayer(conv2);
    freePoolingLayer(pool2);
    freeFCLayer(fc1);
    freeFCLayer(fc2);

    for (int i = 0; i < TRAIN_SIZE; i++) {
        free(trainImages[i]);
    }
    free(trainImages);
    free(trainLabels);

    for (int i = 0; i < TEST_SIZE; i++) {
        free(testImages[i]);
    }
    free(testImages);
    free(testLabels);

    return 0;
}

float* forwardPass(ConvLayer* conv1, PoolingLayer* pool1, ConvLayer* conv2, PoolingLayer* pool2, FCLayer* fc1, FCLayer* fc2, float* input) {

    float* conv1Output = (float*)malloc(26 * 26 * 32 * sizeof(float));
    float* pool1Output = (float*)malloc(13 * 13 * 32 * sizeof(float));
    float* conv2Output = (float*)malloc(11 * 11 * 64 * sizeof(float));
    float* pool2Output = (float*)malloc(5 * 5 * 64 * sizeof(float));
    float* fc1Output = (float*)malloc(128 * sizeof(float));
    float* fc2Output = (float*)malloc(NUM_CLASSES * sizeof(float));

    forwardConv(conv1, input, conv1Output);
    forwardPooling(pool1, conv1Output, pool1Output);
    forwardConv(conv2, pool1Output, conv2Output);
    forwardPooling(pool2, conv2Output, pool2Output);
    forwardFC(fc1, pool2Output, fc1Output);
    forwardFC(fc2, fc1Output, fc2Output);

    free(conv1Output);
    free(pool1Output);
    free(conv2Output);
    free(pool2Output);
    free(fc1Output);

    return fc2Output;
}

void backwardPass(ConvLayer* conv1, PoolingLayer* pool1, ConvLayer* conv2, PoolingLayer* pool2, FCLayer* fc1, FCLayer* fc2,
                  float* input, float* conv1Output, float* pool1Output, float* conv2Output, float* pool2Output, float* fc1Output, float* fc2Output,
                  int label, float learningRate) {
    // Compute gradients
    float* fc2Gradient = (float*)calloc(NUM_CLASSES, sizeof(float));
    fc2Gradient[label] = -1.0f / fc2Output[label];

    float* fc1Gradient = (float*)malloc(128 * sizeof(float));
    float* pool2Gradient = (float*)malloc(5 * 5 * 64 * sizeof(float));
    float* conv2Gradient = (float*)malloc(11 * 11 * 64 * sizeof(float));
    float* pool1Gradient = (float*)malloc(13 * 13 * 32 * sizeof(float));
    float* conv1Gradient = (float*)malloc(26 * 26 * 32 * sizeof(float));

    backwardFC(fc2, fc1Output, fc2Output, fc2Gradient, fc1Gradient, learningRate);
    backwardFC(fc1, pool2Output, fc1Output, fc1Gradient, pool2Gradient, learningRate);
    backwardPooling(pool2, conv2Output, pool2Output, pool2Gradient, conv2Gradient);
    backwardConv(conv2, pool1Output, conv2Output, conv2Gradient, pool1Gradient, learningRate);
    backwardPooling(pool1, conv1Output, pool1Output, pool1Gradient, conv1Gradient);
    backwardConv(conv1, input, conv1Output, conv1Gradient, NULL, learningRate);

    free(fc2Gradient);
    free(fc1Gradient);
    free(pool2Gradient);
    free(conv2Gradient);
    free(pool1Gradient);
    free(conv1Gradient);
}