#ifndef UTILS_H
#define UTILS_H

#include "cnn.h"

typedef struct {
    float* images;
    int* labels;
    int size;
} Dataset;

// Dataset fnctions
Dataset* loadDataset(const char* filename, int size);
void freeDataset(Dataset* dataset);
void shuffleDataset(Dataset* dataset);

// Tensor Operations
Tensor* createTensor(int width, int height, int depth);
void freeTensor(Tensor* tensor);
void zeroTensor(Tensor* tensor);

// Activation Functions
float relu(float x);
float reluGradient(float x);
void softmax(float* input, int size);

// Loss Function
float categoricalCrossEntropy(float* predictions, int labels);

#endif
