#ifndef CNN_H
#define CNN_H

#include <stdint.h>

#define WIDTH 28
#define HEIGHT 28
#define DEPTH 1
#define CLASSES 24

typedef struct {
    float* data;
    int width;
    int height;
    int depth;
} Tensor;

typedef struct {
    int inputWidth;
    int inputHeight;
    int inputDepth;
    int numFilters;
    int kernelSize;
    int stride;
    int padding;
    float* weights;
    float* bias;
} ConvLayer;

typedef struct {
    int poolSize;
    int stride;
} MaxPoolLayer;

typedef struct {
    float rate;
} DropoutLayer;

typedef struct {
    int inputSize;
    int outputSize;
    float* weights;
    float* bias;
} FCLayer;

typedef struct {
    ConvLayer conv1;
    MaxPoolLayer pool1;
    ConvLayer conv2;
    MaxPoolLayer pool2;
    ConvLayer conv3;
    FCLayer fc1;
    DropoutLayer dropout;
    FCLayer fc2;
} CNN;

CNN* createCNN();
void freeCNN(CNN* cnn);
void forwardPass(CNN* cnn, float* input, float* output);
void backwardPass(CNN* cnn, float* input, float* gradOutput);
void updateParameters(CNN* cnn, float learningRate);

#endif
