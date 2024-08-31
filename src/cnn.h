#ifndef CNN_H
#define CNN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Define structures for CNN layers
typedef struct {
    int inputWidth;
    int inputHeight;
    int inputDepth;
    int kernelSize;
    int numFilters;
    float* weights;
    float* bias;
} ConvLayer;

typedef struct {
    int inputWidth;
    int inputHeight;
    int inputDepth;
    int poolSize;
} PoolingLayer;

typedef struct {
    int inputSize;
    int outputSize;
    float* weights;
    float* bias;
} FCLayer;

// Function declarations
ConvLayer* createConvLayer(int inputWidth, int inputHeight, int inputDepth, int kernelSize, int numFilters);
PoolingLayer* createPoolingLayer(int inputWidth, int inputHeight, int inputDepth, int poolSize);
FCLayer* createFCLayer(int inputSize, int outputSize);

void forwardConv(ConvLayer* layer, float* input, float* output);
void forwardPooling(PoolingLayer* layer, float* input, float* output);
void forwardFC(FCLayer* layer, float* input, float* output);

void backwardConv(ConvLayer* layer, float* input, float* output, float* outputGradient, float* inputGradient, float learningRate);
void backwardPooling(PoolingLayer* layer, float* input, float* output, float* outputGradient, float* inputGradient);
void backwardFC(FCLayer* layer, float* input, float* output, float* outputGradient, float* inputGradient, float learningRate);

void freeConvLayer(ConvLayer* layer);
void freePoolingLayer(PoolingLayer* layer);
void freeFCLayer(FCLayer* layer);

#endif