#ifndef LAYERS_H
#define LAYERS_H

#include "cnn.h"

// Convolutional layer
ConvLayer* createConvLayer(int inputWidth, int inputHeight, int inputDepth, int numFilters, int kernelSize, int Stride, int padding);
void freeConvLayer(ConvLayer* convLayer);
void forwardConv(ConvLayer* layer, Tensor* input, Tensor* output);
void backwardConv(ConvLayer* layer, Tensor* input, Tensor* gradInput, Tensor* gradOutput);

// Max Pooling Layer
MaxPoolLayer* createMaxPoolLayer(int poolSize, int stride);
void freeMaxPoolLayer(MaxPoolLayer* layer);
void forwardMaxPool(MaxPoolLayer* layer, Tensor* input, Tensor* output);
void backwardMaxPool(MaxPoolLayer* layer, Tensor* input, Tensor* gradInput, Tensor* gradOutput);

// Fully Connected Layer
FCLayer* createFCLayer(int inputSize, int outputSize);
void freeFCLayer(FCLayer* layer);
void forwardFC(FCLayer* layer, float* input, float* output);
void backwardFC(FCLayer* layer, float* dropoutOutput, float* gradinput, float* gradoutput);

// DropOut Layer
DropoutLayer* createDropoutLayer(float rate);
void freeDropoutLayer(DropoutLayer* layer);
void forwardDropout(DropoutLayer* layer, float* input, float* output, int size, int isTraining);
void backwardDropout(DropoutLayer* layer, float* gradInput, float* gradOutput, int size);

#endif
