#include "cnn.h"

ConvLayer* createConvLayer(int inputWidth, int inputHeight, int inputDepth, int kernelSize, int numFilters) {
    ConvLayer* layer = (ConvLayer*)malloc(sizeof(ConvLayer));
    layer->inputWidth = inputWidth;
    layer->inputHeight = inputHeight;
    layer->inputDepth = inputDepth;
    layer->kernelSize = kernelSize;
    layer->numFilters = numFilters;

    int weightsSize = kernelSize * kernelSize * inputDepth * numFilters;
    layer->weights = (float*)malloc(weightsSize * sizeof(float));
    layer->bias = (float*)malloc(numFilters * sizeof(float));

    // Initialize weights and biases (Xavier initialization)
    float scale = sqrt(2.0f / (inputWidth * inputHeight * inputDepth + numFilters));
    for (int i = 0; i < weightsSize; i++) {
        layer->weights[i] = ((float)rand() / RAND_MAX) * 2 * scale - scale;
    }
    for (int i = 0; i < numFilters; i++) {
        layer->bias[i] = 0;
    }

    return layer;
}

PoolingLayer* createPoolingLayer(int inputWidth, int inputHeight, int inputDepth, int poolSize) {
    PoolingLayer* layer = (PoolingLayer*)malloc(sizeof(PoolingLayer));
    layer->inputWidth = inputWidth;
    layer->inputHeight = inputHeight;
    layer->inputDepth = inputDepth;
    layer->poolSize = poolSize;
    return layer;
}

FCLayer* createFCLayer(int inputSize, int outputSize) {
    FCLayer* layer = (FCLayer*)malloc(sizeof(FCLayer));
    layer->inputSize = inputSize;
    layer->outputSize = outputSize;

    layer->weights = (float*)malloc(inputSize * outputSize * sizeof(float));
    layer->bias = (float*)malloc(outputSize * sizeof(float));

    // Initialize weights and biases (Xavier initialization)
    float scale = sqrt(2.0f / (inputSize + outputSize));
    for (int i = 0; i < inputSize * outputSize; i++) {
        layer->weights[i] = ((float)rand() / RAND_MAX) * 2 * scale - scale;
    }
    for (int i = 0; i < outputSize; i++) {
        layer->bias[i] = 0;
    }

    return layer;
}

void forwardConv(ConvLayer* layer, float* input, float* output) {
    int outputWidth = layer->inputWidth - layer->kernelSize + 1;
    int outputHeight = layer->inputHeight - layer->kernelSize + 1;

    for (int f = 0; f < layer->numFilters; f++) {
        for (int y = 0; y < outputHeight; y++) {
            for (int x = 0; x < outputWidth; x++) {
                float sum = 0;
                for (int d = 0; d < layer->inputDepth; d++) {
                    for (int ky = 0; ky < layer->kernelSize; ky++) {
                        for (int kx = 0; kx < layer->kernelSize; kx++) {
                            int inputIndex = (y + ky) * layer->inputWidth * layer->inputDepth + (x + kx) * layer->inputDepth + d;
                            int weightIndex = f * layer->kernelSize * layer->kernelSize * layer->inputDepth + d * layer->kernelSize * layer->kernelSize + ky * layer->kernelSize + kx;
                            sum += input[inputIndex] * layer->weights[weightIndex];
                        }
                    }
                }
                sum += layer->bias[f];
                output[f * outputWidth * outputHeight + y * outputWidth + x] = fmaxf(0, sum); // ReLU activation
            }
        }
    }
}

void forwardPooling(PoolingLayer* layer, float* input, float* output) {
    int outputWidth = layer->inputWidth / layer->poolSize;
    int outputHeight = layer->inputHeight / layer->poolSize;

    for (int d = 0; d < layer->inputDepth; d++) {
        for (int y = 0; y < outputHeight; y++) {
            for (int x = 0; x < outputWidth; x++) {
                float maxVal = -INFINITY;
                for (int py = 0; py < layer->poolSize; py++) {
                    for (int px = 0; px < layer->poolSize; px++) {
                        int inputIndex = d * layer->inputWidth * layer->inputHeight + (y * layer->poolSize + py) * layer->inputWidth + (x * layer->poolSize + px);
                        maxVal = fmaxf(maxVal, input[inputIndex]);
                    }
                }
                output[d * outputWidth * outputHeight + y * outputWidth + x] = maxVal;
            }
        }
    }
}

void forwardFC(FCLayer* layer, float* input, float* output) {
    for (int i = 0; i < layer->outputSize; i++) {
        float sum = 0;
        for (int j = 0; j < layer->inputSize; j++) {
            sum += input[j] * layer->weights[i * layer->inputSize + j];
        }
        sum += layer->bias[i];
        output[i] = fmaxf(0, sum); // ReLU activation
    }
}

void backwardConv(ConvLayer* layer, float* input, float* output, float* outputGradient, float* inputGradient, float learningRate) {
    int outputWidth = layer->inputWidth - layer->kernelSize + 1;
    int outputHeight = layer->inputHeight - layer->kernelSize + 1;

    // Initialize inputGradient to zero
    for (int i = 0; i < layer->inputWidth * layer->inputHeight * layer->inputDepth; i++) {
        inputGradient[i] = 0;
    }

    for (int f = 0; f < layer->numFilters; f++) {
        for (int y = 0; y < outputHeight; y++) {
            for (int x = 0; x < outputWidth; x++) {
                int outputIndex = f * outputWidth * outputHeight + y * outputWidth + x;
                float gradOutput = outputGradient[outputIndex];

                // ReLU derivative
                if (output[outputIndex] <= 0) {
                    gradOutput = 0;
                }

                for (int d = 0; d < layer->inputDepth; d++) {
                    for (int ky = 0; ky < layer->kernelSize; ky++) {
                        for (int kx = 0; kx < layer->kernelSize; kx++) {
                            int inputIndex = (y + ky) * layer->inputWidth * layer->inputDepth + (x + kx) * layer->inputDepth + d;
                            int weightIndex = f * layer->kernelSize * layer->kernelSize * layer->inputDepth + d * layer->kernelSize * layer->kernelSize + ky * layer->kernelSize + kx;

                            inputGradient[inputIndex] += gradOutput * layer->weights[weightIndex];
                            layer->weights[weightIndex] -= learningRate * gradOutput * input[inputIndex];
                        }
                    }
                }

                layer->bias[f] -= learningRate * gradOutput;
            }
        }
    }
}

void backwardPooling(PoolingLayer* layer, float* input, float* output, float* outputGradient, float* inputGradient) {
    int outputWidth = layer->inputWidth / layer->poolSize;
    int outputHeight = layer->inputHeight / layer->poolSize;

    // Initialize inputGradient to zero
    for (int i = 0; i < layer->inputWidth * layer->inputHeight * layer->inputDepth; i++) {
        inputGradient[i] = 0;
    }

    for (int d = 0; d < layer->inputDepth; d++) {
        for (int y = 0; y < outputHeight; y++) {
            for (int x = 0; x < outputWidth; x++) {
                int outputIndex = d * outputWidth * outputHeight + y * outputWidth + x;
                float maxVal = -INFINITY;
                int maxIndex = -1;

                for (int py = 0; py < layer->poolSize; py++) {
                    for (int px = 0; px < layer->poolSize; px++) {
                        int inputIndex = d * layer->inputWidth * layer->inputHeight + (y * layer->poolSize + py) * layer->inputWidth + (x * layer->poolSize + px);
                        if (input[inputIndex] > maxVal) {
                            maxVal = input[inputIndex];
                            maxIndex = inputIndex;
                        }
                    }
                }

                inputGradient[maxIndex] += outputGradient[outputIndex];
            }
        }
    }
}

void backwardFC(FCLayer* layer, float* input, float* output, float* outputGradient, float* inputGradient, float learningRate) {
    // Initialize inputGradient to zero
    for (int i = 0; i < layer->inputSize; i++) {
        inputGradient[i] = 0;
    }

    for (int i = 0; i < layer->outputSize; i++) {
        float gradOutput = outputGradient[i];

        // ReLU derivative
        if (output[i] <= 0) {
            gradOutput = 0;
        }

        for (int j = 0; j < layer->inputSize; j++) {
            inputGradient[j] += gradOutput * layer->weights[i * layer->inputSize + j];
            layer->weights[i * layer->inputSize + j] -= learningRate * gradOutput * input[j];
        }

        layer->bias[i] -= learningRate * gradOutput;
    }
}

void freeConvLayer(ConvLayer* layer) {
    free(layer->weights);
    free(layer->bias);
    free(layer);
}

void freePoolingLayer(PoolingLayer* layer) {
    free(layer);
}

void freeFCLayer(FCLayer* layer) {
    free(layer->weights);
    free(layer->bias);
    free(layer);
}