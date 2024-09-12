#include <cstddef>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cnn.h"
#include "layers.h"
#include "utils.h"

CNN* createCNN() {
    CNN* cnn = (CNN*)malloc(sizeof(CNN));
    if(cnn == NULL)
        {
          return NULL;
        }

    // Initialize layers
    cnn->conv1 = *createConvLayer(28, 28, 1, 32, 3, 1, 1);
    cnn->pool1 = *createMaxPoolLayer(2, 2);
    cnn->conv2 = *createConvLayer(14, 14, 32, 64, 3, 1, 1);
    cnn->pool2 = *createMaxPoolLayer(2, 2);
    cnn->conv3 = *createConvLayer(7, 7, 64, 128, 3, 1, 1);
    cnn->fc1 = *createFCLayer(7 * 7 * 128, 512);
    cnn->dropout = *createDropoutLayer(0.5);
    cnn->fc2 = *createFCLayer(512, CLASSES);

    return cnn;
}

void freeCNN(CNN* cnn) {
    if(cnn == NULL)
        {
            return;
        }

    freeConvLayer(&cnn->conv1);
    freeConvLayer(&cnn->conv2);
    freeConvLayer(&cnn->conv3);
    freeMaxPoolLayer(&cnn->pool1);
    freeMaxPoolLayer(&cnn->pool2);
    freeFCLayer(&cnn->fc1);
    freeFCLayer(&cnn->fc2);
    freeDropoutLayer(&cnn->dropout);
    free(cnn);
}

void forwardPass(CNN* cnn, float* input, float* output){
    Tensor inputTensor = {input, WIDTH, HEIGHT, DEPTH};

    // Convolutional Layer 1 + ReLU
    Tensor conv1Output = {NULL, 28, 28, 32};
    conv1Output.data = (float*)malloc(28 * 28 * 32 * sizeof(float));
    forwardConv(&cnn->conv1, &inputTensor, &conv1Output);
    for(int i = 0; i < 28 * 28 * 32; i++)
       {
           conv1Output.data[i] = relu(conv1Output.data[i]);
       }

   // Max Pooling Layer 1
   Tensor pool1Output = {NULL, 14, 14, 32};
   pool1Output.data = (float*)malloc(14*14*32*sizeof(float));
   forwardMaxPool(&cnn->pool1, &conv1Output, &pool1Output);

   // Convolutional Layer 2 + ReLU
   Tensor conv2Output = {NULL, 14, 14, 64};
   conv2Output.data = (float*)malloc(14*14*64 * sizeof(float));
   forwardConv(&cnn->conv2, &pool1Output, &conv2Output);
   for(int i = 0; i<14*14*64; i++)
       {
           conv2Output.data[i] = relu(conv2Output.data[i]);
       }

    // Max Pooling Layer 2
    Tensor pool2Output = {NULL, 7, 7 , 64};
    pool2Output.data = (float*)malloc(7*7*64*sizeof(float));
    forwardMaxPool(&cnn->pool2, &conv2Output, &pool2Output);


    // Covolutional Layer 3 = ReLU
    Tensor conv3Output = {NULL, 7, 7, 128};
    conv3Output.data = (float*)malloc(7*7*128*sizeof(float));
    forwardConv(&cnn->conv3, &pool2Output, &conv3Output);
    for(int i = 0; i<7*7*128; i++)
        {
            conv3Output.data[i] = relu(conv3Output.data[i]);
        }

    // Flatten
    float* flattenOutput = conv3Output.data;

    // Fully Connected Layer 1 + ReLU
    float* fc1Output = (float*)malloc(512*sizeof(float));
    forwardFC(&cnn->fc1, flattenOutput, fc1Output);
    for(int i = 0; i<512; i++)
        {
            fc1Output[i] = relu(fc1Output[i]);
        }

    // Dropout Layer
    float* dropoutOutput = (float*)malloc(512 * sizeof(float));
    forwardDropout(&cnn->dropout, fc1Output, dropoutOutput, 512, 1);

    // Fully Connected Layer 2 (Output Layer)
   forwardFC(&cnn->fc2, dropoutOutput, output);

   // Apply Softmax
   softmax(output, CLASSES);

   // Free temporary memory
   free(conv1Output.data);
   free(pool1Output.data);
   free(conv2Output.data);
   free(pool2Output.data);
   free(conv3Output.data);
   free(fc1Output);
   free(dropoutOutput);
}

void backwardPass(CNN *cnn, float *input, int trueLabel)
{
    Tensor inputTensor = {input, WIDTH, HEIGHT, DEPTH};

    Tensor conv1Output = {NULL, 28, 28, 32};
    conv1Output.data = (float*)malloc(28*28*32*sizeof(float));
    forwardConv(&cnn->conv1, &inputTensor, &conv1Output);
    for(int i = 0; i<28*28*32; i++)
        {
            conv1Output.data[i] = relu(conv1Output.data[i]);
        }

    Tensor pool1Output = {NULL, 14, 14, 32};
    pool1Output.data = (float*)malloc(14*14*32*sizeof(float));
    forwardMaxPool(&cnn->pool1, &conv1Output, &pool1Output);

    Tensor conv2Output = {NULL, 14, 14, 64};
    conv2Output.data = (float*)malloc(14*14*64*sizeof(float));
    forwardConv(&cnn->conv2, &pool1Output, &conv2Output);
    for(int i = 0; i<14*14*64; i++)
        {
            conv2Output.data[i] = relu(conv2Output.data[i]);
        }


}
