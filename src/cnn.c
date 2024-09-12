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


}
