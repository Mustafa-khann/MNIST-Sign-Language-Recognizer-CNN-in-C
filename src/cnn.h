#ifndef CNN_H
#define CNN_H

typedef struct {
    int numFilters;
    int filterSize;
    double ***filters;
    double **biases;
    double ***deltaFilters;
    double **deltaBiases;
} ConvLayer;

typdef struct {
    int numNeurons;
    int inputSize;
    double **weights;
    double *biases;
    double **deltaWeights;
    double *deltaBiases;
} FCLayer;

typedef struct {
    ConvLayer *convLayer;
    int numConvLayer;
    FCLayer *fcLayers;
    int numFCLayers;
} CNN;

// Function prototypes
void initConvLayer(ConvLayer *layer, int numFilters, int filterSize);
void initFcLayer(FCLayer *layer, int numNeurons, int inputSize);
void initCNN(CNN *cnn, int numConvLayers, int numFCLayers);

void forwardConvLayer(ConvLayer *layer, double **input, double **output);
void forwardFcLayer(FCLayer *layer, double *input, double *output);
void forwardCNN(CNN *cnn, double **input, double *output);

void backwardConvLayer(ConvLayer *layer, double **input, double **deltaOutput, double **deltaInput);
void backwardFcLayer(FCLayer *layer, double *input, double *deltaOutput, double *deltaInput);
void backwardCNN(CNN *cnn, double **input, double *output, double **deltaInput, double *deltaOutput);

void updataConvLayer(ConvLayer *layer, double learningRate);
void updateFcLayer(FCLayer *layer, double learningRate);
void updateCNN(CNN *cnn, double learningRate);

void trainCNN(CNN *cnn, double **trainData, int **trainLabels, int numEpochs, double learningRate);
double testCNN(CNN *cnn, double **testData, double **testLabels);

void freeConvLayer(ConvLayer *layer);
void freeFcLayer(FCLayer *layer);
void freeCNN(CNN *cnn); 

#endif