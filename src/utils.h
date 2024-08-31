#ifndef UTILS_H
#define UTILS_H

void loadMNISTData(const char* trainFile, const char* testFile,
    double **trainingImages, int *trainingLabels,
    double **testImages, int *testLabels,
    int trainingSize, int testSize);

int argmax(float* array, int size);

void shuffleData(double** images, int* labels, int size);
#endif
