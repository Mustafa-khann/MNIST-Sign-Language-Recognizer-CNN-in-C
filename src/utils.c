#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

#define MAX_LINE_LENGTH 10000

void loadMNISTData(const char *trainFile, const char *testFile, double **trainingImages, int *trainingLabels, double **testImages, int *testLabels, int trainingSize, int testSize)
{
    FILE *file;
    char line[MAX_LINE_LENGTH];
    char *token;

    // Load training data
    file = fopen(trainFile, "r");
    if(file == NULL)
    {
        fprintf(stderr, "Error opening training file\n");
        exit(1);
    }

    // Skip header
    if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
        fprintf(stderr, "Error reading header from training file\n");
        fclose(file);
        exit(1);
    }

    for(int i = 0; i < trainingSize; i++)
    {
        if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
            fprintf(stderr, "Error reading line %d from training file\n", i+1);
            fclose(file);
            exit(1);
        }

        token = strtok(line, ",");
        if (token == NULL) {
            fprintf(stderr, "Error parsing label in line %d of training file\n", i+1);
            fclose(file);
            exit(1);
        }
        trainingLabels[i] = atoi(token);

        for(int j = 0; j < 784; j++)
        {
            token = strtok(NULL, ",");
            if (token == NULL) {
                fprintf(stderr, "Error parsing pixel %d in line %d of training file\n", j+1, i+1);
                fclose(file);
                exit(1);
            }
            trainingImages[i][j] = atof(token) / 255.0;
        }
    }
    fclose(file);

    // Load test data
    file = fopen(testFile, "r");
    if(file == NULL)
    {
        fprintf(stderr, "Error opening test file\n");
        exit(1);
    }

    // Skip header
    if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
        fprintf(stderr, "Error reading header from test file\n");
        fclose(file);
        exit(1);
    }

    for(int i = 0; i < testSize; i++)
    {
        if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
            fprintf(stderr, "Error reading line %d from test file\n", i+1);
            fclose(file);
            exit(1);
        }

        token = strtok(line, ",");
        if (token == NULL) {
            fprintf(stderr, "Error parsing label in line %d of test file\n", i+1);
            fclose(file);
            exit(1);
        }
        testLabels[i] = atoi(token);

        for(int j = 0; j < 784; j++)
        {
            token = strtok(NULL, ",");
            if (token == NULL) {
                fprintf(stderr, "Error parsing pixel %d in line %d of test file\n", j+1, i+1);
                fclose(file);
                exit(1);
            }
            testImages[i][j] = atof(token) / 255.0;
        }
    }
    fclose(file);

    printf("MNIST data loaded from CSV files\n");
}

int argmax(float* array, int size) {
    int max_index = 0;
    float max_value = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_value) {
            max_value = array[i];
            max_index = i;
        }
    }
    return max_index;
}

void shuffleData(double** images, int* labels, int size) {
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        double* temp_image = images[i];
        images[i] = images[j];
        images[j] = temp_image;
        int temp_label = labels[i];
        labels[i] = labels[j];
        labels[j] = temp_label;
    }
}