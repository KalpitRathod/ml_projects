#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double weights[2];
    double bias;
    double learning_rate;
} Perceptron;

//Activation function
static int activate(double sum) {
    if (sum>=0)
    {
        return 1;
    } else {
        return 0;
    }
}

#endif