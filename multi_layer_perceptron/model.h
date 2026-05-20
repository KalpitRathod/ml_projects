#ifndef MODEL_H
#define MODEL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_IN 128    // Max characters of input
#define HIDDEN 16     // Hidden neurons
#define MAX_OUT 32    // Max characters of output

typedef struct {
    double w_ih[MAX_IN][HIDDEN]; // Weights: Input to Hidden
    double w_ho[HIDDEN][MAX_OUT]; // Weights: Hidden to Output
    double bias_h[HIDDEN];
    double bias_o[MAX_OUT];
    double lr; // Learning Rate
} Network;

static double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static double d_sigmoid(double x) { return x * (1.0 - x); } // Derivative for backprop

#endif
