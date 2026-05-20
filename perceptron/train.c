#include "perceptron.h"

int main() {
    Perceptron p = {{0.0, 0.0}, 0.0, 0.1}; //Initial weights, bias, and LR
    int inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    int targets[4] = {0, 1, 1, 1};

    printf("Training initial model...\n");
    for (int epoch = 0;  epoch < 100;  epoch++)
    {
        for (int i = 0; i < 4; i++)
        {
            double sum = (inputs[i][0] * p.weights[0]) + (inputs[i][1] * p.weights[1] + p.bias);
            int error = targets[i] - activate(sum);

            p.weights[0] += p.learning_rate * error * inputs[i][0];
            p.weights[1] += p.learning_rate * error * inputs[i][1];
            p.bias += p.learning_rate * error;
        }
    }
    
    FILE *f = fopen("model.bin", "wb");
    fwrite(&p, sizeof(Perceptron), 1, f);
    fclose(f);
    printf("Model saved to model.bin\n");

    return 0;
}