#include "perceptron.h"

int main(){
    Perceptron p;
    FILE *f = fopen("model.bin", "rb");
    if (!f) { perror("Could not load model.bin"); return 1;}
    fread(&p, sizeof(Perceptron), 1, f);
    fclose(f);

    int test_data[4][2] = {{0,0}, {0,1}, {1, 0}, {1, 1}};

    printf("Testing loaded model:\n");
    printf("Weight: [%.2f, %.2f] | Bias: %.2f\n", p.weights[0], p.weights[1], p.bias);
    printf("-------------------------------\n");
    for (int i = 0; i < 4; i++)
    {
        double sum = (test_data[i][0] * p.weights[0]) + (test_data[i][1] * p.weights[1]) + p.bias;
        printf("Input: %d, %d -> Prediction: %d\n", test_data[i][0], test_data[i][1], activate(sum));
    }

    return 0;
}