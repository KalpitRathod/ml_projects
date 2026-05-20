#include "perceptron.h"

int main(){
    Perceptron p;
    FILE *f = fopen("model.bin", "rb");
    if (!f) { perror("Binary file not found"); return 1;}
    fread(&p, sizeof(Perceptron), 1, f);
    fclose(f);

    //New Data to refine the model (e.g., correcting specific edge cases)
    int new_inputs[2][2] = {{0,0}, {1,1}};
    int new_targets[2] = {0, 1};

    printf("Refining existing weights...\n");
    for (int epoch = 0; epoch < 50; epoch++)
    {
        for (int i = 0; i < 2; i++)
        {
            double sum = (new_inputs[i][0] * p.weights[0]) + (new_inputs[i][1] * p.weights[1]) + p.bias;
            int error = new_targets[i] - activate(sum);
            p.weights[0] += p.learning_rate * error * new_inputs[i][0];
            p.weights[1] += p.learning_rate * error * new_inputs[i][1];
            p.bias += p.learning_rate * error;
        }
    }

    f = fopen("model.bin", "wb");
    fwrite(&p, sizeof(Perceptron), 1, f);
    fclose(f);
    printf("Refined model updated in model.bin\n");
    return 0;
}