#include "model.h"

void refine_model(Network *n, char *input_str, char *target_str) {
    double input[MAX_IN] = {0};
    double target[MAX_OUT] = {0};
    double hidden[HIDDEN] = {0};
    double output[MAX_OUT] = {0};

    // 1. Convert Text to Numbers (ASCII 32 to 126)
    for(int i=0; i<strlen(input_str) && i<MAX_IN; i++) {
        input[i] = (double)(input_str[i] - 32) / 95.0;
    }
    for(int i=0; i<strlen(target_str) && i<MAX_OUT; i++) {
        target[i] = (double)(target_str[i] - 32) / 95.0;
    }

    printf("Training [%s] -> [%s]...\n", input_str, target_str);

    // 2. Train for 50,000 Epochs
    for(int epoch = 0; epoch <= 50000; epoch++) {
        double total_error = 0;

        // --- FORWARD PASS ---
        for(int j=0; j<HIDDEN; j++) {
            double activation = n->bias_h[j];
            for(int i=0; i<MAX_IN; i++) activation += input[i] * n->w_ih[i][j];
            hidden[j] = sigmoid(activation);
        }
        for(int k=0; k<MAX_OUT; k++) {
            double activation = n->bias_o[k];
            for(int j=0; j<HIDDEN; j++) activation += hidden[j] * n->w_ho[j][k];
            output[k] = sigmoid(activation);
        }

        // --- BACKPROPAGATION & ERROR TRACKING ---
        double out_errors[MAX_OUT];
        for(int k=0; k<MAX_OUT; k++) {
            double err = target[k] - output[k];
            total_error += err * err; // Track the error!
            out_errors[k] = err * d_sigmoid(output[k]);
        }

        // Print progress every 5000 epochs
        if(epoch % 5000 == 0) {
            printf("  Epoch %d | Error: %f\n", epoch, total_error);
        }

        double hid_errors[HIDDEN];
        for(int j=0; j<HIDDEN; j++) {
            double err = 0;
            for(int k=0; k<MAX_OUT; k++) err += out_errors[k] * n->w_ho[j][k];
            hid_errors[j] = err * d_sigmoid(hidden[j]);
        }

        // --- UPDATE WEIGHTS ---
        for(int k=0; k<MAX_OUT; k++) {
            for(int j=0; j<HIDDEN; j++) n->w_ho[j][k] += n->lr * out_errors[k] * hidden[j];
            n->bias_o[k] += n->lr * out_errors[k];
        }
        for(int j=0; j<HIDDEN; j++) {
            for(int i=0; i<MAX_IN; i++) n->w_ih[i][j] += n->lr * hid_errors[j] * input[i];
            n->bias_h[j] += n->lr * hid_errors[j];
        }
    }
}

int main() {
    Network n;
    FILE *f = fopen("model.bin", "rb");
    if(!f) { printf("No model.bin found! Run train first.\n"); return 1; }
    fread(&n, sizeof(Network), 1, f);
    fclose(f);

    // Let's teach it exactly what you want
    refine_model(&n, "ball", "bat");
    
    f = fopen("model.bin", "wb");
    fwrite(&n, sizeof(Network), 1, f);
    fclose(f);
    printf("Model updated successfully.\n");
    return 0;
}