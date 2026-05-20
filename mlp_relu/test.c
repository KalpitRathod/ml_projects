#include "model.h"

int main() {
    Network n;
    FILE *f = fopen("model.bin", "rb");
    if(!f) return 1;
    fread(&n, sizeof(Network), 1, f);
    fclose(f);

    char input_str[MAX_IN];
    printf("Enter input: ");
    scanf("%s", input_str);

    double hidden[HIDDEN] = {0};
    double complexity = 0;

    // 1. Convert Input using new ASCII math
    for(int j=0; j<HIDDEN; j++) {
        double act = n.bias_h[j];
        for(int i=0; i<strlen(input_str); i++) {
            double normalized_in = (double)(input_str[i] - 32) / 95.0;
            act += normalized_in * n.w_ih[i][j];
        }
        hidden[j] = relu(act);
        complexity += hidden[j];
    }

    int length = ((int)complexity % MAX_OUT) + 1;
    printf("Output: ");

    for(int k=0; k<length; k++) {
        double out_val = n.bias_o[k];
        for(int j=0; j<HIDDEN; j++) out_val += hidden[j] * n.w_ho[j][k];
        
        // 2. Decode using new ASCII math (95 characters, starting at Space)
        // When converting a decimal double to an int in C, it truncates (rounds down) the number
        char c = (int)(sigmoid(out_val) * 95 + 0.5) + 32; //Adding 0.5 forces C to round to the nearest whole number
        
        // Ensure we only print readable characters
        if(c >= 32 && c <= 126) {
            printf("%c", c);
        }
    }
    printf("\n");

    return 0;
}