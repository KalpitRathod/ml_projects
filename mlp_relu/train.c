#include "model.h"
#include <time.h>

int main() {
    srand(time(NULL));
    Network n;
    n.lr = 0.1;

    // Initialize weights with random values between -1 and 1
    for(int i=0; i<MAX_IN; i++) 
        for(int j=0; j<HIDDEN; j++) n.w_ih[i][j] = ((double)rand()/RAND_MAX) * 2 - 1;
    for(int i=0; i<HIDDEN; i++) 
        for(int j=0; j<MAX_OUT; j++) n.w_ho[i][j] = ((double)rand()/RAND_MAX) * 2 - 1;

    FILE *f = fopen("model.bin", "wb");
    fwrite(&n, sizeof(Network), 1, f);
    fclose(f);
    printf("Initialized modular model.bin\n");
    return 0;
}