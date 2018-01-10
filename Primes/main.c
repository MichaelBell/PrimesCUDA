#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

void primeTest(int N_Size, int LIST_SIZE, const uint32_t* M, uint32_t* is_prime);

int main(void) {
	printf("started running\n");

	// Create the two input vectors
	int i;
	const int LIST_SIZE = 4096;
	const int N_Size = 53;
	uint32_t *M = (uint32_t*)malloc(sizeof(uint32_t)*N_Size*LIST_SIZE);
	for (i = 0; i < LIST_SIZE; i++) {
		M[i*N_Size] = i*2 + 1;
		for (int j = 1; j < N_Size - 1; ++j)
		{
			M[i*N_Size + j] = 0;
		}
		M[(i+1)*N_Size - 1] = 1;
	}

	uint32_t *is_prime = (uint32_t*)malloc(sizeof(uint32_t)*LIST_SIZE);
	primeTest(N_Size, LIST_SIZE, M, is_prime);

	// Display the result to the screen
	for (i = 0; i < LIST_SIZE; i++)
		if (is_prime[i]) printf("%d\n", M[i*N_Size]);

	free(M);
	free(is_prime);
	return 0;
}