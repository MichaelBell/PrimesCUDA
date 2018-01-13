#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <random>
#include <thread>
#include <chrono>
#include <climits>

#include "mini-gmp.h"

#define SINGLE_CXT 0

using namespace std::chrono;

struct PrimeTestCxt* primeTestInit();
void primeTestTerm(struct PrimeTestCxt* cxt);

void primeTest(struct PrimeTestCxt* cxt, int N_Size, int LIST_SIZE, const uint32_t* M, uint32_t* is_prime);

void myThread(struct PrimeTestCxt* cxt)
{
#if !SINGLE_CXT
	cxt = primeTestInit();

	const int LIST_SIZE = 1024;
#else
	const int LIST_SIZE = 2048;
#endif

	const int N_Size = 54;
	uint32_t *M = (uint32_t*)malloc(sizeof(uint32_t)*N_Size*LIST_SIZE);
	uint32_t *is_prime = (uint32_t*)malloc(sizeof(uint32_t)*LIST_SIZE);
	std::mt19937 generator;
	std::uniform_int_distribution<uint32_t> distrib(0, UINT_MAX);

	uint32_t k = 0;

	mpz_t z_ft_r, z_ft_n, z_m;
	mpz_init(z_ft_r);
	mpz_init(z_ft_n);
	mpz_init_set_ui(z_m, 2);
	mpz_pow_ui(z_m, z_m, (N_Size - 1) * 32);
	mpz_t z_ft_b;
	mpz_init_set_ui(z_ft_b, 2);

	while (k < (1 << 12))
	{
		for (int j = 1; j < N_Size; ++j)
		{
			M[j] = distrib(generator);
		}
		for (uint32_t i = 0; i < LIST_SIZE; i++) {
			M[i*N_Size] = (i + k) * 2 + 1;
			for (int j = 1; j < N_Size; ++j)
			{
				M[i*N_Size + j] = M[j];
			}
		}
		k += LIST_SIZE;

		high_resolution_clock::time_point startTime;
		high_resolution_clock::time_point endTime;

		startTime = high_resolution_clock::now();

		primeTest(cxt, N_Size, LIST_SIZE, M, is_prime);

		endTime = high_resolution_clock::now();

		printf("k: %d Time: %lf\n", k, duration_cast<duration<double>>(endTime - startTime).count());

#if 0
		for (int j = 1; j < (N_Size / 2); ++j)
		{
			z_m->_mp_d[j] = (mp_limb_t(M[j * 2 + 1]) << 32) + M[j * 2];
		}
		if (N_Size & 1)
		{
			z_m->_mp_d[N_Size / 2] = M[N_Size - 1];
		}
			
		mp_limb_t high_m = mp_limb_t(M[1]) << 32;
		for (uint32_t i = 0; i < LIST_SIZE; i+=3) {
			z_m->_mp_d[0] = high_m + M[i*N_Size];

			mpz_sub_ui(z_ft_n, z_m, 1);
			mpz_powm(z_ft_r, z_ft_b, z_ft_n, z_m);
			bool gmp_is_prime = mpz_cmp_ui(z_ft_r, 1) == 0;
			if (gmp_is_prime != bool(is_prime[i])) {
				printf("GMP %d disagrees with CUDA %d Index: %d\n", gmp_is_prime, is_prime[i], i);
				//abort();
			}
		}

#if 0
		// Display the result to the screen
		for (int i = 0; i < LIST_SIZE; i++)
			if (is_prime[i]) printf("%d\n", M[i*N_Size]);
#endif
#endif
	}

	free(M);
	free(is_prime);

#if !SINGLE_CXT
	primeTestTerm(cxt);
#endif
}

int main(void) {
	printf("started running\n");

#if SINGLE_CXT
	struct PrimeTestCxt* cxt = primeTestInit();
	printf("initialized\n");
#else
	struct PrimeTestCxt* cxt = NULL;
#endif

	high_resolution_clock::time_point startTime;
	high_resolution_clock::time_point endTime;

	startTime = high_resolution_clock::now();
	endTime = high_resolution_clock::now();

	startTime = high_resolution_clock::now();

	//std::this_thread::sleep_for(std::chrono::seconds(20));

	std::thread first(myThread, cxt);
	std::thread second(myThread, cxt);
//	std::thread third(myThread, cxt);
//	std::thread fourth(myThread, cxt);
	first.join();
	second.join();
//	third.join();
//	fourth.join();

	endTime = high_resolution_clock::now();

	printf("Total time: %lf\n", duration_cast<duration<double>>(endTime - startTime).count());

#if SINGLE_CXT
	primeTestTerm(cxt);
	printf("terminated\n");
#endif

	return 0;
}
