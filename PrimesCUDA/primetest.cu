#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

#include "prime-gmp.h"

typedef uint32_t uint;
typedef uint64_t ulong;

#define MAX_N_SIZE 128
#define MAX_BLOCK_SIZE 32
#define MAX_JOB_SIZE 2048

template<uint N_Size>
__global__
void fermat_test(const uint *M_in, const uint *Mi_in, const uint *R_in, uint *is_prime) {

	uint R[N_Size];
	uint M[N_Size];

	{
		// Get the index of the current element to be processed
		const int offset = (blockDim.x*blockIdx.x + threadIdx.x) * N_Size;

		for (int i = 0; i < N_Size; ++i)
		{
			M[i] = M_in[offset + i];
			R[i] = R_in[offset + i];
		}
	}

	const uint shift = __clz(M[N_Size - 1]);
	const uint highbit = ((uint)1) << 31;
	uint startbit = highbit >> shift;

	const uint mi = Mi_in[blockDim.x*blockIdx.x + threadIdx.x];

	int en = N_Size;
#pragma unroll 1
	while (en-- > 0)
	{
		uint bit = startbit;
		startbit = highbit;
		uint E = M[en];
		if (en == 0) E--;

		do
		{
			{
				uint P[N_Size * 2];
				//mpn_sqr(pp, rp, mn);
				P[0] = R[0] * R[0];
				P[1] = __umulhi(R[0], R[0]);
				{
					uint T[N_Size * 2];

					{
						uint cy = 0;
						for (int i = 0; i < N_Size - 1; ++i)
						{
							T[i] = R[i + 1] * R[0];
							T[i] += cy;
							cy = (T[i] < cy) + __umulhi(R[i + 1], R[0]);
						}
						T[N_Size - 1] = cy;
					}

#pragma unroll 1
					for (int j = 2; j < N_Size; ++j)
					{
						uint cy = 0;
						for (int i = j; i < N_Size; ++i)
						{
							uint lp = R[i] * R[j - 1];
							lp += cy;
							cy = (lp < cy) + __umulhi(R[i], R[j - 1]);
							lp += T[i + j - 2];
							cy += lp < T[i + j - 2];
							T[i + j - 2] = lp;
						}
						T[N_Size + j - 2] = cy;
					}

					for (int i = 0; i < N_Size; ++i)
					{
						P[2 * i] = R[i] * R[i];
						P[2 * i + 1] = __umulhi(R[i], R[i]);
					}

					uint cy = 0;
					for (int i = 0; i < N_Size * 2 - 2; ++i)
					{
						uint t = T[i] & highbit;
						T[i] <<= 1;
						T[i] |= cy;
						cy = t >> 31;
					}
					P[2 * N_Size - 1] += cy;

					cy = 0;
					for (int i = 0; i < N_Size * 2 - 2; ++i)
					{
						uint a = P[i + 1] + cy;
						cy = (a < P[i + 1]);
						a += T[i];
						cy += (a < T[i]);
						P[i + 1] = a;
					}
					P[2 * N_Size - 1] += cy;
				}

				//if (mpn_redc_1(rp, pp, mp, mn, mi) != 0) 
				//  mpn_sub_n(rp, rp, mshifted, n);
#pragma unroll 1
				for (int j = 0; j < N_Size; ++j)
				{
					uint cy = 0;
					uint v = P[j] * mi;
					for (int i = 0; i < N_Size; ++i)
					{
						uint lp = M[i] * v;
						lp += cy;
						cy = (lp < cy) + __umulhi(M[i], v);
						lp += P[i + j];
						cy += lp < P[i + j];
						P[i + j] = lp;
					}
					R[j] = cy;
				}

				{
					uint cy = 0;
					for (int i = 0; i < N_Size; ++i)
					{
						uint a = R[i] + cy;
						cy = (a < R[i]);
						a += P[i + N_Size];
						cy += (a < P[i + N_Size]);
						R[i] = a;
					}

					if (cy != 0)
					{
						cy = 0;
						uint last_shifted = 0;
						for (int i = 0; i < N_Size; ++i)
						{
							uint a = R[i];
							uint b = (M[i] << shift) | last_shifted;
							last_shifted = M[i] >> (32 - shift);
							b += cy;
							cy = (b < cy);
							cy += (a < b);
							R[i] = a - b;
						}
					}
				}
			}

			if (E & bit)
			{
				//mp_limb_t carry = mpn_lshift(rp, rp, mn, 1);
				uint carry = 0;
				for (int i = 0; i < N_Size; ++i)
				{
					uint t = R[i] & highbit;
					R[i] <<= 1;
					R[i] |= carry;
					carry = t >> 31;
				}
				while (carry)
				{
					//carry -= mpn_sub_n(rp, rp, mshifted, mn);
					uint cy = 0;
					uint last_shifted = 0;
					for (int i = 0; i < N_Size; ++i)
					{
						uint a = R[i];
						uint b = (M[i] << shift) | last_shifted;
						last_shifted = M[i] >> (32 - shift);
						b += cy;
						cy = (b < cy);
						cy += (a < b);
						R[i] = a - b;
					}
					carry -= cy;
				}
			}
			bit >>= 1;
		} while (bit > 0);

	}

	// DeREDCify - necessary as rp can have a large
	//             multiple of m in it (although I'm not 100% sure
	//             why it can't after this redc!)
	{
		uint T[N_Size * 2];
		for (int i = 0; i < N_Size; ++i)
		{
			T[i] = R[i];
			T[N_Size + i] = 0;
		}

		// MPN_REDC_1(rp, tp, mp, mn, mi);
#pragma unroll 1
		for (int j = 0; j < N_Size; ++j)
		{
			uint cy = 0;
			uint v = T[j] * mi;
			for (int i = 0; i < N_Size; ++i)
			{
				uint lp = M[i] * v;
				lp += cy;
				cy = (lp < cy) + __umulhi(M[i], v);
				lp += T[i + j];
				cy += lp < T[i + j];
				T[i + j] = lp;
			}
			R[j] = cy;
		}

		{
			uint cy = 0;
			for (int i = 0; i < N_Size; ++i)
			{
				uint a = R[i] + cy;
				cy = (a < R[i]);
				a += T[i + N_Size];
				cy += (a < T[i + N_Size]);
				R[i] = a;
			}

			if (cy != 0)
			{
				cy = 0;
				uint last_shifted = 0;
				for (int i = 0; i < N_Size; ++i)
				{
					uint a = R[i];
					uint b = (M[i] << shift) | last_shifted;
					last_shifted = M[i] >> (32 - shift);
					b += cy;
					cy = (b < cy);
					cy += (a < b);
					R[i] = a - b;
				}
			}
		}
	}

	bool result = true;
	if (R[N_Size - 1] != 0)
	{
		// Compare to m+1
		uint cy = 1;
		for (int i = 0; i < N_Size && result; ++i)
		{
			uint a = M[i] + cy;
			cy = a < M[i];
			if (R[i] != a) result = false;
		}
	}
	else
	{
		// Compare to 1
		result = R[0] == 1;
		for (int i = 1; i < N_Size && result; ++i)
		{
			if (R[i] != 0) result = false;
		}
	}

	is_prime[blockDim.x*blockIdx.x + threadIdx.x] = result;
}

#define DEBUG 0

#define MAX_SOURCE_SIZE (0x100000)

const unsigned char  binvert_limb_table[128] = {
	0x01, 0xAB, 0xCD, 0xB7, 0x39, 0xA3, 0xC5, 0xEF,
	0xF1, 0x1B, 0x3D, 0xA7, 0x29, 0x13, 0x35, 0xDF,
	0xE1, 0x8B, 0xAD, 0x97, 0x19, 0x83, 0xA5, 0xCF,
	0xD1, 0xFB, 0x1D, 0x87, 0x09, 0xF3, 0x15, 0xBF,
	0xC1, 0x6B, 0x8D, 0x77, 0xF9, 0x63, 0x85, 0xAF,
	0xB1, 0xDB, 0xFD, 0x67, 0xE9, 0xD3, 0xF5, 0x9F,
	0xA1, 0x4B, 0x6D, 0x57, 0xD9, 0x43, 0x65, 0x8F,
	0x91, 0xBB, 0xDD, 0x47, 0xC9, 0xB3, 0xD5, 0x7F,
	0x81, 0x2B, 0x4D, 0x37, 0xB9, 0x23, 0x45, 0x6F,
	0x71, 0x9B, 0xBD, 0x27, 0xA9, 0x93, 0xB5, 0x5F,
	0x61, 0x0B, 0x2D, 0x17, 0x99, 0x03, 0x25, 0x4F,
	0x51, 0x7B, 0x9D, 0x07, 0x89, 0x73, 0x95, 0x3F,
	0x41, 0xEB, 0x0D, 0xF7, 0x79, 0xE3, 0x05, 0x2F,
	0x31, 0x5B, 0x7D, 0xE7, 0x69, 0x53, 0x75, 0x1F,
	0x21, 0xCB, 0xED, 0xD7, 0x59, 0xC3, 0xE5, 0x0F,
	0x11, 0x3B, 0x5D, 0xC7, 0x49, 0x33, 0x55, 0xFF
};

#define binvert_limb(inv,n)                                             \
  do {                                                                  \
    mp_limb_t  __n = (n);                                               \
    mp_limb_t  __inv;                                                   \
    assert ((__n & 1) == 1);                                            \
                                                                        \
    __inv = binvert_limb_table[(__n/2) & 0x7F]; /*  8 */                \
    if (GMP_LIMB_BITS > 8)   __inv = 2 * __inv - __inv * __inv * __n;   \
    if (GMP_LIMB_BITS > 16)  __inv = 2 * __inv - __inv * __inv * __n;   \
    if (GMP_LIMB_BITS > 32)  __inv = 2 * __inv - __inv * __inv * __n;   \
                                                                        \
    assert ((__inv * __n) == 1);                        \
    (inv) = __inv;                                      \
  } while (0)

static void setup_fermat(int N_Size, int num, const mp_limb_t* M, mp_limb_t* MI, mp_limb_t* R)
{
	assert(N_Size <= MAX_N_SIZE);
	for (int j = 0; j < num; ++j)
	{
		mp_size_t mn = N_Size;
		mp_limb_t mshifted[MAX_N_SIZE];
		mp_srcptr mp;
		mp_ptr rp;
		struct gmp_div_inverse minv;

		// REDCify: r = B^n * 2 % M
		mp = &M[j*N_Size];
		rp = &R[j*N_Size];
		mpn_div_qr_invert(&minv, mp, mn);

		if (minv.shift > 0)
		{
			mpn_lshift(mshifted, mp, mn, minv.shift);
			mp = mshifted;
		}
		else
		{
			for (size_t i = 0; i < mn; ++i) mshifted[i] = mp[i];
		}

		for (size_t i = 0; i < mn; ++i) rp[i] = 0;
		rp[mn] = 1 << minv.shift;
		mpn_div_r_preinv_ns(rp, mn + 1, mp, mn, &minv);

		if (minv.shift > 0)
		{
			mpn_rshift(rp, rp, mn, minv.shift);
			mp = &M[j*N_Size];
		}

		mp_limb_t mi;
		binvert_limb(mi, mp[0]);
		MI[j] = -mi;
	}
}

#if DEBUG
#define DPRINTF(fmt, args...) do { printf("line %d: " fmt, __LINE__, ##args); fflush(stdout); } while(0)
#else
#define DPRINTF(fmt, ...) do { } while(0)
#endif

typedef struct PrimeTestCxt
{
	uint* m_mem_obj;
	uint* mi_mem_obj;
	uint* r_mem_obj;
	uint* is_prime_mem_obj;

	uint *R;
	uint *MI;
} PrimeTestCxt;

PrimeTestCxt* primeTestInit()
{
	cudaError_t cudaStatus;

	PrimeTestCxt* cxt = (PrimeTestCxt*)malloc(sizeof(PrimeTestCxt));

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return NULL;
	}

	// Create memory buffers on the device for each vector 
	cudaStatus = cudaMalloc((void**)&cxt->m_mem_obj, MAX_JOB_SIZE * MAX_N_SIZE * sizeof(uint));
	cudaStatus = cudaMalloc((void**)&cxt->mi_mem_obj, MAX_JOB_SIZE * sizeof(uint));
	cudaStatus = cudaMalloc((void**)&cxt->r_mem_obj, MAX_JOB_SIZE * MAX_N_SIZE * sizeof(uint));
	cudaStatus = cudaMalloc((void**)&cxt->is_prime_mem_obj, MAX_JOB_SIZE * sizeof(uint));

	// Create buffers on host
	cxt->R = (uint*)malloc(sizeof(uint)*(MAX_N_SIZE*MAX_JOB_SIZE + 1));
	cxt->MI = (uint*)malloc(sizeof(uint)*MAX_JOB_SIZE);

	return cxt;
}

void primeTest(PrimeTestCxt* cxt, int N_Size, int listSize, const uint* M, uint* is_prime)
{
	cudaError_t cudaStatus;

	int nextJobSize = min(MAX_JOB_SIZE, listSize);

	if (nextJobSize > 0)
	{
		setup_fermat(N_Size, nextJobSize, M, cxt->MI, cxt->R);
	}

	while (nextJobSize > 0)
	{
		int jobSize = nextJobSize;
		listSize -= jobSize;
		nextJobSize = min(MAX_JOB_SIZE, listSize);

		// Copy the lists A and B to their respective memory buffers
		cudaStatus = cudaMemcpy(cxt->mi_mem_obj, cxt->MI, jobSize * sizeof(uint), cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpy(cxt->r_mem_obj, cxt->R, jobSize * N_Size * sizeof(uint), cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpy(cxt->m_mem_obj, M, jobSize * N_Size * sizeof(uint), cudaMemcpyHostToDevice);

		int blockSize = 1;
		int numBlocks = jobSize;
		while (blockSize < MAX_BLOCK_SIZE && ((numBlocks & 1) == 0))
		{
			numBlocks >>= 1;
			blockSize <<= 1;
		}

		DPRINTF("before execution\n");
		switch (N_Size)
		{
		case 3: fermat_test<3> << <numBlocks, blockSize >> >(cxt->m_mem_obj, cxt->mi_mem_obj, cxt->r_mem_obj, cxt->is_prime_mem_obj); break;
		case 20: fermat_test<20> << <numBlocks, blockSize >> >(cxt->m_mem_obj, cxt->mi_mem_obj, cxt->r_mem_obj, cxt->is_prime_mem_obj); break;
		case 40: fermat_test<40> << <numBlocks, blockSize >> >(cxt->m_mem_obj, cxt->mi_mem_obj, cxt->r_mem_obj, cxt->is_prime_mem_obj); break;
		case 53: fermat_test<53> <<<numBlocks, blockSize >>>(cxt->m_mem_obj, cxt->mi_mem_obj, cxt->r_mem_obj, cxt->is_prime_mem_obj); break;
		default: abort();
		}
		
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return;
		}

		if (nextJobSize > 0)
		{
			M += jobSize*N_Size;
			setup_fermat(N_Size, nextJobSize, M, cxt->MI, cxt->R);
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
			return;
		}

		cudaStatus = cudaMemcpy(is_prime, cxt->is_prime_mem_obj, jobSize * sizeof(uint), cudaMemcpyDeviceToHost);

		is_prime += jobSize;
	}
}

void primeTestTerm(PrimeTestCxt* cxt)
{
	cudaFree(cxt->mi_mem_obj);
	cudaFree(cxt->m_mem_obj);
	cudaFree(cxt->r_mem_obj);
	cudaFree(cxt->is_prime_mem_obj);

	free(cxt->R);
	free(cxt->MI);
	free(cxt);

	cudaDeviceReset();
}