#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable

// Compute 2^(M-1) mod M
// R is B^n * 2 % M
__kernel void fermat_test(__global const uint *M_in, __global const uint *Mi_in, __global const uint *R_in, __global uint *is_prime) {

	uint R[N_Size];

	// Get the index of the current element to be processed
	const int offset = get_global_id(0) * N_Size;
	__global const uint* M = &M_in[offset];
	const uint shift = clz(M[N_Size - 1]);

	for (int i = 0; i < N_Size; ++i)
	{
		R[i] = R_in[offset + i];
	}

	const uint highbit = ((uint)1) << 31;
	uint startbit = highbit >> shift;

	const uint mi = Mi_in[get_global_id(0)];

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
				P[1] = mul_hi(R[0], R[0]);
				{
					uint T[N_Size * 2];

					{
						uint cy = 0;
						for (int i = 0; i < N_Size - 1; ++i)
						{
							T[i] = R[i + 1] * R[0];
							T[i] += cy;
							cy = (T[i] < cy) + mul_hi(R[i+1], R[0]);
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
							cy = (lp < cy) + mul_hi(R[i], R[j - 1]);
							lp += T[i + j - 2];
							cy += lp < T[i + j - 2];
							T[i + j - 2] = lp;
						}
						T[N_Size + j - 2] = cy;
					}

					for (int i = 0; i < N_Size; ++i)
					{
						P[2 * i] = R[i] * R[i];
						P[2 * i + 1] = mul_hi(R[i], R[i]);
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
						cy = (lp < cy) + mul_hi(M[i], v);
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
				cy = (lp < cy) + mul_hi(M[i], v);
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

	is_prime[get_global_id(0)] = result;
}