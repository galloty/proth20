/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#define	P2I_WGS		16
#define	P2I_BLK		16

__kernel __attribute__((reqd_work_group_size(P2I_WGS, 1, 1)))
void poly2int0(__global uint2 * restrict const x, __global long * restrict const cr, const uint2 norm)
{
	__local long L[P2I_WGS * P2I_BLK];
	__local uint X[P2I_WGS * P2I_BLK];

	const size_t i = get_local_id(0), blk = get_group_id(0);
	const size_t kc = (get_global_id(0) + 1) & (get_global_size(0) - 1);

	__global uint2 * const xo = &x[P2I_WGS * P2I_BLK * blk];

	for (size_t j = 0; j < P2I_BLK; ++j)
	{
		const size_t k = P2I_WGS * j + i;
		L[P2I_WGS * (k % P2I_BLK) + (k / P2I_BLK)] = getlong(mulmod(xo[k], norm));	// -n/2 . (B-1)^2 <= l <= n/2 . (B-1)^2
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	long l = 0;
//#pragma unroll
	for (size_t j = 0; j < P2I_BLK; ++j)
	{
		l += L[P2I_WGS * j + i];
		X[P2I_WGS * j + i] = (uint)(l) & digit_mask;
	 	l >>= digit_bit;
	}
	cr[kc] = l;

	barrier(CLK_LOCAL_MEM_FENCE);

//#pragma unroll
	for (size_t j = 0; j < P2I_BLK; ++j)
	{
		const size_t k = P2I_WGS * j + i;
		xo[k].s0 = X[P2I_WGS * (k % P2I_BLK) + (k / P2I_BLK)];
	}
}

__kernel
void poly2int1(__global uint2 * restrict const x, __global const long * restrict const cr, __global int * const err)
{
	const size_t k = get_global_id(0);

	__global uint2 * const xi = &x[P2I_BLK * k];

	long l = cr[k] + xi[0].s0;
	xi[0].s0 = (uint)(l) & digit_mask;
	l >>= digit_bit;						// |l| < n/2

	int f = (int)(l);
#pragma unroll
	for (size_t j = 1; j < P2I_BLK; ++j)
	{
		f += xi[j].s0;
		xi[j].s0 = (uint)(f) & digit_mask;
		f >>= digit_bit;					// f = -1, 0 or 1
		if (f == 0) break;
	}

	if (f != 0) atomic_or(err, f);
}
