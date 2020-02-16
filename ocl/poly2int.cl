/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#define POLY2INT0_VAR(P2I_BLK, P2I_WGS) \
	__local long L[P2I_WGS * P2I_BLK]; \
	__local uint X[P2I_WGS * P2I_BLK];
 
inline void poly2int0(__local long * restrict const L, __local uint * restrict const X, const size_t P2I_BLK, const size_t P2I_WGS,
 	__global uint2 * restrict const x, __global long * restrict const cr)
 {
	const size_t i = get_local_id(0), blk = get_group_id(0);
	const size_t kc = (get_global_id(0) + 1) & (get_global_size(0) - 1);

	__global uint2 * const xo = &x[P2I_WGS * P2I_BLK * blk];

	for (size_t j = 0; j < P2I_BLK; ++j)
	{
		const size_t k = P2I_WGS * j + i;
		L[P2I_WGS * (k % P2I_BLK) + (k / P2I_BLK)] = getlong(mulmod(xo[k], pconst_norm));	// -n/2 . (B-1)^2 <= l <= n/2 . (B-1)^2
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

inline void poly2int1(const size_t P2I_BLK, __global uint2 * restrict const x, __global const long * restrict const cr, __global int * const err)
{
	const size_t k = get_global_id(0);

	__global uint2 * const xi = &x[P2I_BLK * k];

	long l = cr[k] + xi[0].s0;
	xi[0].s0 = (uint)(l) & digit_mask;
	l >>= digit_bit;						// |l| < n/2

	int f = (int)(l);
//#pragma unroll
	for (size_t j = 1; j < P2I_BLK - 1; ++j)
	{
		f += xi[j].s0;
		xi[j].s0 = (uint)(f) & digit_mask;
		f >>= digit_bit;					// f = -1, 0 or 1
		if (f == 0) return;
	}

	f += xi[P2I_BLK - 1].s0;
	xi[P2I_BLK - 1].s0 = (uint)(f);
	f >>= digit_bit;
	if (f != 0) atomic_or(&err[1], f);
}

__kernel
void poly2int2(__global uint2 * restrict const x, __global int * const err)
{
	if (err[1] == 0) return;

	int f = 0;
	for (size_t k = 0; k < pconst_size; ++k)
	{
		f += x[k].s0;
		x[k] = (uint)(f) & digit_mask;
		f >>= digit_bit;
	}

	err[0] = f;
	err[1] = 0;
}

// P2I_BLK = 4

__kernel __attribute__((reqd_work_group_size(16, 1, 1)))
void poly2int0_4_16(__global uint2 * restrict const x, __global long * restrict const cr)
{
	POLY2INT0_VAR(4, 16);
	poly2int0(L, X, 4, 16, x, cr);
}

__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void poly2int0_4_32(__global uint2 * restrict const x, __global long * restrict const cr)
{
	POLY2INT0_VAR(4, 32);
	poly2int0(L, X, 4, 32, x, cr);
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void poly2int0_4_64(__global uint2 * restrict const x, __global long * restrict const cr)
{
	POLY2INT0_VAR(4, 64);
	poly2int0(L, X, 4, 64, x, cr);
}

__kernel
void poly2int1_4(__global uint2 * restrict const x, __global const long * restrict const cr, __global int * const err)
{
	poly2int1(4, x, cr, err);
}

// P2I_BLK = 8

__kernel __attribute__((reqd_work_group_size(16, 1, 1)))
void poly2int0_8_16(__global uint2 * restrict const x, __global long * restrict const cr)
{
	POLY2INT0_VAR(8, 16);
	poly2int0(L, X, 8, 16, x, cr);
}

__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void poly2int0_8_32(__global uint2 * restrict const x, __global long * restrict const cr)
{
	POLY2INT0_VAR(8, 32);
	poly2int0(L, X, 8, 32, x, cr);
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void poly2int0_8_64(__global uint2 * restrict const x, __global long * restrict const cr)
{
	POLY2INT0_VAR(8, 64);
	poly2int0(L, X, 8, 64, x, cr);
}

__kernel
void poly2int1_8(__global uint2 * restrict const x, __global const long * restrict const cr, __global int * const err)
{
	poly2int1(8, x, cr, err);
}

// P2I_BLK = 16

__kernel __attribute__((reqd_work_group_size(8, 1, 1)))
void poly2int0_16_8(__global uint2 * restrict const x, __global long * restrict const cr)
{
	POLY2INT0_VAR(16, 8);
	poly2int0(L, X, 16, 8, x, cr);
}

__kernel __attribute__((reqd_work_group_size(16, 1, 1)))
void poly2int0_16_16(__global uint2 * restrict const x, __global long * restrict const cr)
{
	POLY2INT0_VAR(16, 16);
	poly2int0(L, X, 16, 16, x, cr);
}

__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void poly2int0_16_32(__global uint2 * restrict const x, __global long * restrict const cr)
{
	POLY2INT0_VAR(16, 32);
	poly2int0(L, X, 16, 32, x, cr);
}

__kernel
void poly2int1_16(__global uint2 * restrict const x, __global const long * restrict const cr, __global int * const err)
{
	poly2int1(16, x, cr, err);
}
