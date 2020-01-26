/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#define	RED_BLK		4

#define	R64		(RED_BLK * 64 / 4)

__kernel __attribute__((reqd_work_group_size(R64, 1, 1)))
void reduce_upsweep64(__global uint * restrict const t, const uint d, const uint s, const uint j)
{
	__local uint4 T_4[R64 / 4 + R64 / 16];	// alignment
	__local uint4 * const T2_4 = &T_4[R64 / 4];
	__local uint * const T = (__local uint *)T_4;
	__local uint * const T_2 = (__local uint *)T2_4;

	const size_t i = get_local_id(0), blk = get_group_id(0), k = get_global_id(0);	// blk * R64 + i;
	__global uint * const tj = &t[j];

	__global const uint4 * const tj1_4 = (__global const uint4 *)&tj[0];
	const uint4 u = tj1_4[k];
	const uint u01 = _addmod(u.s0, u.s1, d), u23 = _addmod(u.s2, u.s3, d), u0123 = _addmod(u01, u23, d);
	tj[4 * (16 * s) + k] = u23; tj[5 * (16 * s) + k] = u0123; T[i] = u0123;

	barrier(CLK_LOCAL_MEM_FENCE);

	if (i < R64 / 4)
	{
		const size_t k_4 = blk * R64 / 4 + i;
		const uint4 u = T_4[i];
		const uint u01 = _addmod(u.s0, u.s1, d), u23 = _addmod(u.s2, u.s3, d), u0123 = _addmod(u01, u23, d);
		tj[5 * (16 * s) + 4 * (4 * s) + k_4] = u23; tj[5 * (16 * s) + 5 * (4 * s) + k_4] = u0123; T_2[i] = u0123;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (i < R64 / 16)
	{
		const size_t k_16 = blk * R64 / 16 + i;
		const uint4 u = T2_4[i];
		const uint u01 = _addmod(u.s0, u.s1, d), u23 = _addmod(u.s2, u.s3, d), u0123 = _addmod(u01, u23, d);
		tj[5 * (16 * s) + 5 * (4 * s) + 4 * s + k_16] = u23; tj[5 * (16 * s) + 5 * (4 * s) + 5 * s + k_16] = u0123;
	}
}

__kernel __attribute__((reqd_work_group_size(R64, 1, 1)))
void reduce_downsweep64(__global uint * restrict const t, const uint d, const uint s, const uint j)
{
	__local uint4 T_4[R64 / 4 + R64 / 16];	// alignment
	__local uint4 * const T2_4 = &T_4[R64 / 4];
	__local uint * const T = (__local uint *)T_4;
	__local uint * const T_2 = (__local uint *)T2_4;

	const size_t i = get_local_id(0), blk = get_group_id(0), k = get_global_id(0);	// blk * R64 + i;
	__global uint * const tj = &t[j];

	if (i < R64 / 16)
	{
		const size_t k_16 = blk * R64 / 16 + i;
		__global const uint4 * const tj16_4 = (__global uint4 *)&tj[5 * (16 * s) + 5 * (4 * s)];
		const uint u2 = tj[5 * (16 * s) + 5 * (4 * s) + 4 * s + k_16], u0 = tj[5 * (16 * s) + 5 * (4 * s) + 5 * s + k_16], u02 = _addmod(u0, u2, d);
		const uint4 u13 = tj16_4[k_16];
		const uint u012 = _addmod(u02, u13.s1, d), u03 = _addmod(u0, u13.s3, d);
		T2_4[i] = (uint4)(u012, u02, u03, u0);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (i < R64 / 4)
	{
		const size_t k_4 = blk * R64 / 4 + i;
		__global const uint4 * const tj4_4 = (__global uint4 *)&tj[5 * (16 * s)];
		const uint u2 = tj[5 * (16 * s) + 4 * (4 * s) + k_4], u0 = T_2[i], u02 = _addmod(u0, u2, d);
		const uint4 u13 = tj4_4[k_4];
		const uint u012 = _addmod(u02, u13.s1, d), u03 = _addmod(u0, u13.s3, d);
		T_4[i] = (uint4)(u012, u02, u03, u0);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	__global uint4 * const tj1_4 = (__global uint4 *)&tj[0];
	const uint u2 = tj[4 * (16 * s) + k], u0 = T[i], u02 = _addmod(u0, u2, d);
	const uint4 u13 = tj1_4[k];
	const uint u012 = _addmod(u02, u13.s1, d), u03 = _addmod(u0, u13.s3, d);
	tj1_4[k] = (uint4)(u012, u02, u03, u0);
}

inline void _reduce_upsweep4i(__local uint * restrict const T, __global const uint * restrict const t, const uint d, const uint s, const size_t k)
{
	__global const uint4 * const ti = (__global const uint4 *)&t[4 * k];
	__local uint * const To = &T[k];

	const uint4 u = ti[0];
	const uint u01 = _addmod(u.s0, u.s1, d), u23 = _addmod(u.s2, u.s3, d), u0123 = _addmod(u01, u23, d);
	To[0] = u23; To[s] = u0123;
}

inline void _reduce_upsweep4(__local uint * restrict const T, const uint d, const uint s, const size_t k)
{
	__local const uint * const Ti = &T[0 * s + 4 * k];
	__local uint * const To = &T[4 * s + 1 * k];

	const uint u0 = Ti[0], u1 = Ti[1], u2 = Ti[2], u3 = Ti[3];
	const uint u01 = _addmod(u0, u1, d), u23 = _addmod(u2, u3, d), u0123 = _addmod(u01, u23, d);
	To[0] = u23; To[s] = u0123;
}

inline void _reduce_downsweep4(__local uint * restrict const T, const uint d, const uint s, const size_t k)
{
	__local const uint * const Ti = &T[4 * s + 1 * k];
	__local uint * const To = &T[0 * s + 4 * k];

	const uint u2 = Ti[0], u0 = Ti[s], u02 = _addmod(u0, u2, d);
	const uint u1 = To[1], u3 = To[3];
	const uint u012 = _addmod(u02, u1, d), u03 = _addmod(u0, u3, d);
	To[0] = u012; To[1] = u02; To[2] = u03; To[3] = u0;
}

inline void _reduce_downsweep4o(__global uint * restrict const t, __local const uint * restrict const T, const uint d, const uint s, const size_t k)
{
	__local const uint * const Ti = &T[k];
	__global uint4 * const to = (__global uint4 *)&t[4 * k];

	const uint u2 = Ti[0], u0 = Ti[s], u02 = _addmod(u0, u2, d);
	const uint4 u13 = to[0];
	const uint u012 = _addmod(u02, u13.s1, d), u03 = _addmod(u0, u13.s3, d);
	to[0] = (uint4)(u012, u02, u03, u0);
}

inline void _reduce_topsweep2(__global uint * restrict const t, __local uint * restrict const T, const uint d)
{
	const uint u0 = T[0], u1 = T[1];
	const uint u01 = _addmod(u0, u1, d);
	t[0] = u01;
	T[0] = u1; T[1] = 0;
}

inline void _reduce_topsweep4(__global uint * restrict const t, __local uint * restrict const T, const uint d)
{
	const uint u0 = T[0], u1 = T[1], u2 = T[2], u3 = T[3];
	const uint u01 = _addmod(u0, u1, d), u23 = _addmod(u2, u3, d);
	const uint u123 = _addmod(u1, u23, d), u0123 = _addmod(u01, u23, d);
	t[0] = u0123;
	T[0] = u123; T[1] = u23; T[2] = u3; T[3] = 0;
}

#define	S32		(32 / 4)
__kernel __attribute__((reqd_work_group_size(S32, 1, 1)))
void reduce_topsweep32(__global uint * restrict const t, const uint d, const uint j)
{
	__local uint T[64];	// 20

	const size_t i = get_local_id(0);

	_reduce_upsweep4i(T, &t[j], d, S32, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S32 / 4) _reduce_upsweep4(&T[S32], d, S32 / 4, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i == 0) _reduce_topsweep2(t, &T[S32 + 5 * (S32 / 4)], d);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S32 / 4) _reduce_downsweep4(&T[S32], d, S32 / 4, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	_reduce_downsweep4o(&t[j], T, d, S32, i);
}

#define	S64		(64 / 4)
__kernel __attribute__((reqd_work_group_size(S64, 1, 1)))
void reduce_topsweep64(__global uint * restrict const t, const uint d, const uint j)
{
	__local uint T[64];	// 40

	const size_t i = get_local_id(0);

	_reduce_upsweep4i(T, &t[j], d, S64, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S64 / 4) _reduce_upsweep4(&T[S64], d, S64 / 4, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i == 0) _reduce_topsweep4(t, &T[S64 + 5 * (S64 / 4)], d);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S64 / 4) _reduce_downsweep4(&T[S64], d, S64 / 4, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	_reduce_downsweep4o(&t[j], T, d, S64, i);
}

#define	S128	(128 / 4)
__kernel __attribute__((reqd_work_group_size(S128, 1, 1)))
void reduce_topsweep128(__global uint * restrict const t, const uint d, const uint j)
{
	__local uint T[128];	// 82

	const size_t i = get_local_id(0);

	_reduce_upsweep4i(T, &t[j], d, S128, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S128 / 4) _reduce_upsweep4(&T[S128], d, S128 / 4, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S128 / 16) _reduce_upsweep4(&T[S128 + 5 * (S128 / 4)], d, S128 / 16, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i == 0) _reduce_topsweep2(t, &T[S128 + 5 * (S128 / 4) + 5 * (S128 / 16)], d);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S128 / 16) _reduce_downsweep4(&T[S128 + 5 * (S128 / 4)], d, S128 / 16, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S128 / 4) _reduce_downsweep4(&T[S128], d, S128 / 4, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	_reduce_downsweep4o(&t[j], T, d, S128, i);
}

#define	S256	(256 / 4)
__kernel __attribute__((reqd_work_group_size(S256, 1, 1)))
void reduce_topsweep256(__global uint * restrict const t, const uint d, const uint j)
{
	__local uint T[256];	// 168

	const size_t i = get_local_id(0);

	_reduce_upsweep4i(T, &t[j], d, S256, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S256 / 4) _reduce_upsweep4(&T[S256], d, S256 / 4, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S256 / 16) _reduce_upsweep4(&T[S256 + 5 * (S256 / 4)], d, S256 / 16, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i == 0) _reduce_topsweep4(t, &T[S256 + 5 * (S256 / 4) + 5 * (S256 / 16)], d);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S256 / 16) _reduce_downsweep4(&T[S256 + 5 * (S256 / 4)], d, S256 / 16, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S256 / 4) _reduce_downsweep4(&T[S256], d, S256 / 4, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	_reduce_downsweep4o(&t[j], T, d, S256, i);
}

#define	S512	(512 / 4)
__kernel __attribute__((reqd_work_group_size(S512, 1, 1)))
void reduce_topsweep512(__global uint * restrict const t, const uint d, const uint j)
{
	__local uint T[512];	// 340

	const size_t i = get_local_id(0);

	_reduce_upsweep4i(T, &t[j], d, S512, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S512 / 4) _reduce_upsweep4(&T[S512], d, S512 / 4, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S512 / 16) _reduce_upsweep4(&T[S512 + 5 * (S512 / 4)], d, S512 / 16, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S512 / 64) _reduce_upsweep4(&T[S512 + 5 * (S512 / 4) + 5 * (S512 / 16)], d, S512 / 64, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i == 0) _reduce_topsweep2(t, &T[S512 + 5 * (S512 / 4) + 5 * (S512 / 16) + 5 * (S512 / 64)], d);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S512 / 64) _reduce_downsweep4(&T[S512 + 5 * (S512 / 4) + 5 * (S512 / 16)], d, S512 / 64, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S512 / 16) _reduce_downsweep4(&T[S512 + 5 * (S512 / 4)], d, S512 / 16, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S512 / 4) _reduce_downsweep4(&T[S512], d, S512 / 4, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	_reduce_downsweep4o(&t[j], T, d, S512, i);
}

#define	S1024	(1024 / 4)
__kernel __attribute__((reqd_work_group_size(S1024, 1, 1)))
void reduce_topsweep1024(__global uint * restrict const t, const uint d, const uint j)
{
	__local uint T[1024];	// 680

	const size_t i = get_local_id(0);

	_reduce_upsweep4i(T, &t[j], d, S1024, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S1024 / 4) _reduce_upsweep4(&T[S1024], d, S1024 / 4, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S1024 / 16) _reduce_upsweep4(&T[S1024 + 5 * (S1024 / 4)], d, S1024 / 16, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S1024 / 64) _reduce_upsweep4(&T[S1024 + 5 * (S1024 / 4) + 5 * (S1024 / 16)], d, S1024 / 64, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i == 0) _reduce_topsweep4(t, &T[S1024 + 5 * (S1024 / 4) + 5 * (S1024 / 16) + 5 * (S1024 / 64)], d);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S1024 / 64) _reduce_downsweep4(&T[S1024 + 5 * (S1024 / 4) + 5 * (S1024 / 16)], d, S1024 / 64, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S1024 / 16) _reduce_downsweep4(&T[S1024 + 5 * (S1024 / 4)], d, S1024 / 16, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i < S1024 / 4) _reduce_downsweep4(&T[S1024], d, S1024 / 4, i);
	barrier(CLK_LOCAL_MEM_FENCE);
	_reduce_downsweep4o(&t[j], T, d, S1024, i);
}

__kernel
void reduce_i(__global const uint2 * restrict const x, __global uint * restrict const y, __global uint * restrict const t,
	__global const uint * restrict const bp, const uint4 e_d_d_inv_d_shift, const int s)
{
	const size_t k = get_global_id(0);

	const uint e = e_d_d_inv_d_shift.s0, d = e_d_d_inv_d_shift.s1, d_inv = e_d_d_inv_d_shift.s2;
	const int d_shift = (int)(e_d_d_inv_d_shift.s3);

	const uint xs = ((x[e + k + 1].s0 << (digit_bit - s)) | (x[e + k].s0 >> s)) & digit_mask;
	const uint u = _rem(xs * (ulong)(bp[k]), d, d_inv, d_shift);

	y[k] = xs;
	t[k + 4] = u;
}

__kernel
void reduce_o(__global uint2 * restrict const x, __global const uint * restrict const y, __global const uint * restrict const t,
	__global const uint * restrict const ibp, const uint4 e_d_d_inv_d_shift)
{
	const size_t n = get_global_size(0), k = get_global_id(0);

	const uint e = e_d_d_inv_d_shift.s0, d = e_d_d_inv_d_shift.s1, d_inv = e_d_d_inv_d_shift.s2;
	const int d_shift = (int)(e_d_d_inv_d_shift.s3);

	const uint tk = t[k + 4];
	//const uint rbk_prev = (k + 1 != n) ? tk : 0;	// NVidia compiler generates a conditionnal branch instruction then the code must be written with a mask
	const uint mask = (k + 1 != n) ? (uint)(-1) : 0;
	const uint rbk_prev = tk & mask;
	const uint r_prev = _rem(rbk_prev * (ulong)(ibp[k]), d, d_inv, d_shift);

	const ulong q = ((ulong)(r_prev) << digit_bit) | y[k];

	const uint q_d = mul_hi((uint)(q >> d_shift), d_inv);	// d < 2^29
	const uint r = (uint)(q) - q_d * d;
	const uint c = (r >= d) ? 1 : 0;

	const uint2 x_k = x[k];
	x[k] = (uint2)((k > e) ? 0 : x_k.s0, q_d + c);
}

__kernel
void reduce_f(__global uint2 * restrict const x, __global const uint * restrict const t, const uint n, const uint e, const int s)
{
	const uint rs = x[e].s0 & ((1u << s) - 1);
	ulong l = ((ulong)(t[0]) << s) | rs;		// rds < 2^(29 + digit_bit - 1)

	x[e].s0 = (uint)(l) & digit_mask;
	l >>= digit_bit;

	for (size_t k = e + 1; l != 0; ++k)
	{
		x[k].s0 = (uint)(l) & digit_mask;
		l >>= digit_bit;
	}
}

__kernel
void reduce_x(__global uint2 * restrict const x, const uint n, __global int * const err)
{
	int c = 0;
	for (size_t k = 0; k < n; ++k)
	{
		const uint2 x_k = x[k];
		c += x_k.s0 - x_k.s1;
		x[k] = (uint2)((uint)(c) & digit_mask, 0);
		c >>= digit_bit;
	}

	if (c != 0) atomic_or(err, c);
}

__kernel
void reduce_z(__global uint2 * restrict const x, const uint n, __global int * const err)
{
	// s0 = x, s1 = k.2^n + 1
	// if s0 >= s1, s0 -= s1;

	for (size_t i = 0; i < n; ++i)
	{
		const size_t j = n - 1 - i;
		const uint2 x_j = x[j];
		if (x_j.s0 < x_j.s1) return;
		if (x_j.s0 > x_j.s1) break;
	}

	int c = 0;
	for (size_t k = 0; k < n; ++k)
	{
		const uint2 x_k = x[k];
		c += x_k.s0 - x_k.s1;
		x[k] = (uint2)((uint)(c) & digit_mask, 0);
		c >>= digit_bit;
	}

	if (c != 0) atomic_or(err, c);
}