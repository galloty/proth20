/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

// __global is faster than __constant on NVIDIA GPU
//#define __constmem	__constant
#define __constmem	__global

#define	digit_bit		21
#define digit_mask		((1u << digit_bit) - 1)

#define	P1			2130706433u		// 127 * 2^24 + 1 = 2^31 - 2^24 + 1
#define	P2			2013265921u		//  15 * 2^27 + 1 = 2^31 - 2^27 + 1
#define	P1_INV		2164392967u		// 2^62 / P1
#define	P2_INV 		2290649223u		// 2^62 / P2
#define	P1_I		2113994754u		// P1_PRIM_ROOT^((P1 - 1) / 4)
#define	P2_I		1728404513u		// P2_PRIM_ROOT^((P2 - 1) / 4)
#define	InvP2_P1	913159918u		// 1 / P2 mod P1
#define	P1P2		(P1 * (ulong)(P2))

#define CHUNK64		16
#define BLK32		8
#define BLK64		4
#define BLK128		2
#define BLK256		1
#define	P2I_WGS		16
#define	P2I_BLK		16

/*
Barrett's product/reduction, where P is such that h (the number of iterations in the 'while loop') is 0 or 1.

Let m < P^2 < 2^62, R = [2^62/P] and h = [m/P] - [[m/2^30] R / 2^32].

We have h = ([m/P] - m/P) + m/2^62 (2^62/P - R) + R/2^32 (m/2^30 - [m/2^30]) + ([m/2^30] R / 2^32 - [[m/2^30] * R / 2^32]).
Then -1 + 0 + 0 + 0 < h < 0 + (2^62/P - R) + R/2^32 + 1,
0 <= h < 1 + (2^62/P - R) + R/2^32.

P = 127 * 2^24 + 1 = 2130706433 => R = 2164392967, h < 1.56
P =  63 * 2^25 + 1 = 2113929217 => R = 2181570688, h < 2.51 NOK
P =  15 * 2^27 + 1 = 2013265921 => R = 2290649223, h < 1.93
*/

inline uint _addmod(const uint lhs, const uint rhs, const uint d)
{
	const uint r = lhs + rhs;
	const uint t = (lhs >= d - rhs) ? d : 0;
	return r - t;
}

inline uint _rem(const ulong q, const uint d, const uint d_inv, const int d_shift)
{
	const uint q_d = mul_hi((uint)(q >> d_shift), d_inv);
	const uint r = (uint)(q) - q_d * d;
	const uint t = (r >= d) ? d : 0;
	return r - t;
}

inline uint _mulMod(const uint a, const uint b, const uint p, const uint p_inv)
{
	return _rem(a * (ulong)(b), p, p_inv, 30);
}

inline uint mulModP1(const uint a, const uint b) { return _mulMod(a, b, P1, P1_INV); }
inline uint mulModP2(const uint a, const uint b) { return _mulMod(a, b, P2, P2_INV); }

inline long getlong(const uint2 lhs)
{
	// Garner Algorithm
	uint d = lhs.s0 - lhs.s1; const uint t = (lhs.s0 < lhs.s1) ? P1 : 0; d += t;	// mod P1
	const uint u = mulModP1(d, InvP2_P1);		// P2 < P1
	const ulong r = u * (ulong)(P2) + lhs.s1;
	const ulong s = (r > P1P2 / 2) ? P1P2 : 0;
	return (long)(r - s);
}

inline uint2 addmod(const uint2 lhs, const uint2 rhs)
{
	const uint2 r = lhs + rhs;
	const uint2 t = (uint2)((lhs.s0 >= P1 - rhs.s0) ? P1 : 0, (lhs.s1 >= P2 - rhs.s1) ? P2 : 0);
	return r - t;
}

inline uint2 submod(const uint2 lhs, const uint2 rhs)
{
	const uint2 r = lhs - rhs;
	const uint2 t = (uint2)((lhs.s0 < rhs.s0) ? P1 : 0, (lhs.s1 < rhs.s1) ? P2 : 0);
	return r + t;
}

inline uint2 mulmod(const uint2 lhs, const uint2 rhs)
{
	return (uint2)(mulModP1(lhs.s0, rhs.s0), mulModP2(lhs.s1, rhs.s1));
}

inline uint2 sqrmod(const uint2 lhs) { return mulmod(lhs, lhs); }

inline uint2 mulI(const uint2 lhs) { return mulmod(lhs, (uint2)(P1_I, P2_I)); }

inline void _sub_forward4i(const size_t ml, __local uint2 * restrict const X, const size_t mg, __global const uint2 * restrict const x, const uint2 r2, const uint4 r1ir1)
{
	const uint2 abi = x[0 * mg], abim = x[1 * mg];
	const uint2 abi0 = (uint2)(abi.s0, abi.s0), abi1 = (uint2)(abi.s1, abi.s1);
	const uint2 abim0 = (uint2)(abim.s0, abim.s0), abim1 = (uint2)(abim.s1, abim.s1);
	const uint2 u0 = submod(abi0, abi1), u1 = submod(abim0, abim1), u3 = mulI(u1);
	X[0 * ml] = addmod(u0, u1); X[1 * ml] = mulmod(submod(u0, u1), r2);
	X[2 * ml] = mulmod(submod(u0, u3), r1ir1.s23); X[3 * ml] = mulmod(addmod(u0, u3), r1ir1.s01);
}

inline void _forward4i(const size_t ml, __local uint2 * restrict const X, const size_t mg, __global const uint2 * restrict const x, const uint2 r2, const uint4 r1ir1)
{
	const uint2 u0 = x[0 * mg], u2 = x[2 * mg], u1 = x[1 * mg], u3 = x[3 * mg];
	const uint2 v0 = addmod(u0, u2), v2 = submod(u0, u2), v1 = addmod(u1, u3), v3 = mulI(submod(u3, u1));
	X[0 * ml] = addmod(v0, v1); X[1 * ml] = mulmod(submod(v0, v1), r2);
	X[2 * ml] = mulmod(addmod(v2, v3), r1ir1.s23); X[3 * ml] = mulmod(submod(v2, v3), r1ir1.s01);
}

inline void _forward4(const size_t m, __local uint2 * restrict const X, const uint2 r2, const uint4 r1ir1)
{
	barrier(CLK_LOCAL_MEM_FENCE);

	const uint2 u0 = X[0 * m], u2 = X[2 * m], u1 = X[1 * m], u3 = X[3 * m];
	const uint2 v0 = addmod(u0, u2), v2 = submod(u0, u2), v1 = addmod(u1, u3), v3 = mulI(submod(u3, u1));
	X[0 * m] = addmod(v0, v1); X[1 * m] = mulmod(submod(v0, v1), r2);
	X[2 * m] = mulmod(addmod(v2, v3), r1ir1.s23); X[3 * m] = mulmod(submod(v2, v3), r1ir1.s01);
}

inline void _forward4o(const size_t mg, __global uint2 * restrict const x, const size_t ml, __local const uint2 * restrict const X, const uint2 r2, const uint4 r1ir1)
{
	barrier(CLK_LOCAL_MEM_FENCE);

	const uint2 u0 = X[0 * ml], u2 = X[2 * ml], u1 = X[1 * ml], u3 = X[3 * ml];
	const uint2 v0 = addmod(u0, u2), v2 = submod(u0, u2), v1 = addmod(u1, u3), v3 = mulI(submod(u3, u1));
	x[0 * mg] = addmod(v0, v1); x[1 * mg] = mulmod(submod(v0, v1), r2);
	x[2 * mg] = mulmod(addmod(v2, v3), r1ir1.s23); x[3 * mg] = mulmod(submod(v2, v3), r1ir1.s01);
}

inline void _backward4i(const size_t ml, __local uint2 * restrict const X, const size_t mg, __global const uint2 * restrict const x, const uint2 ir2, const uint4 r1ir1)
{
	const uint2 v0 = x[0 * mg], v1 = mulmod(x[1 * mg], ir2), v2 = mulmod(x[2 * mg], r1ir1.s01), v3 = mulmod(x[3 * mg], r1ir1.s23);
	const uint2 u0 = addmod(v0, v1), u2 = addmod(v2, v3), u1 = submod(v0, v1), u3 = mulI(submod(v2, v3));
	X[0 * ml] = addmod(u0, u2); X[2 * ml] = submod(u0, u2); X[1 * ml] = addmod(u1, u3); X[3 * ml] = submod(u1, u3);
}

inline void _backward4(const size_t m, __local uint2 * restrict const X, const uint2 ir2, const uint4 r1ir1)
{
	barrier(CLK_LOCAL_MEM_FENCE);

	const uint2 v0 = X[0 * m], v1 = mulmod(X[1 * m], ir2), v2 = mulmod(X[2 * m], r1ir1.s01), v3 = mulmod(X[3 * m], r1ir1.s23);
	const uint2 u0 = addmod(v0, v1), u2 = addmod(v2, v3), u1 = submod(v0, v1), u3 = mulI(submod(v2, v3));
	X[0 * m] = addmod(u0, u2); X[2 * m] = submod(u0, u2); X[1 * m] = addmod(u1, u3); X[3 * m] = submod(u1, u3);
}

inline void _backward4o(const size_t mg, __global uint2 * restrict const x, const size_t ml, __local const uint2 * restrict const X, const uint2 ir2, const uint4 r1ir1)
{
	barrier(CLK_LOCAL_MEM_FENCE);

	const uint2 v0 = X[0 * ml], v1 = mulmod(X[1 * ml], ir2), v2 = mulmod(X[2 * ml], r1ir1.s01), v3 = mulmod(X[3 * ml], r1ir1.s23);
	const uint2 u0 = addmod(v0, v1), u2 = addmod(v2, v3), u1 = submod(v0, v1), u3 = mulI(submod(v2, v3));
	x[0 * mg] = addmod(u0, u2); x[2 * mg] = submod(u0, u2); x[1 * mg] = addmod(u1, u3); x[3 * mg] = submod(u1, u3);
}

inline void _square2(__local uint2 * restrict const X)
{
	barrier(CLK_LOCAL_MEM_FENCE);

	// TODO X01 & X45
	const uint2 u0 = X[0], u1 = X[1], u2 = X[2], u3 = X[3];
	const uint2 v0 = addmod(u0, u1), v1 = submod(u0, u1), v2 = addmod(u2, u3), v3 = submod(u2, u3);
	const uint2 s0 = sqrmod(v0), s1 = sqrmod(v1), s2 = sqrmod(v2), s3 = sqrmod(v3);
	X[0] = addmod(s0, s1); X[1] = submod(s0, s1); X[2] = addmod(s2, s3); X[3] = submod(s2, s3);
}

inline void _square4(__local uint2 * restrict const X)
{
	barrier(CLK_LOCAL_MEM_FENCE);

	const uint2 u0 = X[0], u2 = X[2], u1 = X[1], u3 = X[3];
	const uint2 v0 = addmod(u0, u2), v2 = submod(u0, u2), v1 = addmod(u1, u3), v3 = mulI(submod(u3, u1));
	const uint2 s0 = sqrmod(addmod(v0, v1)), s1 = sqrmod(submod(v0, v1)), s2 = sqrmod(addmod(v2, v3)), s3 = sqrmod(submod(v2, v3));
	const uint2 t0 = addmod(s0, s1), t2 = addmod(s2, s3), t1 = submod(s0, s1), t3 = mulI(submod(s2, s3));
	X[0] = addmod(t0, t2); X[2] = submod(t0, t2); X[1] = addmod(t1, t3); X[3] = submod(t1, t3);
}


__kernel  __attribute__((reqd_work_group_size(64 / 4 * CHUNK64, 1, 1)))
void sub_ntt64(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2)
{
	__local uint2 X[64 * CHUNK64];

	const size_t m = get_global_size(0) / 16;

	const size_t local_id = get_local_id(0), chunk_idx = local_id % CHUNK64, threadIdx = local_id / CHUNK64, block_idx = get_group_id(0) * CHUNK64;	// threadIdx < 64 / 4

	const size_t bl_i = block_idx | chunk_idx;

	const size_t _i_16m = threadIdx;											// [+ 16, 32, 48] 0, 1, 2, 3, 4, ..., 15
	const size_t _i_4m = ((4 * threadIdx) & ~(4 * 4 - 1)) | (threadIdx % 4);	// [+ 4, 8, 12] 0, 1, 2, 3, 16, 17, ...
	const size_t _i_m = 4 * threadIdx;											// [+ 1, 2, 3] 0, 4, 8, 12, ...

	const size_t i_16m = _i_16m * CHUNK64 | chunk_idx;
	const size_t i_4m = _i_4m * CHUNK64 | chunk_idx;
	const size_t i_m = _i_m * CHUNK64 | chunk_idx;

	const size_t k_16m = _i_16m * m | bl_i;
	const size_t k_4m = _i_4m * m | bl_i;
	const size_t k_m = _i_m * m | bl_i;

	const size_t j_16m = k_16m;	// & (16 * m - 1); We have _i_16m < 16 and bl_i < m
	const size_t j_4m = k_4m & (4 * m - 1);
	const size_t j_m = bl_i;	// k_m & (m - 1); We have bl_i < m

	_sub_forward4i(16 * CHUNK64, &X[i_16m], 16 * m, &x[k_16m], r2[j_16m], r1ir1[j_16m]);
	_forward4(4 * CHUNK64, &X[i_4m], r2[16 * m + j_4m], r1ir1[16 * m + j_4m]);
	_forward4o(m, &x[k_m], CHUNK64, &X[i_m], r2[16 * m + 4 * m + j_m], r1ir1[16 * m + 4 * m + j_m]);
}

__kernel  __attribute__((reqd_work_group_size(64 / 4 * CHUNK64, 1, 1)))
void ntt64(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)
{
	__local uint2 X[64 * CHUNK64];

	const size_t local_id = get_local_id(0), chunk_idx = local_id % CHUNK64, threadIdx = local_id / CHUNK64, block_idx = get_group_id(0) * CHUNK64;	// threadIdx < 64 / 4

	__global uint2 * restrict const xo = &x[64 * (block_idx & ~(m - 1))];		// m-block offset
	const size_t bl_i = (block_idx & (m - 1)) | chunk_idx;

	const size_t _i_16m = threadIdx;											// [+ 16, 32, 48] 0, 1, 2, 3, 4, ..., 15
	const size_t _i_4m = ((4 * threadIdx) & ~(4 * 4 - 1)) | (threadIdx % 4);	// [+ 4, 8, 12] 0, 1, 2, 3, 16, 17, ...
	const size_t _i_m = 4 * threadIdx;											// [+ 1, 2, 3] 0, 4, 8, 12, ...

	const size_t i_16m = _i_16m * CHUNK64 | chunk_idx;
	const size_t i_4m = _i_4m * CHUNK64 | chunk_idx;
	const size_t i_m = _i_m * CHUNK64 | chunk_idx;

	const size_t k_16m = _i_16m * m | bl_i;
	const size_t k_4m = _i_4m * m | bl_i;
	const size_t k_m = _i_m * m | bl_i;

	const size_t j_16m = k_16m;	// & (16 * m - 1); We have _i_16m < 16 and bl_i < m
	const size_t j_4m = k_4m & (4 * m - 1);
	const size_t j_m = bl_i;	// k_m & (m - 1); We have bl_i < m

	_forward4i(16 * CHUNK64, &X[i_16m], 16 * m, &xo[k_16m], r2[rindex + j_16m], r1ir1[rindex + j_16m]);
	_forward4(4 * CHUNK64, &X[i_4m], r2[rindex + 16 * m + j_4m], r1ir1[rindex + 16 * m + j_4m]);
	_forward4o(m, &xo[k_m], CHUNK64, &X[i_m], r2[rindex + 16 * m + 4 * m + j_m], r1ir1[rindex + 16 * m + 4 * m + j_m]);
}

__kernel  __attribute__((reqd_work_group_size(64 / 4 * CHUNK64, 1, 1)))
void intt64(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)
{
	__local uint2 X[64 * CHUNK64];

	const size_t local_id = get_local_id(0), chunk_idx = local_id % CHUNK64, threadIdx = local_id / CHUNK64, block_idx = get_group_id(0) * CHUNK64;	// threadIdx < 64 / 4

	__global uint2 * restrict const xo = &x[64 * (block_idx & ~(m - 1))];		// m-block offset
	const size_t bl_i = (block_idx & (m - 1)) | chunk_idx;

	const size_t _i_16m = threadIdx;											// [+ 16, 32, 48] 0, 1, 2, 3, 4, ..., 15
	const size_t _i_4m = ((4 * threadIdx) & ~(4 * 4 - 1)) | (threadIdx % 4);	// [+ 4, 8, 12] 0, 1, 2, 3, 16, 17, ...
	const size_t _i_m = 4 * threadIdx;											// [+ 1, 2, 3] 0, 4, 8, 12, ...

	const size_t i_16m = _i_16m * CHUNK64 | chunk_idx;
	const size_t i_4m = _i_4m * CHUNK64 | chunk_idx;
	const size_t i_m = _i_m * CHUNK64 | chunk_idx;

	const size_t k_16m = _i_16m * m | bl_i;
	const size_t k_4m = _i_4m * m | bl_i;
	const size_t k_m = _i_m * m | bl_i;

	const size_t j_16m = k_16m;	// & (16 * m - 1); We have _i_16m < 16 and bl_i < m
	const size_t j_4m = k_4m & (4 * m - 1);
	const size_t j_m = bl_i;	// k_m & (m - 1); We have bl_i < m

	_backward4i(CHUNK64, &X[i_m], m, &xo[k_m], ir2[rindex + 16 * m + 4 * m + j_m], r1ir1[rindex + 16 * m + 4 * m + j_m]);
	_backward4(4 * CHUNK64, &X[i_4m], ir2[rindex + 16 * m + j_4m], r1ir1[rindex + 16 * m + j_4m]);
	_backward4o(16 * m, &xo[k_16m], 16 * CHUNK64, &X[i_16m], ir2[rindex + j_16m], r1ir1[rindex + j_16m]);
}

__kernel __attribute__((reqd_work_group_size(32 / 4 * BLK32, 1, 1)))
void square32(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)
{
	__local uint2 X[32 * BLK32];

	// copy mem first ?

	const size_t i = get_local_id(0);
	const size_t i_8 = i % 8, i8 = ((4 * i) & (size_t)~(4 * 8 - 1)) | i_8, j8 = i_8 + 2;
	const size_t i_2 = i % 2, i2 = ((4 * i) & (size_t)~(4 * 2 - 1)) | i_2, j2 = i_2;
	const size_t k8 = get_group_id(0) * 32 * BLK32 | i8;

	_forward4i(8, &X[i8], 8, &x[k8], r2ir2[j8].s01, r1ir1[j8]);
	_forward4(2, &X[i2], r2ir2[j2].s01, r1ir1[j2]);
	_square2(&X[4 * i]);
	_backward4(2, &X[i2], r2ir2[j2].s23, r1ir1[j2]);
	_backward4o(8, &x[k8], 8, &X[i8], r2ir2[j8].s23, r1ir1[j8]);
}

__kernel __attribute__((reqd_work_group_size(64 / 4 * BLK64, 1, 1)))
void square64(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)
{
	__local uint2 X[64 * BLK64];

	const size_t i = get_local_id(0);
	const size_t i_16 = i % 16, i16 = ((4 * i) & (size_t)~(4 * 16 - 1)) | i_16, j16 = i_16 + 4;
	const size_t i_4 = i % 4, i4 = ((4 * i) & (size_t)~(4 * 4 - 1)) | i_4, j4 = i_4;
	const size_t k16 = get_group_id(0) * 64 * BLK64 | i16;

	_forward4i(16, &X[i16], 16, &x[k16], r2ir2[j16].s01, r1ir1[j16]);
	_forward4(4, &X[i4], r2ir2[j4].s01, r1ir1[j4]);
	_square4(&X[4 * i]);
	_backward4(4, &X[i4], r2ir2[j4].s23, r1ir1[j4]);
	_backward4o(16, &x[k16], 16, &X[i16], r2ir2[j16].s23, r1ir1[j16]);
}

__kernel __attribute__((reqd_work_group_size(128 / 4 * BLK128, 1, 1)))
void square128(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)
{
	__local uint2 X[128 * BLK128];

	const size_t i = get_local_id(0);
	const size_t i_32 = i % 32, i32 = ((4 * i) & (size_t)~(4 * 32 - 1)) | i_32, j32 = i_32 + 2 + 8;
	const size_t i_8 = i % 8, i8 = ((4 * i) & (size_t)~(4 * 8 - 1)) | i_8, j8 = i_8 + 2;
	const size_t i_2 = i % 2, i2 = ((4 * i) & (size_t)~(4 * 2 - 1)) | i_2, j2 = i_2;
	const size_t k32 = get_group_id(0) * 128 * BLK128 | i32;

	_forward4i(32, &X[i32], 32, &x[k32], r2ir2[j32].s01, r1ir1[j32]);
	_forward4(8, &X[i8], r2ir2[j8].s01, r1ir1[j8]);
	_forward4(2, &X[i2], r2ir2[j2].s01, r1ir1[j2]);
	_square2(&X[4 * i]);
	_backward4(2, &X[i2], r2ir2[j2].s23, r1ir1[j2]);
	_backward4(8, &X[i8], r2ir2[j8].s23, r1ir1[j8]);
	_backward4o(32, &x[k32], 32, &X[i32], r2ir2[j32].s23, r1ir1[j32]);
}

__kernel __attribute__((reqd_work_group_size(256 / 4 * BLK256, 1, 1)))
void square256(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)
{
	__local uint2 X[256 * BLK256];

	const size_t i = get_local_id(0);
	const size_t i_64 = i % 64, i64 = ((4 * i) & (size_t)~(4 * 64 - 1)) | i_64, j64 = i_64 + 4 + 16;
	const size_t i_16 = i % 16, i16 = ((4 * i) & (size_t)~(4 * 16 - 1)) | i_16, j16 = i_16 + 4;
	const size_t i_4 = i % 4, i4 = ((4 * i) & (size_t)~(4 * 4 - 1)) | i_4, j4 = i_4;
	const size_t k64 = get_group_id(0) * 256 * BLK256 | i64;

	_forward4i(64, &X[i64], 64, &x[k64], r2ir2[j64].s01, r1ir1[j64]);
	_forward4(16, &X[i16], r2ir2[j16].s01, r1ir1[j16]);
	_forward4(4, &X[i4], r2ir2[j4].s01, r1ir1[j4]);
	_square4(&X[4 * i]);
	_backward4(4, &X[i4], r2ir2[j4].s23, r1ir1[j4]);
	_backward4(16, &X[i16], r2ir2[j16].s23, r1ir1[j16]);
	_backward4o(64, &x[k64], 64, &X[i64], r2ir2[j64].s23, r1ir1[j64]);
}

__kernel __attribute__((reqd_work_group_size(512 / 4, 1, 1)))
void square512(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)
{
	__local uint2 X[512];

	const size_t i = get_local_id(0);
	const size_t i128 = i, j128 = i + 2 + 8 + 32;
	const size_t i_32 = i % 32, i32 = ((4 * i) & (size_t)~(4 * 32 - 1)) | i_32, j32 = i_32 + 2 + 8;
	const size_t i_8 = i % 8, i8 = ((4 * i) & (size_t)~(4 * 8 - 1)) | i_8, j8 = i_8 + 2;
	const size_t i_2 = i % 2, i2 = ((4 * i) & (size_t)~(4 * 2 - 1)) | i_2, j2 = i_2;
	const size_t k128 = get_group_id(0) * 512 | i128;

	_forward4i(128, &X[i128], 128, &x[k128], r2ir2[j128].s01, r1ir1[j128]);
	_forward4(32, &X[i32], r2ir2[j32].s01, r1ir1[j32]);
	_forward4(8, &X[i8], r2ir2[j8].s01, r1ir1[j8]);
	_forward4(2, &X[i2], r2ir2[j2].s01, r1ir1[j2]);
	_square2(&X[4 * i]);
	_backward4(2, &X[i2], r2ir2[j2].s23, r1ir1[j2]);
	_backward4(8, &X[i8], r2ir2[j8].s23, r1ir1[j8]);
	_backward4(32, &X[i32], r2ir2[j32].s23, r1ir1[j32]);
	_backward4o(128, &x[k128], 128, &X[i128], r2ir2[j128].s23, r1ir1[j128]);
}

__kernel __attribute__((reqd_work_group_size(1024 / 4, 1, 1)))
void square1024(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)
{
	__local uint2 X[1024];

	const size_t i = get_local_id(0);
	const size_t i256 = i, j256 = i + 4 + 16 + 64;
	const size_t i_64 = i % 64, i64 = ((4 * i) & (size_t)~(4 * 64 - 1)) | i_64, j64 = i_64 + 4 + 16;
	const size_t i_16 = i % 16, i16 = ((4 * i) & (size_t)~(4 * 16 - 1)) | i_16, j16 = i_16 + 4;
	const size_t i_4 = i % 4, i4 = ((4 * i) & (size_t)~(4 * 4 - 1)) | i_4, j4 = i_4;
	const size_t k256 = get_group_id(0) * 1024 | i256;

	_forward4i(256, &X[i256], 256, &x[k256], r2ir2[j256].s01, r1ir1[j256]);
	_forward4(64, &X[i64], r2ir2[j64].s01, r1ir1[j64]);
	_forward4(16, &X[i16], r2ir2[j16].s01, r1ir1[j16]);
	_forward4(4, &X[i4], r2ir2[j4].s01, r1ir1[j4]);
	_square4(&X[4 * i]);
	_backward4(4, &X[i4], r2ir2[j4].s23, r1ir1[j4]);
	_backward4(16, &X[i16], r2ir2[j16].s23, r1ir1[j16]);
	_backward4(64, &X[i64], r2ir2[j64].s23, r1ir1[j64]);
	_backward4o(256, &x[k256], 256, &X[i256], r2ir2[j256].s23, r1ir1[j256]);
}

__kernel __attribute__((reqd_work_group_size(P2I_WGS, 1, 1)))
void poly2int0(__global uint2 * restrict const x, __global long * restrict const cr, const uint2 norm)
{
	__local long L[P2I_WGS * P2I_BLK];
	__local uint X[P2I_WGS * P2I_BLK];

	const size_t i = get_local_id(0), blk = get_group_id(0);
	const size_t kc = (get_global_id(0) + 1) & (get_global_size(0) - 1);

	__global uint2 * restrict const xo = &x[P2I_WGS * P2I_BLK * blk];

	for (size_t j = 0; j < P2I_BLK; ++j)
	{
		const size_t k = P2I_WGS * j + i;
		L[P2I_WGS * (k % P2I_BLK) + (k / P2I_BLK)] = getlong(mulmod(xo[k], norm));	// -n/2 . (B-1)^2 <= l <= n/2 . (B-1)^2
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	long l = 0;
#pragma unroll
	for (size_t j = 0; j < P2I_BLK; ++j)
	{
		l += L[P2I_WGS * j + i];
		X[P2I_WGS * j + i] = (uint)(l) & digit_mask;
	 	l >>= digit_bit;
	}
	cr[kc] = l;

	barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
	for (size_t j = 0; j < P2I_BLK; ++j)
	{
		const size_t k = P2I_WGS * j + i;
		xo[k].s0 = (uint)X[P2I_WGS * (k % P2I_BLK) + (k / P2I_BLK)];
	}
}

__kernel
void poly2int1(__global uint2 * restrict const x, __global const long * restrict const cr, __global int * const err)
{
	const size_t k = get_global_id(0);

	__global uint2 * restrict const xi = &x[P2I_BLK * k];

	long l = cr[k] + xi[0].s0;
	xi[0].s0 = (uint)(l) & digit_mask;
	l >>= digit_bit;						// |l| < n/2

	int f = (int)(l);
	for (size_t j = 1; j < P2I_BLK; ++j)
	{
		f += xi[j].s0;
		xi[j].s0 = (uint)(f) & digit_mask;
		f >>= digit_bit;					// f = -1, 0 or 1
		if (f == 0) break;
	}

	if (f != 0) atomic_or(err, 1);
}

__kernel
void reduce_i(__global uint2 * restrict const x, __global uint * restrict const t, __global const uint4 * restrict const bp,
	const uint e, const int s, const uint d, const uint d_inv, const int d_shift)
{
	const size_t k = get_global_id(0);

	__global uint2 * restrict const xi = &x[4 * k];

	const uint xe0 = xi[e + 0].s0, xe1 = xi[e + 1].s0, xe2 = xi[e + 2].s0, xe3 = xi[e + 3].s0, xe4 = xi[e + 4].s0; 

	const uint x0 = ((xe1 << (digit_bit - s)) | (xe0 >> s)) & digit_mask;
	const uint x1 = ((xe2 << (digit_bit - s)) | (xe1 >> s)) & digit_mask;
	const uint x2 = ((xe3 << (digit_bit - s)) | (xe2 >> s)) & digit_mask;
	const uint x3 = ((xe4 << (digit_bit - s)) | (xe3 >> s)) & digit_mask;

	const uint4 b = bp[k];

	xi[0].s1 = x0; xi[1].s1 = x1; xi[2].s1 = x2; xi[3].s1 = x3;

	const uint u0 = _rem(x0 * (ulong)(b.s0), d, d_inv, d_shift), u1 = _rem(x1 * (ulong)(b.s1), d, d_inv, d_shift);
	const uint u2 = _rem(x2 * (ulong)(b.s2), d, d_inv, d_shift), u3 = _rem(x3 * (ulong)(b.s3), d, d_inv, d_shift);
	t[4 * k + 0 + 1] = u0; t[4 * k + 1 + 1] = u1; t[4 * k + 2 + 1] = u2; t[4 * k + 3 + 1] = u3;
}

__kernel
void reduce_upsweep4(__global uint * restrict const t, const uint d, const uint m)
{
	const size_t k = get_global_id(0);
	__global uint * restrict const tk = &t[k * 4 * m + 1];	// TODO shift s
	const uint u0 = tk[0 * m], u1 = tk[1 * m], u2 = tk[2 * m], u3 = tk[3 * m];
	const uint v0 = _addmod(u0, u1, d), v2 = _addmod(u2, u3, d);
	tk[0 * m] = _addmod(v0, v2, d);
	tk[2 * m] = v2;
}

__kernel
void reduce_downsweep4(__global uint * restrict const t, const uint d, const uint m)
{
	const size_t k = get_global_id(0);
	__global uint * restrict const tk = &t[k * 4 * m + 1];	// TODO shift s
	const uint u0 = tk[0 * m], u1 = tk[1 * m], u2 = tk[2 * m], u3 = tk[3 * m];
	const uint v0 = _addmod(u0, u2, d);
	tk[0 * m] = _addmod(v0, u1, d);
	tk[1 * m] = v0;
	tk[2 * m] = _addmod(u0, u3, d);
	tk[3 * m] = u0;
}

__kernel
void reduce_topsweep2(__global uint * restrict const t, const uint d, const uint n_2)
{
	const uint u0 = t[1], u1 = t[1 + n_2];
	t[0] = _addmod(u0, u1, d);
	t[1] = u1;
	t[1 + n_2] = 0;
}

__kernel
void reduce_topsweep4(__global uint * restrict const t, const uint d, const uint n_4)
{
	const uint u0 = t[1 + 0 * n_4], u1 = t[1 + 1 * n_4], u2 = t[1 + 2 * n_4], u3 = t[1 + 3 * n_4];
	const uint v0 = _addmod(u0, u1, d), v2 = _addmod(u2, u3, d);
	t[0] = _addmod(v0, v2, d);
	t[1 + 0 * n_4] = _addmod(u1, v2, d);
	t[1 + 1 * n_4] = v2;
	t[1 + 2 * n_4] = u3;
	t[1 + 3 * n_4] = 0;
}

__kernel
void reduce_o(__global uint2 * restrict const x, __global const uint * restrict const t, __global const uint * restrict const ibp,
	const uint e, const int s, const uint d, const uint d_inv, const int d_shift)
{
	const size_t n = get_global_size(0), k = get_global_id(0);

	const uint rbk_prev = (k + 1 < n) ? t[k + 1] : 0;
	const uint r_prev = _rem(rbk_prev * (ulong)(ibp[k]), d, d_inv, d_shift);

	const uint2 x_k = x[k];

	const ulong q = ((ulong)(r_prev) << digit_bit) | x_k.s1;

	const uint q_d = mul_hi((uint)(q >> d_shift), d_inv);	// d < 2^29
	const uint r = (uint)(q) - q_d * d;
	const uint c = (r >= d) ? 1 : 0;

	x[k] = (uint2)((k > e + 2) ? 0 : x_k.s0, q_d + c);

	if (k + 1 == n) { x[n].s0 = t[0]; }
}

__kernel
void reduce_f(__global uint2 * restrict const x, const uint n, const uint e, const int s)
{
	const uint rs = x[e].s0 & ((1u << s) - 1);
	const ulong rds = ((ulong)(x[n].s0) << s) | rs;		// rds < 2^(29 + digit_bit - 1)

	x[e].s0 = (uint)(rds) & digit_mask;
	if (e + 1 < n) x[e + 1].s0 = (uint)(rds >> digit_bit) & digit_mask;
	if (e + 2 < n) x[e + 2].s0 = (uint)(rds >> (2 * digit_bit)) & digit_mask;
}
