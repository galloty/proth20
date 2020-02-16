/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

__kernel
void ntt4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)
{
	const size_t k = get_global_id(0);

	const size_t i = k & (m - 1), j = 4 * k - 3 * i;

	const uint2 r2_i = r2[rindex + i];
	const uint4 r1ir1_i = r1ir1[rindex + i];

	const uint2 u0 = x[j + 0 * m], u2 = x[j + 2 * m], u1 = x[j + 1 * m], u3 = x[j + 3 * m];
	const uint2 v0 = addmod(u0, u2), v2 = submod(u0, u2), v1 = addmod(u1, u3), v3 = mulI(submod(u3, u1));
	x[j + 0 * m] = addmod(v0, v1); x[j + 1 * m] = mulmod(submod(v0, v1), r2_i);
	x[j + 2 * m] = mulmod(addmod(v2, v3), r1ir1_i.s23); x[j + 3 * m] = mulmod(submod(v2, v3), r1ir1_i.s01);
}

__kernel
void intt4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)
{
	const size_t k = get_global_id(0);

	const size_t i = k & (m - 1), j = 4 * k - 3 * i;

	const uint2 ir2_i = ir2[rindex + i];
	const uint4 r1ir1_i = r1ir1[rindex + i];

	const uint2 v0 = x[j + 0 * m], v1 = mulmod(x[j + 1 * m], ir2_i), v2 = mulmod(x[j + 2 * m], r1ir1_i.s01), v3 = mulmod(x[j + 3 * m], r1ir1_i.s23);
	const uint2 u0 = addmod(v0, v1), u2 = addmod(v2, v3), u1 = submod(v0, v1), u3 = mulI(submod(v2, v3));
	x[j + 0 * m] = addmod(u0, u2); x[j + 2 * m] = submod(u0, u2);
	x[j + 1 * m] = addmod(u1, u3); x[j + 3 * m] = submod(u1, u3);
}


#define FORWARD4(M, CHUNK, R) \
{ \
	const size_t t = threadIdx % M; \
	const size_t i = (threadIdx & ~(M - 1)) * 4 | t; \
	const size_t j = t * m | bl_i; \
	_forward4(M * CHUNK, &X[i * CHUNK | chunk_idx], r2[R + j], r1ir1[R + j]); \
}

#define BACKWARD4(M, CHUNK, R) \
{ \
	const size_t t = threadIdx % M; \
	const size_t i = (threadIdx & ~(M - 1)) * 4 | t; \
	const size_t j = t * m | bl_i; \
	_backward4(M * CHUNK, &X[i * CHUNK | chunk_idx], ir2[R + j], r1ir1[R + j]); \
}

#define SUB_FORWARD4i(M, CHUNK) \
{ \
	const size_t j = threadIdx * m | bl_i; \
	_sub_forward4i(M * CHUNK, &X[threadIdx * CHUNK | chunk_idx], M * m, &xo[j], r2[j], r1ir1[j]); \
}

#define FORWARD4i(M, CHUNK, R) \
{ \
	const size_t j = threadIdx * m | bl_i; \
	_forward4i(M * CHUNK, &X[threadIdx * CHUNK | chunk_idx], M * m, &xo[j], r2[R + j], r1ir1[R + j]); \
}

#define FORWARD4o(CHUNK, R) \
{ \
	const size_t i = threadIdx * 4; \
	_forward4o(m, &xo[i * m | bl_i], CHUNK, &X[i * CHUNK | chunk_idx], r2[R + bl_i], r1ir1[R + bl_i]); \
}

#define BACKWARD4i(CHUNK, R) \
{ \
	const size_t i = threadIdx * 4; \
	_backward4i(CHUNK, &X[i * CHUNK | chunk_idx], m, &xo[i * m | bl_i], ir2[R + bl_i], r1ir1[R + bl_i]); \
}

#define BACKWARD4o(M, CHUNK, R) \
{ \
	const size_t j = threadIdx * m | bl_i; \
	_backward4o(M * m, &xo[j], M * CHUNK, &X[threadIdx * CHUNK | chunk_idx], ir2[R + j], r1ir1[R + j]); \
}


#define SETVAR(M, CHUNK) \
	__local uint2 X[M * CHUNK]; \
	const size_t local_id = get_local_id(0), chunk_idx = local_id % CHUNK, threadIdx = local_id / CHUNK, block_idx = get_group_id(0) * CHUNK;

#define SETVAR_SUB_NTT(M) \
	__global uint2 * const xo = x; \
	const size_t bl_i = block_idx | chunk_idx; \
	const size_t m = (pconst_size / 4) / (M / 4);

#define SETVAR_NTT(M) \
	__global uint2 * const xo = &x[M * (block_idx & ~(m - 1))]; \
	const size_t bl_i = (block_idx & (m - 1)) | chunk_idx;


#define SUB_NTT64(CHUNK) \
	SETVAR(64, CHUNK); \
	SETVAR_SUB_NTT(64); \
	SUB_FORWARD4i(16, CHUNK); \
	FORWARD4(4, CHUNK, 16 * m); \
	FORWARD4o(CHUNK, 16 * m + 4 * m);

#define NTT64(CHUNK) \
	SETVAR(64, CHUNK); \
	SETVAR_NTT(64); \
	FORWARD4i(16, CHUNK, rindex); \
	FORWARD4(4, CHUNK, rindex + 16 * m); \
	FORWARD4o(CHUNK, rindex + 16 * m + 4 * m);

#define INTT64(CHUNK) \
	SETVAR(64, CHUNK); \
	SETVAR_NTT(64); \
	BACKWARD4i(CHUNK, rindex + 16 * m + 4 * m); \
	BACKWARD4(4, CHUNK, rindex + 16 * m); \
	BACKWARD4o(16, CHUNK, rindex);

#define SUB_NTT256(CHUNK) \
	SETVAR(256, CHUNK); \
	SETVAR_SUB_NTT(256); \
	SUB_FORWARD4i(64, CHUNK); \
	FORWARD4(16, CHUNK, 64 * m); \
	FORWARD4(4, CHUNK, 64 * m + 16 * m); \
	FORWARD4o(CHUNK, 64 * m + 16 * m + 4 * m);

#define NTT256(CHUNK) \
	SETVAR(256, CHUNK); \
	SETVAR_NTT(256); \
	FORWARD4i(64, CHUNK, rindex); \
	FORWARD4(16, CHUNK, rindex + 64 * m); \
	FORWARD4(4, CHUNK, rindex + 64 * m + 16 * m); \
	FORWARD4o(CHUNK, rindex + 64 * m + 16 * m + 4 * m);

#define INTT256(CHUNK) \
	SETVAR(256, CHUNK); \
	SETVAR_NTT(256); \
	BACKWARD4i(CHUNK, rindex + 64 * m + 16 * m + 4 * m); \
	BACKWARD4(4, CHUNK, rindex + 64 * m + 16 * m); \
	BACKWARD4(16, CHUNK, rindex + 64 * m); \
	BACKWARD4o(64, CHUNK, rindex);

#define SUB_NTT1024(CHUNK) \
	SETVAR(1024, CHUNK); \
	SETVAR_SUB_NTT(1024); \
	SUB_FORWARD4i(256, CHUNK); \
	FORWARD4(64, CHUNK, 256 * m); \
	FORWARD4(16, CHUNK, 256 * m + 64 * m); \
	FORWARD4(4, CHUNK, 256 * m + 64 * m + 16 * m); \
	FORWARD4o(CHUNK, 256 * m + 64 * m + 16 * m + 4 * m);

#define NTT1024(CHUNK) \
	SETVAR(1024, CHUNK); \
	SETVAR_NTT(1024); \
	FORWARD4i(256, CHUNK, rindex); \
	FORWARD4(64, CHUNK, rindex + 256 * m); \
	FORWARD4(16, CHUNK, rindex + 256 * m + 64 * m); \
	FORWARD4(4, CHUNK, rindex + 256 * m + 64 * m + 16 * m); \
	FORWARD4o(CHUNK, rindex + 256 * m + 64 * m + 16 * m + 4 * m);

#define INTT1024(CHUNK) \
	SETVAR(1024, CHUNK); \
	SETVAR_NTT(1024); \
	BACKWARD4i(CHUNK, rindex + 256 * m + 64 * m + 16 * m + 4 * m); \
	BACKWARD4(4, CHUNK, rindex + 256 * m + 64 * m + 16 * m); \
	BACKWARD4(16, CHUNK, rindex + 256 * m + 64 * m); \
	BACKWARD4(64, CHUNK, rindex + 256 * m); \
	BACKWARD4o(256, CHUNK, rindex);


__kernel __attribute__((reqd_work_group_size(64 / 4 * 16, 1, 1)))
void sub_ntt64_16(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2)
{
	SUB_NTT64(16);
}

__kernel __attribute__((reqd_work_group_size(64 / 4 * 16, 1, 1)))
void ntt64_16(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)
{
	NTT64(16);
}

__kernel __attribute__((reqd_work_group_size(64 / 4 * 16, 1, 1)))
void intt64_16(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)
{
	INTT64(16);
}


__kernel __attribute__((reqd_work_group_size(256 / 4 * 4, 1, 1)))
void sub_ntt256_4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2)
{
	SUB_NTT256(4);
}

__kernel __attribute__((reqd_work_group_size(256 / 4 * 4, 1, 1)))
void ntt256_4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)
{
	NTT256(4);
}

__kernel __attribute__((reqd_work_group_size(256 / 4 * 4, 1, 1)))
void intt256_4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)
{
	INTT256(4);
}


__kernel __attribute__((reqd_work_group_size(1024 / 4 * 1, 1, 1)))
void sub_ntt1024_1(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2)
{
	SUB_NTT1024(1);
}

__kernel __attribute__((reqd_work_group_size(1024 / 4 * 1, 1, 1)))
void ntt1024_1(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)
{
	NTT1024(1);
}

__kernel __attribute__((reqd_work_group_size(1024 / 4 * 1, 1, 1)))
void intt1024_1(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)
{
	INTT1024(1);
}
