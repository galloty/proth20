/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

static const char * const src_ocl_NTT = \
"/*\n" \
"Copyright 2020, Yves Gallot\n" \
"\n" \
"proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\n" \
"Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n" \
"*/\n" \
"\n" \
"__kernel\n" \
"void ntt4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)\n" \
"{\n" \
"	const size_t k = get_global_id(0);\n" \
"\n" \
"	const size_t i = k & (m - 1), j = 4 * k - 3 * i;\n" \
"\n" \
"	const uint2 r2_i = r2[rindex + i];\n" \
"	const uint4 r1ir1_i = r1ir1[rindex + i];\n" \
"\n" \
"	const uint2 u0 = x[j + 0 * m], u2 = x[j + 2 * m], u1 = x[j + 1 * m], u3 = x[j + 3 * m];\n" \
"	const uint2 v0 = addmod(u0, u2), v2 = submod(u0, u2), v1 = addmod(u1, u3), v3 = mulI(submod(u3, u1));\n" \
"	x[j + 0 * m] = addmod(v0, v1); x[j + 1 * m] = mulmod(submod(v0, v1), r2_i);\n" \
"	x[j + 2 * m] = mulmod(addmod(v2, v3), r1ir1_i.s23); x[j + 3 * m] = mulmod(submod(v2, v3), r1ir1_i.s01);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void intt4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)\n" \
"{\n" \
"	const size_t k = get_global_id(0);\n" \
"\n" \
"	const size_t i = k & (m - 1), j = 4 * k - 3 * i;\n" \
"\n" \
"	const uint2 ir2_i = ir2[rindex + i];\n" \
"	const uint4 r1ir1_i = r1ir1[rindex + i];\n" \
"\n" \
"	const uint2 v0 = x[j + 0 * m], v1 = mulmod(x[j + 1 * m], ir2_i), v2 = mulmod(x[j + 2 * m], r1ir1_i.s01), v3 = mulmod(x[j + 3 * m], r1ir1_i.s23);\n" \
"	const uint2 u0 = addmod(v0, v1), u2 = addmod(v2, v3), u1 = submod(v0, v1), u3 = mulI(submod(v2, v3));\n" \
"	x[j + 0 * m] = addmod(u0, u2); x[j + 2 * m] = submod(u0, u2);\n" \
"	x[j + 1 * m] = addmod(u1, u3); x[j + 3 * m] = submod(u1, u3);\n" \
"}\n" \
"\n" \
"\n" \
"#define FORWARD4(M, CHUNK, R) \\\n" \
"{ \\\n" \
"	const size_t t = threadIdx % M; \\\n" \
"	const size_t i = (threadIdx & ~(M - 1)) * 4 | t; \\\n" \
"	const size_t j = t * m | bl_i; \\\n" \
"	_forward4(M * CHUNK, &X[i * CHUNK | chunk_idx], r2[R + j], r1ir1[R + j]); \\\n" \
"}\n" \
"\n" \
"#define BACKWARD4(M, CHUNK, R) \\\n" \
"{ \\\n" \
"	const size_t t = threadIdx % M; \\\n" \
"	const size_t i = (threadIdx & ~(M - 1)) * 4 | t; \\\n" \
"	const size_t j = t * m | bl_i; \\\n" \
"	_backward4(M * CHUNK, &X[i * CHUNK | chunk_idx], ir2[R + j], r1ir1[R + j]); \\\n" \
"}\n" \
"\n" \
"#define SUB_FORWARD4i(M, CHUNK) \\\n" \
"{ \\\n" \
"	const size_t j = threadIdx * m | bl_i; \\\n" \
"	_sub_forward4i(M * CHUNK, &X[threadIdx * CHUNK | chunk_idx], M * m, &xo[j], r2[j], r1ir1[j]); \\\n" \
"}\n" \
"\n" \
"#define FORWARD4i(M, CHUNK, R) \\\n" \
"{ \\\n" \
"	const size_t j = threadIdx * m | bl_i; \\\n" \
"	_forward4i(M * CHUNK, &X[threadIdx * CHUNK | chunk_idx], M * m, &xo[j], r2[R + j], r1ir1[R + j]); \\\n" \
"}\n" \
"\n" \
"#define FORWARD4o(CHUNK, R) \\\n" \
"{ \\\n" \
"	const size_t i = threadIdx * 4; \\\n" \
"	_forward4o(m, &xo[i * m | bl_i], CHUNK, &X[i * CHUNK | chunk_idx], r2[R + bl_i], r1ir1[R + bl_i]); \\\n" \
"}\n" \
"\n" \
"#define BACKWARD4i(CHUNK, R) \\\n" \
"{ \\\n" \
"	const size_t i = threadIdx * 4; \\\n" \
"	_backward4i(CHUNK, &X[i * CHUNK | chunk_idx], m, &xo[i * m | bl_i], ir2[R + bl_i], r1ir1[R + bl_i]); \\\n" \
"}\n" \
"\n" \
"#define BACKWARD4o(M, CHUNK, R) \\\n" \
"{ \\\n" \
"	const size_t j = threadIdx * m | bl_i; \\\n" \
"	_backward4o(M * m, &xo[j], M * CHUNK, &X[threadIdx * CHUNK | chunk_idx], ir2[R + j], r1ir1[R + j]); \\\n" \
"}\n" \
"\n" \
"\n" \
"#define SETVAR(M, CHUNK) \\\n" \
"	__local uint2 X[M * CHUNK]; \\\n" \
"	const size_t local_id = get_local_id(0), chunk_idx = local_id % CHUNK, threadIdx = local_id / CHUNK, block_idx = get_group_id(0) * CHUNK;\n" \
"\n" \
"#define SETVAR_FL_NTT(M) \\\n" \
"	__global uint2 * const xo = x; \\\n" \
"	const size_t bl_i = block_idx | chunk_idx; \\\n" \
"	const size_t m = (pconst_size / 4) / (M / 4);\n" \
"\n" \
"#define SETVAR_NTT(M) \\\n" \
"	__global uint2 * const xo = &x[M * (block_idx & ~(m - 1))]; \\\n" \
"	const size_t bl_i = (block_idx & (m - 1)) | chunk_idx;\n" \
"\n" \
"\n" \
"#define SUB_NTT64(CHUNK) \\\n" \
"	SETVAR(64, CHUNK); \\\n" \
"	SETVAR_FL_NTT(64); \\\n" \
"	SUB_FORWARD4i(16, CHUNK); \\\n" \
"	FORWARD4(4, CHUNK, 16 * m); \\\n" \
"	FORWARD4o(CHUNK, 16 * m + 4 * m);\n" \
"\n" \
"#define LST_INTT64(CHUNK) \\\n" \
"	SETVAR(64, CHUNK); \\\n" \
"	SETVAR_FL_NTT(64); \\\n" \
"	BACKWARD4i(CHUNK, 16 * m + 4 * m); \\\n" \
"	BACKWARD4(4, CHUNK, 16 * m); \\\n" \
"	BACKWARD4o(16, CHUNK, 0);\n" \
"\n" \
"#define NTT64(CHUNK) \\\n" \
"	SETVAR(64, CHUNK); \\\n" \
"	SETVAR_NTT(64); \\\n" \
"	FORWARD4i(16, CHUNK, rindex); \\\n" \
"	FORWARD4(4, CHUNK, rindex + 16 * m); \\\n" \
"	FORWARD4o(CHUNK, rindex + 16 * m + 4 * m);\n" \
"\n" \
"#define INTT64(CHUNK) \\\n" \
"	SETVAR(64, CHUNK); \\\n" \
"	SETVAR_NTT(64); \\\n" \
"	BACKWARD4i(CHUNK, rindex + 16 * m + 4 * m); \\\n" \
"	BACKWARD4(4, CHUNK, rindex + 16 * m); \\\n" \
"	BACKWARD4o(16, CHUNK, rindex);\n" \
"\n" \
"#define SUB_NTT256(CHUNK) \\\n" \
"	SETVAR(256, CHUNK); \\\n" \
"	SETVAR_FL_NTT(256); \\\n" \
"	SUB_FORWARD4i(64, CHUNK); \\\n" \
"	FORWARD4(16, CHUNK, 64 * m); \\\n" \
"	FORWARD4(4, CHUNK, 64 * m + 16 * m); \\\n" \
"	FORWARD4o(CHUNK, 64 * m + 16 * m + 4 * m);\n" \
"\n" \
"#define LST_INTT256(CHUNK) \\\n" \
"	SETVAR(256, CHUNK); \\\n" \
"	SETVAR_FL_NTT(256); \\\n" \
"	BACKWARD4i(CHUNK, 64 * m + 16 * m + 4 * m); \\\n" \
"	BACKWARD4(4, CHUNK, 64 * m + 16 * m); \\\n" \
"	BACKWARD4(16, CHUNK, 64 * m); \\\n" \
"	BACKWARD4o(64, CHUNK, 0);\n" \
"\n" \
"#define NTT256(CHUNK) \\\n" \
"	SETVAR(256, CHUNK); \\\n" \
"	SETVAR_NTT(256); \\\n" \
"	FORWARD4i(64, CHUNK, rindex); \\\n" \
"	FORWARD4(16, CHUNK, rindex + 64 * m); \\\n" \
"	FORWARD4(4, CHUNK, rindex + 64 * m + 16 * m); \\\n" \
"	FORWARD4o(CHUNK, rindex + 64 * m + 16 * m + 4 * m);\n" \
"\n" \
"#define INTT256(CHUNK) \\\n" \
"	SETVAR(256, CHUNK); \\\n" \
"	SETVAR_NTT(256); \\\n" \
"	BACKWARD4i(CHUNK, rindex + 64 * m + 16 * m + 4 * m); \\\n" \
"	BACKWARD4(4, CHUNK, rindex + 64 * m + 16 * m); \\\n" \
"	BACKWARD4(16, CHUNK, rindex + 64 * m); \\\n" \
"	BACKWARD4o(64, CHUNK, rindex);\n" \
"\n" \
"#define SUB_NTT1024(CHUNK) \\\n" \
"	SETVAR(1024, CHUNK); \\\n" \
"	SETVAR_FL_NTT(1024); \\\n" \
"	SUB_FORWARD4i(256, CHUNK); \\\n" \
"	FORWARD4(64, CHUNK, 256 * m); \\\n" \
"	FORWARD4(16, CHUNK, 256 * m + 64 * m); \\\n" \
"	FORWARD4(4, CHUNK, 256 * m + 64 * m + 16 * m); \\\n" \
"	FORWARD4o(CHUNK, 256 * m + 64 * m + 16 * m + 4 * m);\n" \
"\n" \
"#define LST_INTT1024(CHUNK) \\\n" \
"	SETVAR(1024, CHUNK); \\\n" \
"	SETVAR_FL_NTT(1024); \\\n" \
"	BACKWARD4i(CHUNK, 256 * m + 64 * m + 16 * m + 4 * m); \\\n" \
"	BACKWARD4(4, CHUNK, 256 * m + 64 * m + 16 * m); \\\n" \
"	BACKWARD4(16, CHUNK, 256 * m + 64 * m); \\\n" \
"	BACKWARD4(64, CHUNK, 256 * m); \\\n" \
"	BACKWARD4o(256, CHUNK, 0);\n" \
"\n" \
"#define NTT1024(CHUNK) \\\n" \
"	SETVAR(1024, CHUNK); \\\n" \
"	SETVAR_NTT(1024); \\\n" \
"	FORWARD4i(256, CHUNK, rindex); \\\n" \
"	FORWARD4(64, CHUNK, rindex + 256 * m); \\\n" \
"	FORWARD4(16, CHUNK, rindex + 256 * m + 64 * m); \\\n" \
"	FORWARD4(4, CHUNK, rindex + 256 * m + 64 * m + 16 * m); \\\n" \
"	FORWARD4o(CHUNK, rindex + 256 * m + 64 * m + 16 * m + 4 * m);\n" \
"\n" \
"#define INTT1024(CHUNK) \\\n" \
"	SETVAR(1024, CHUNK); \\\n" \
"	SETVAR_NTT(1024); \\\n" \
"	BACKWARD4i(CHUNK, rindex + 256 * m + 64 * m + 16 * m + 4 * m); \\\n" \
"	BACKWARD4(4, CHUNK, rindex + 256 * m + 64 * m + 16 * m); \\\n" \
"	BACKWARD4(16, CHUNK, rindex + 256 * m + 64 * m); \\\n" \
"	BACKWARD4(64, CHUNK, rindex + 256 * m); \\\n" \
"	BACKWARD4o(256, CHUNK, rindex);\n" \
"\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(64 / 4 * 16, 1, 1)))\n" \
"void sub_ntt64_16(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2)\n" \
"{\n" \
"	SUB_NTT64(16);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(64 / 4 * 16, 1, 1)))\n" \
"void lst_intt64_16(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2)\n" \
"{\n" \
"	LST_INTT64(16);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(64 / 4 * 16, 1, 1)))\n" \
"void ntt64_16(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)\n" \
"{\n" \
"	NTT64(16);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(64 / 4 * 16, 1, 1)))\n" \
"void intt64_16(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)\n" \
"{\n" \
"	INTT64(16);\n" \
"}\n" \
"\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 4 * 4, 1, 1)))\n" \
"void sub_ntt256_4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2)\n" \
"{\n" \
"	SUB_NTT256(4);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 4 * 4, 1, 1)))\n" \
"void lst_intt256_4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2)\n" \
"{\n" \
"	LST_INTT256(4);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 4 * 4, 1, 1)))\n" \
"void ntt256_4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)\n" \
"{\n" \
"	NTT256(4);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 4 * 4, 1, 1)))\n" \
"void intt256_4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)\n" \
"{\n" \
"	INTT256(4);\n" \
"}\n" \
"\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 4 * 1, 1, 1)))\n" \
"void sub_ntt1024_1(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2)\n" \
"{\n" \
"	SUB_NTT1024(1);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 4 * 1, 1, 1)))\n" \
"void lst_intt1024_1(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2)\n" \
"{\n" \
"	LST_INTT1024(1);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 4 * 1, 1, 1)))\n" \
"void ntt1024_1(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)\n" \
"{\n" \
"	NTT1024(1);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 4 * 1, 1, 1)))\n" \
"void intt1024_1(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)\n" \
"{\n" \
"	INTT1024(1);\n" \
"}\n" \
"";
