/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

static const char * const src_ocl_squareNTT_1024 = \
"/*\n" \
"Copyright 2020, Yves Gallot\n" \
"\n" \
"proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\n" \
"Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n" \
"*/\n" \
"\n" \
"// __local 32k\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 4 * 16, 1, 1)))\n" \
"void sub_ntt256_16(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2)\n" \
"{\n" \
"	SUB_NTT256(16);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 4 * 16, 1, 1)))\n" \
"void lst_intt256_16(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2)\n" \
"{\n" \
"	LST_INTT256(16);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 4 * 16, 1, 1)))\n" \
"void ntt256_16(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)\n" \
"{\n" \
"	NTT256(16);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 4 * 16, 1, 1)))\n" \
"void intt256_16(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)\n" \
"{\n" \
"	INTT256(16);\n" \
"}\n" \
"\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 4 * 4, 1, 1)))\n" \
"void sub_ntt1024_4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2)\n" \
"{\n" \
"	SUB_NTT1024(4);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 4 * 4, 1, 1)))\n" \
"void lst_intt1024_4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2)\n" \
"{\n" \
"	LST_INTT1024(4);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 4 * 4, 1, 1)))\n" \
"void ntt1024_4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)\n" \
"{\n" \
"	NTT1024(4);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 4 * 4, 1, 1)))\n" \
"void intt1024_4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)\n" \
"{\n" \
"	INTT1024(4);\n" \
"}\n" \
"\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(4096 / 4, 1, 1)))\n" \
"void square4096(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1, __constmem const uint4 * restrict const ir1,\n" \
"	__constmem const uint4 * restrict const r2, __constmem const uint4 * restrict const ir2)\n" \
"{\n" \
"	__local uint2 X[4096];	// 32k\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"	const size_t i1024 = i, j1024 = i + 4 + 16 + 64 + 256;\n" \
"	const size_t i_256 = i % 256, i256 = ((4 * i) & (size_t)~(4 * 256 - 1)) | i_256, j256 = i_256 + 4 + 16 + 64;\n" \
"	const size_t i_64 = i % 64, i64 = ((4 * i) & (size_t)~(4 * 64 - 1)) | i_64, j64 = i_64 + 4 + 16;\n" \
"	const size_t i_16 = i % 16, i16 = ((4 * i) & (size_t)~(4 * 16 - 1)) | i_16, j16 = i_16 + 4;\n" \
"	const size_t i_4 = i % 4, i4 = ((4 * i) & (size_t)~(4 * 4 - 1)) | i_4, j4 = i_4;\n" \
"	const size_t k1024 = get_group_id(0) * 4096 | i1024;\n" \
"\n" \
"	const uint4 r1_1024 = r1[j1024], ir1_1024 = ir1[j1024];\n" \
"	_forward4pi(1024, &X[i1024], 1024, &x[k1024], r2[j1024], r1_1024, ir1_1024);\n" \
"	const uint4 r1_256 = r1[j256], ir1_256 = ir1[j256];\n" \
"	_forward4p(256, &X[i256], r2[j256], r1_256, ir1_256);\n" \
"	const uint4 r1_64 = r1[j64], ir1_64 = ir1[j64];\n" \
"	_forward4p(64, &X[i64], r2[j64], r1_64, ir1_64);\n" \
"	const uint4 r1_16 = r1[j16], ir1_16 = ir1[j16];\n" \
"	_forward4p(16, &X[i16], r2[j16], r1_16, ir1_16);\n" \
"	const uint4 r1_4 = r1[j4], ir1_4 = ir1[j4];\n" \
"	_forward4p(4, &X[i4], r2[j4], r1_4, ir1_4);\n" \
"	_square4(&X[4 * i]);\n" \
"	_backward4p(4, &X[i4], ir2[j4], r1_4, ir1_4);\n" \
"	_backward4p(16, &X[i16], ir2[j16], r1_16, ir1_16);\n" \
"	_backward4p(64, &X[i64], ir2[j64], r1_64, ir1_64);\n" \
"	_backward4p(256, &X[i256], ir2[j256], r1_256, ir1_256);\n" \
"	_backward4po(1024, &x[k1024], 1024, &X[i1024], ir2[j1024], r1_1024, ir1_1024);\n" \
"}\n" \
"";
