/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

static const char * const src_ocl_squareNTT_512 = \
"/*\n" \
"Copyright 2020, Yves Gallot\n" \
"\n" \
"proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\n" \
"Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n" \
"*/\n" \
"\n" \
"#define CHUNK256	8\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 4 * CHUNK256, 1, 1)))\n" \
"void sub_ntt256(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2)\n" \
"{\n" \
"	SETVAR(256, CHUNK256);\n" \
"	SETVAR_SUB_NTT(256);\n" \
"\n" \
"	SUB_FORWARD4i(64, CHUNK256);\n" \
"	FORWARD4(16, CHUNK256, 64 * m);\n" \
"	FORWARD4(4, CHUNK256, 64 * m + 16 * m);\n" \
"	FORWARD4o(CHUNK256, 64 * m + 16 * m + 4 * m);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 4 * CHUNK256, 1, 1)))\n" \
"void ntt256(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)\n" \
"{\n" \
"	SETVAR(256, CHUNK256);\n" \
"	SETVAR_NTT(256);\n" \
"\n" \
"	FORWARD4i(64, CHUNK256, rindex);\n" \
"	FORWARD4(16, CHUNK256, rindex + 64 * m);\n" \
"	FORWARD4(4, CHUNK256, rindex + 64 * m + 16 * m);\n" \
"	FORWARD4o(CHUNK256, rindex + 64 * m + 16 * m + 4 * m);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 4 * CHUNK256, 1, 1)))\n" \
"void intt256(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)\n" \
"{\n" \
"	SETVAR(256, CHUNK256);\n" \
"	SETVAR_NTT(256);\n" \
"\n" \
"	BACKWARD4i(CHUNK256, rindex + 64 * m + 16 * m + 4 * m);\n" \
"	BACKWARD4(4, CHUNK256, rindex + 64 * m + 16 * m);\n" \
"	BACKWARD4(16, CHUNK256, rindex + 64 * m);\n" \
"	BACKWARD4o(64, CHUNK256, rindex);\n" \
"}\n" \
"\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(2048 / 4, 1, 1)))\n" \
"void square2048(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)\n" \
"{\n" \
"	__local uint2 X[2048];	// 16k\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"	const size_t i512 = i, j512 = i + 2 + 8 + 32 + 128;\n" \
"	const size_t i_128 = i % 128, i128 = ((4 * i) & (size_t)~(4 * 128 - 1)) | i_128, j128 = i_128 + 2 + 8 + 32;\n" \
"	const size_t i_32 = i % 32, i32 = ((4 * i) & (size_t)~(4 * 32 - 1)) | i_32, j32 = i_32 + 2 + 8;\n" \
"	const size_t i_8 = i % 8, i8 = ((4 * i) & (size_t)~(4 * 8 - 1)) | i_8, j8 = i_8 + 2;\n" \
"	const size_t i_2 = i % 2, _i2 = ((4 * i) & (size_t)~(4 * 2 - 1)), i2 = _i2 | i_2, j2 = i_2, i_0 = _i2 | (2 * i_2);\n" \
"	const size_t k512 = get_group_id(0) * 2048 | i512;\n" \
"\n" \
"	_forward4i(512, &X[i512], 512, &x[k512], r2ir2[j512].s01, r1ir1[j512]);\n" \
"	_forward4(128, &X[i128], r2ir2[j128].s01, r1ir1[j128]);\n" \
"	_forward4(32, &X[i32], r2ir2[j32].s01, r1ir1[j32]);\n" \
"	_forward4(8, &X[i8], r2ir2[j8].s01, r1ir1[j8]);\n" \
"	_forward4(2, &X[i2], r2ir2[j2].s01, r1ir1[j2]);\n" \
"	_square2(&X[i_0]);\n" \
"	_backward4(2, &X[i2], r2ir2[j2].s23, r1ir1[j2]);\n" \
"	_backward4(8, &X[i8], r2ir2[j8].s23, r1ir1[j8]);\n" \
"	_backward4(32, &X[i32], r2ir2[j32].s23, r1ir1[j32]);\n" \
"	_backward4(128, &X[i128], r2ir2[j128].s23, r1ir1[j128]);\n" \
"	_backward4o(512, &x[k512], 512, &X[i512], r2ir2[j512].s23, r1ir1[j512]);\n" \
"}\n" \
;
