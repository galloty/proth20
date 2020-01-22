/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

static const char * const src_proth_1024_ocl = \
"/*\n" \
"Copyright 2020, Yves Gallot\n" \
"\n" \
"proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\n" \
"Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n" \
"*/\n" \
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
"\n" \
"__kernel __attribute__((reqd_work_group_size(4096 / 4, 1, 1)))\n" \
"void square4096(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)\n" \
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
"	_forward4i(1024, &X[i1024], 1024, &x[k1024], r2ir2[j1024].s01, r1ir1[j1024]);\n" \
"	_forward4(256, &X[i256], r2ir2[j256].s01, r1ir1[j256]);\n" \
"	_forward4(64, &X[i64], r2ir2[j64].s01, r1ir1[j64]);\n" \
"	_forward4(16, &X[i16], r2ir2[j16].s01, r1ir1[j16]);\n" \
"	_forward4(4, &X[i4], r2ir2[j4].s01, r1ir1[j4]);\n" \
"	_square4(&X[4 * i]);\n" \
"	_backward4(4, &X[i4], r2ir2[j4].s23, r1ir1[j4]);\n" \
"	_backward4(16, &X[i16], r2ir2[j16].s23, r1ir1[j16]);\n" \
"	_backward4(64, &X[i64], r2ir2[j64].s23, r1ir1[j64]);\n" \
"	_backward4(256, &X[i256], r2ir2[j256].s23, r1ir1[j256]);\n" \
"	_backward4o(1024, &x[k1024], 1024, &X[i1024], r2ir2[j1024].s23, r1ir1[j1024]);\n" \
"}\n" \
;
