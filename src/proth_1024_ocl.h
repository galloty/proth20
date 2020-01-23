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
"#define CHUNK256	8\n" \
"#define CHUNK1024	4\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 4 * CHUNK256, 1, 1)))\n" \
"void sub_ntt256(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2)\n" \
"{\n" \
"	__local uint2 X[256 * CHUNK256];\n" \
"\n" \
"	const size_t m = get_global_size(0) / 64;\n" \
"\n" \
"	const size_t local_id = get_local_id(0), chunk_idx = local_id % CHUNK256, threadIdx = local_id / CHUNK256, block_idx = get_group_id(0) * CHUNK256;\n" \
"\n" \
"	const size_t bl_i = block_idx | chunk_idx;\n" \
"\n" \
"	const size_t _i_64m = threadIdx;\n" \
"	const size_t _i_16m = ((4 * threadIdx) & ~(16 * 4 - 1)) | (threadIdx % 16);\n" \
"	const size_t _i_4m = ((4 * threadIdx) & ~(4 * 4 - 1)) | (threadIdx % 4);\n" \
"	const size_t _i_m = 4 * threadIdx;\n" \
"\n" \
"	const size_t i_64m = _i_64m * CHUNK256 | chunk_idx;\n" \
"	const size_t i_16m = _i_16m * CHUNK256 | chunk_idx;\n" \
"	const size_t i_4m = _i_4m * CHUNK256 | chunk_idx;\n" \
"	const size_t i_m = _i_m * CHUNK256 | chunk_idx;\n" \
"\n" \
"	const size_t k_64m = _i_64m * m | bl_i;\n" \
"	const size_t k_16m = _i_16m * m | bl_i;\n" \
"	const size_t k_4m = _i_4m * m | bl_i;\n" \
"	const size_t k_m = _i_m * m | bl_i;\n" \
"\n" \
"	const size_t j_64m = k_64m;	// & (64 * m - 1);\n" \
"	const size_t j_16m = k_16m & (16 * m - 1);\n" \
"	const size_t j_4m = k_4m & (4 * m - 1);\n" \
"	const size_t j_m = bl_i;	// k_m & (m - 1);\n" \
"\n" \
"	_sub_forward4i(64 * CHUNK256, &X[i_64m], 64 * m, &x[k_64m], r2[j_64m], r1ir1[j_64m]);\n" \
"	_forward4(16 * CHUNK256, &X[i_16m], r2[64 * m + j_16m], r1ir1[64 * m + j_16m]);\n" \
"	_forward4(4 * CHUNK256, &X[i_4m], r2[64 * m + 16 * m + j_4m], r1ir1[64 * m + 16 * m + j_4m]);\n" \
"	_forward4o(m, &x[k_m], CHUNK256, &X[i_m], r2[64 * m + 16 * m + 4 * m + j_m], r1ir1[64 * m + 16 * m + 4 * m + j_m]);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 4 * CHUNK256, 1, 1)))\n" \
"void ntt256(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)\n" \
"{\n" \
"	__local uint2 X[256 * CHUNK256];\n" \
"\n" \
"	const size_t local_id = get_local_id(0), chunk_idx = local_id % CHUNK256, threadIdx = local_id / CHUNK256, block_idx = get_group_id(0) * CHUNK256;\n" \
"\n" \
"	__global uint2 * const xo = &x[256 * (block_idx & ~(m - 1))];		// m-block offset\n" \
"	const size_t bl_i = (block_idx & (m - 1)) | chunk_idx;\n" \
"\n" \
"	const size_t _i_64m = threadIdx;\n" \
"	const size_t _i_16m = ((4 * threadIdx) & ~(16 * 4 - 1)) | (threadIdx % 16);\n" \
"	const size_t _i_4m = ((4 * threadIdx) & ~(4 * 4 - 1)) | (threadIdx % 4);\n" \
"	const size_t _i_m = 4 * threadIdx;\n" \
"\n" \
"	const size_t i_64m = _i_64m * CHUNK256 | chunk_idx;\n" \
"	const size_t i_16m = _i_16m * CHUNK256 | chunk_idx;\n" \
"	const size_t i_4m = _i_4m * CHUNK256 | chunk_idx;\n" \
"	const size_t i_m = _i_m * CHUNK256 | chunk_idx;\n" \
"\n" \
"	const size_t k_64m = _i_64m * m | bl_i;\n" \
"	const size_t k_16m = _i_16m * m | bl_i;\n" \
"	const size_t k_4m = _i_4m * m | bl_i;\n" \
"	const size_t k_m = _i_m * m | bl_i;\n" \
"\n" \
"	const size_t j_64m = k_64m;	// & (64 * m - 1);\n" \
"	const size_t j_16m = k_16m & (16 * m - 1);\n" \
"	const size_t j_4m = k_4m & (4 * m - 1);\n" \
"	const size_t j_m = bl_i;	// k_m & (m - 1); We have bl_i < m\n" \
"\n" \
"	_forward4i(64 * CHUNK256, &X[i_64m], 64 * m, &xo[k_64m], r2[rindex + j_64m], r1ir1[rindex + j_64m]);\n" \
"	_forward4(16 * CHUNK256, &X[i_16m], r2[rindex + 64 * m + j_16m], r1ir1[rindex + 64 * m + j_16m]);\n" \
"	_forward4(4 * CHUNK256, &X[i_4m], r2[rindex + 64 * m + 16 * m + j_4m], r1ir1[rindex + 64 * m + 16 * m + j_4m]);\n" \
"	_forward4o(m, &xo[k_m], CHUNK256, &X[i_m], r2[rindex + 64 * m + 16 * m + 4 * m + j_m], r1ir1[rindex + 64 * m + 16 * m + 4 * m + j_m]);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 4 * CHUNK256, 1, 1)))\n" \
"void intt256(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)\n" \
"{\n" \
"	__local uint2 X[256 * CHUNK256];\n" \
"\n" \
"	const size_t local_id = get_local_id(0), chunk_idx = local_id % CHUNK256, threadIdx = local_id / CHUNK256, block_idx = get_group_id(0) * CHUNK256;\n" \
"\n" \
"	__global uint2 * const xo = &x[256 * (block_idx & ~(m - 1))];		// m-block offset\n" \
"	const size_t bl_i = (block_idx & (m - 1)) | chunk_idx;\n" \
"\n" \
"	const size_t _i_64m = threadIdx;\n" \
"	const size_t _i_16m = ((4 * threadIdx) & ~(16 * 4 - 1)) | (threadIdx % 16);\n" \
"	const size_t _i_4m = ((4 * threadIdx) & ~(4 * 4 - 1)) | (threadIdx % 4);\n" \
"	const size_t _i_m = 4 * threadIdx;\n" \
"\n" \
"	const size_t i_64m = _i_64m * CHUNK256 | chunk_idx;\n" \
"	const size_t i_16m = _i_16m * CHUNK256 | chunk_idx;\n" \
"	const size_t i_4m = _i_4m * CHUNK256 | chunk_idx;\n" \
"	const size_t i_m = _i_m * CHUNK256 | chunk_idx;\n" \
"\n" \
"	const size_t k_64m = _i_64m * m | bl_i;\n" \
"	const size_t k_16m = _i_16m * m | bl_i;\n" \
"	const size_t k_4m = _i_4m * m | bl_i;\n" \
"	const size_t k_m = _i_m * m | bl_i;\n" \
"\n" \
"	const size_t j_64m = k_64m;	// & (64 * m - 1);\n" \
"	const size_t j_16m = k_16m & (16 * m - 1);\n" \
"	const size_t j_4m = k_4m & (4 * m - 1);\n" \
"	const size_t j_m = bl_i;	// k_m & (m - 1); We have bl_i < m\n" \
"\n" \
"	_backward4i(CHUNK256, &X[i_m], m, &xo[k_m], ir2[rindex + 64 * m + 16 * m + 4 * m + j_m], r1ir1[rindex + 64 * m + 16 * m + 4 * m + j_m]);\n" \
"	_backward4(4 * CHUNK256, &X[i_4m], ir2[rindex + 64 * m + 16 * m + j_4m], r1ir1[rindex + 64 * m + 16 * m + j_4m]);\n" \
"	_backward4(16 * CHUNK256, &X[i_16m], ir2[rindex + 64 * m + j_16m], r1ir1[rindex + 64 * m + j_16m]);\n" \
"	_backward4o(64 * m, &xo[k_64m], 64 * CHUNK256, &X[i_64m], ir2[rindex + j_64m], r1ir1[rindex + j_64m]);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 4 * CHUNK1024, 1, 1)))\n" \
"void sub_ntt1024(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2)\n" \
"{\n" \
"	__local uint2 X[1024 * CHUNK1024];\n" \
"\n" \
"	const size_t m = get_global_size(0) / 256;\n" \
"\n" \
"	const size_t local_id = get_local_id(0), chunk_idx = local_id % CHUNK1024, threadIdx = local_id / CHUNK1024, block_idx = get_group_id(0) * CHUNK1024;\n" \
"\n" \
"	const size_t bl_i = block_idx | chunk_idx;\n" \
"\n" \
"	const size_t _i_256m = threadIdx;\n" \
"	const size_t _i_64m = ((4 * threadIdx) & ~(64 * 4 - 1)) | (threadIdx % 64);\n" \
"	const size_t _i_16m = ((4 * threadIdx) & ~(16 * 4 - 1)) | (threadIdx % 16);\n" \
"	const size_t _i_4m = ((4 * threadIdx) & ~(4 * 4 - 1)) | (threadIdx % 4);\n" \
"	const size_t _i_m = 4 * threadIdx;\n" \
"\n" \
"	const size_t i_256m = _i_256m * CHUNK1024 | chunk_idx;\n" \
"	const size_t i_64m = _i_64m * CHUNK1024 | chunk_idx;\n" \
"	const size_t i_16m = _i_16m * CHUNK1024 | chunk_idx;\n" \
"	const size_t i_4m = _i_4m * CHUNK1024 | chunk_idx;\n" \
"	const size_t i_m = _i_m * CHUNK1024 | chunk_idx;\n" \
"\n" \
"	const size_t k_256m = _i_256m * m | bl_i;\n" \
"	const size_t k_64m = _i_64m * m | bl_i;\n" \
"	const size_t k_16m = _i_16m * m | bl_i;\n" \
"	const size_t k_4m = _i_4m * m | bl_i;\n" \
"	const size_t k_m = _i_m * m | bl_i;\n" \
"\n" \
"	const size_t j_256m = k_256m;	// & (256 * m - 1);\n" \
"	const size_t j_64m = k_64m & (64 * m - 1);\n" \
"	const size_t j_16m = k_16m & (16 * m - 1);\n" \
"	const size_t j_4m = k_4m & (4 * m - 1);\n" \
"	const size_t j_m = bl_i;	// k_m & (m - 1);\n" \
"\n" \
"	_sub_forward4i(256 * CHUNK1024, &X[i_256m], 256 * m, &x[k_256m], r2[j_256m], r1ir1[j_256m]);\n" \
"	_forward4(64 * CHUNK1024, &X[i_64m], r2[256 * m + j_64m], r1ir1[256 * m + j_64m]);\n" \
"	_forward4(16 * CHUNK1024, &X[i_16m], r2[256 * m + 64 * m + j_16m], r1ir1[256 * m + 64 * m + j_16m]);\n" \
"	_forward4(4 * CHUNK1024, &X[i_4m], r2[256 * m + 64 * m + 16 * m + j_4m], r1ir1[256 * m + 64 * m + 16 * m + j_4m]);\n" \
"	_forward4o(m, &x[k_m], CHUNK1024, &X[i_m], r2[256 * m + 64 * m + 16 * m + 4 * m + j_m], r1ir1[256 * m + 64 * m + 16 * m + 4 * m + j_m]);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 4 * CHUNK1024, 1, 1)))\n" \
"void ntt1024(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)\n" \
"{\n" \
"	__local uint2 X[1024 * CHUNK1024];\n" \
"\n" \
"	const size_t local_id = get_local_id(0), chunk_idx = local_id % CHUNK1024, threadIdx = local_id / CHUNK1024, block_idx = get_group_id(0) * CHUNK1024;\n" \
"\n" \
"	__global uint2 * const xo = &x[1024 * (block_idx & ~(m - 1))];		// m-block offset\n" \
"	const size_t bl_i = (block_idx & (m - 1)) | chunk_idx;\n" \
"\n" \
"	const size_t _i_256m = threadIdx;\n" \
"	const size_t _i_64m = ((4 * threadIdx) & ~(64 * 4 - 1)) | (threadIdx % 64);\n" \
"	const size_t _i_16m = ((4 * threadIdx) & ~(16 * 4 - 1)) | (threadIdx % 16);\n" \
"	const size_t _i_4m = ((4 * threadIdx) & ~(4 * 4 - 1)) | (threadIdx % 4);\n" \
"	const size_t _i_m = 4 * threadIdx;\n" \
"\n" \
"	const size_t i_256m = _i_256m * CHUNK1024 | chunk_idx;\n" \
"	const size_t i_64m = _i_64m * CHUNK1024 | chunk_idx;\n" \
"	const size_t i_16m = _i_16m * CHUNK1024 | chunk_idx;\n" \
"	const size_t i_4m = _i_4m * CHUNK1024 | chunk_idx;\n" \
"	const size_t i_m = _i_m * CHUNK1024 | chunk_idx;\n" \
"\n" \
"	const size_t k_256m = _i_256m * m | bl_i;\n" \
"	const size_t k_64m = _i_64m * m | bl_i;\n" \
"	const size_t k_16m = _i_16m * m | bl_i;\n" \
"	const size_t k_4m = _i_4m * m | bl_i;\n" \
"	const size_t k_m = _i_m * m | bl_i;\n" \
"\n" \
"	const size_t j_256m = k_256m;	// & (256 * m - 1);\n" \
"	const size_t j_64m = k_64m & (64 * m - 1);\n" \
"	const size_t j_16m = k_16m & (16 * m - 1);\n" \
"	const size_t j_4m = k_4m & (4 * m - 1);\n" \
"	const size_t j_m = bl_i;	// k_m & (m - 1); We have bl_i < m\n" \
"\n" \
"	_forward4i(256 * CHUNK1024, &X[i_256m], 256 * m, &xo[k_256m], r2[rindex + j_256m], r1ir1[rindex + j_256m]);\n" \
"	_forward4(64 * CHUNK1024, &X[i_64m], r2[rindex + 256 * m + j_64m], r1ir1[rindex + 256 * m + j_64m]);\n" \
"	_forward4(16 * CHUNK1024, &X[i_16m], r2[rindex + 256 * m + 64 * m + j_16m], r1ir1[rindex + 256 * m + 64 * m + j_16m]);\n" \
"	_forward4(4 * CHUNK1024, &X[i_4m], r2[rindex + 256 * m + 64 * m + 16 * m + j_4m], r1ir1[rindex + 256 * m + 64 * m + 16 * m + j_4m]);\n" \
"	_forward4o(m, &xo[k_m], CHUNK1024, &X[i_m], r2[rindex + 256 * m + 64 * m + 16 * m + 4 * m + j_m], r1ir1[rindex + 256 * m + 64 * m + 16 * m + 4 * m + j_m]);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 4 * CHUNK1024, 1, 1)))\n" \
"void intt1024(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)\n" \
"{\n" \
"	__local uint2 X[1024 * CHUNK1024];\n" \
"\n" \
"	const size_t local_id = get_local_id(0), chunk_idx = local_id % CHUNK1024, threadIdx = local_id / CHUNK1024, block_idx = get_group_id(0) * CHUNK1024;\n" \
"\n" \
"	__global uint2 * const xo = &x[1024 * (block_idx & ~(m - 1))];		// m-block offset\n" \
"	const size_t bl_i = (block_idx & (m - 1)) | chunk_idx;\n" \
"\n" \
"	const size_t _i_256m = threadIdx;\n" \
"	const size_t _i_64m = ((4 * threadIdx) & ~(64 * 4 - 1)) | (threadIdx % 64);\n" \
"	const size_t _i_16m = ((4 * threadIdx) & ~(16 * 4 - 1)) | (threadIdx % 16);\n" \
"	const size_t _i_4m = ((4 * threadIdx) & ~(4 * 4 - 1)) | (threadIdx % 4);\n" \
"	const size_t _i_m = 4 * threadIdx;\n" \
"\n" \
"	const size_t i_256m = _i_256m * CHUNK1024 | chunk_idx;\n" \
"	const size_t i_64m = _i_64m * CHUNK1024 | chunk_idx;\n" \
"	const size_t i_16m = _i_16m * CHUNK1024 | chunk_idx;\n" \
"	const size_t i_4m = _i_4m * CHUNK1024 | chunk_idx;\n" \
"	const size_t i_m = _i_m * CHUNK1024 | chunk_idx;\n" \
"\n" \
"	const size_t k_256m = _i_256m * m | bl_i;\n" \
"	const size_t k_64m = _i_64m * m | bl_i;\n" \
"	const size_t k_16m = _i_16m * m | bl_i;\n" \
"	const size_t k_4m = _i_4m * m | bl_i;\n" \
"	const size_t k_m = _i_m * m | bl_i;\n" \
"\n" \
"	const size_t j_256m = k_256m;	// & (256 * m - 1);\n" \
"	const size_t j_64m = k_64m & (64 * m - 1);\n" \
"	const size_t j_16m = k_16m & (16 * m - 1);\n" \
"	const size_t j_4m = k_4m & (4 * m - 1);\n" \
"	const size_t j_m = bl_i;	// k_m & (m - 1); We have bl_i < m\n" \
"\n" \
"	_backward4i(CHUNK1024, &X[i_m], m, &xo[k_m], ir2[rindex + 256 * m + 64 * m + 16 * m + 4 * m + j_m], r1ir1[rindex + 256 * m + 64 * m + 16 * m + 4 * m + j_m]);\n" \
"	_backward4(4 * CHUNK1024, &X[i_4m], ir2[rindex + 256 * m + 64 * m + 16 * m + j_4m], r1ir1[rindex + 256 * m + 64 * m + 16 * m + j_4m]);\n" \
"	_backward4(16 * CHUNK1024, &X[i_16m], ir2[rindex + 256 * m + 64 * m + j_16m], r1ir1[rindex + 256 * m + 64 * m + j_16m]);\n" \
"	_backward4(64 * CHUNK1024, &X[i_64m], ir2[rindex + 256 * m + j_64m], r1ir1[rindex + 256 * m + j_64m]);\n" \
"	_backward4o(256 * m, &xo[k_256m], 256 * CHUNK1024, &X[i_256m], ir2[rindex + j_256m], r1ir1[rindex + j_256m]);\n" \
"}\n" \
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
