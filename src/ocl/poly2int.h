/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

static const char * const src_ocl_poly2int = \
"/*\n" \
"Copyright 2020, Yves Gallot\n" \
"\n" \
"proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\n" \
"Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n" \
"*/\n" \
"\n" \
"#define POLY2INT0_VAR(P2I_BLK, P2I_WGS) \\\n" \
"	__local long L[P2I_WGS * P2I_BLK]; \\\n" \
"	__local uint X[P2I_WGS * P2I_BLK];\n" \
" \n" \
"inline void poly2int0(__local long * restrict const L, __local uint * restrict const X, const size_t P2I_BLK, const size_t P2I_WGS,\n" \
" 	__global uint2 * restrict const x, __global long * restrict const cr)\n" \
" {\n" \
"	const size_t i = get_local_id(0), blk = get_group_id(0);\n" \
"	const size_t kc = (get_global_id(0) + 1) & (get_global_size(0) - 1);\n" \
"\n" \
"	__global uint2 * const xo = &x[P2I_WGS * P2I_BLK * blk];\n" \
"\n" \
"	for (size_t j = 0; j < P2I_BLK; ++j)\n" \
"	{\n" \
"		const size_t k = P2I_WGS * j + i;\n" \
"		L[P2I_WGS * (k % P2I_BLK) + (k / P2I_BLK)] = getlong(mulmod(xo[k], pconst_norm));	// -n/2 . (B-1)^2 <= l <= n/2 . (B-1)^2\n" \
"	}\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	long l = 0;\n" \
"//#pragma unroll\n" \
"	for (size_t j = 0; j < P2I_BLK; ++j)\n" \
"	{\n" \
"		l += L[P2I_WGS * j + i];\n" \
"		X[P2I_WGS * j + i] = (uint)(l) & digit_mask;\n" \
"	 	l >>= digit_bit;\n" \
"	}\n" \
"	cr[kc] = l;\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"//#pragma unroll\n" \
"	for (size_t j = 0; j < P2I_BLK; ++j)\n" \
"	{\n" \
"		const size_t k = P2I_WGS * j + i;\n" \
"		xo[k].s0 = X[P2I_WGS * (k % P2I_BLK) + (k / P2I_BLK)];\n" \
"	}\n" \
"}\n" \
"\n" \
"inline void poly2int1(const size_t P2I_BLK, __global uint2 * restrict const x, __global const long * restrict const cr, __global int * const err)\n" \
"{\n" \
"	const size_t k = get_global_id(0);\n" \
"\n" \
"	__global uint2 * const xi = &x[P2I_BLK * k];\n" \
"\n" \
"	long l = cr[k] + xi[0].s0;\n" \
"	xi[0].s0 = (uint)(l) & digit_mask;\n" \
"	l >>= digit_bit;						// |l| < n/2\n" \
"\n" \
"	int f = (int)(l);\n" \
"//#pragma unroll\n" \
"	for (size_t j = 1; j < P2I_BLK - 1; ++j)\n" \
"	{\n" \
"		f += xi[j].s0;\n" \
"		xi[j].s0 = (uint)(f) & digit_mask;\n" \
"		f >>= digit_bit;					// f = -1, 0 or 1\n" \
"		if (f == 0) return;\n" \
"	}\n" \
"\n" \
"	f += xi[P2I_BLK - 1].s0;\n" \
"	xi[P2I_BLK - 1].s0 = (uint)(f);\n" \
"	f >>= digit_bit;\n" \
"	if (f != 0) atomic_or(&err[1], f);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void poly2int2(__global uint2 * restrict const x, __global int * const err)\n" \
"{\n" \
"	if (err[1] == 0) return;\n" \
"\n" \
"	int f = 0;\n" \
"	for (size_t k = 0; k < pconst_size; ++k)\n" \
"	{\n" \
"		f += x[k].s0;\n" \
"		x[k] = (uint)(f) & digit_mask;\n" \
"		f >>= digit_bit;\n" \
"	}\n" \
"\n" \
"	err[0] = f;\n" \
"	err[1] = 0;\n" \
"}\n" \
"\n" \
"// P2I_BLK = 4\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(16, 1, 1)))\n" \
"void poly2int0_4_16(__global uint2 * restrict const x, __global long * restrict const cr)\n" \
"{\n" \
"	POLY2INT0_VAR(4, 16);\n" \
"	poly2int0(L, X, 4, 16, x, cr);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(32, 1, 1)))\n" \
"void poly2int0_4_32(__global uint2 * restrict const x, __global long * restrict const cr)\n" \
"{\n" \
"	POLY2INT0_VAR(4, 32);\n" \
"	poly2int0(L, X, 4, 32, x, cr);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(64, 1, 1)))\n" \
"void poly2int0_4_64(__global uint2 * restrict const x, __global long * restrict const cr)\n" \
"{\n" \
"	POLY2INT0_VAR(4, 64);\n" \
"	poly2int0(L, X, 4, 64, x, cr);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void poly2int1_4(__global uint2 * restrict const x, __global const long * restrict const cr, __global int * const err)\n" \
"{\n" \
"	poly2int1(4, x, cr, err);\n" \
"}\n" \
"\n" \
"// P2I_BLK = 8\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(16, 1, 1)))\n" \
"void poly2int0_8_16(__global uint2 * restrict const x, __global long * restrict const cr)\n" \
"{\n" \
"	POLY2INT0_VAR(8, 16);\n" \
"	poly2int0(L, X, 8, 16, x, cr);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(32, 1, 1)))\n" \
"void poly2int0_8_32(__global uint2 * restrict const x, __global long * restrict const cr)\n" \
"{\n" \
"	POLY2INT0_VAR(8, 32);\n" \
"	poly2int0(L, X, 8, 32, x, cr);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(64, 1, 1)))\n" \
"void poly2int0_8_64(__global uint2 * restrict const x, __global long * restrict const cr)\n" \
"{\n" \
"	POLY2INT0_VAR(8, 64);\n" \
"	poly2int0(L, X, 8, 64, x, cr);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void poly2int1_8(__global uint2 * restrict const x, __global const long * restrict const cr, __global int * const err)\n" \
"{\n" \
"	poly2int1(8, x, cr, err);\n" \
"}\n" \
"\n" \
"// P2I_BLK = 16\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(8, 1, 1)))\n" \
"void poly2int0_16_8(__global uint2 * restrict const x, __global long * restrict const cr)\n" \
"{\n" \
"	POLY2INT0_VAR(16, 8);\n" \
"	poly2int0(L, X, 16, 8, x, cr);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(16, 1, 1)))\n" \
"void poly2int0_16_16(__global uint2 * restrict const x, __global long * restrict const cr)\n" \
"{\n" \
"	POLY2INT0_VAR(16, 16);\n" \
"	poly2int0(L, X, 16, 16, x, cr);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(32, 1, 1)))\n" \
"void poly2int0_16_32(__global uint2 * restrict const x, __global long * restrict const cr)\n" \
"{\n" \
"	POLY2INT0_VAR(16, 32);\n" \
"	poly2int0(L, X, 16, 32, x, cr);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void poly2int1_16(__global uint2 * restrict const x, __global const long * restrict const cr, __global int * const err)\n" \
"{\n" \
"	poly2int1(16, x, cr, err);\n" \
"}\n" \
"";
