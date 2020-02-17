/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

static const char * const src_ocl_reduce = \
"/*\n" \
"Copyright 2020, Yves Gallot\n" \
"\n" \
"proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\n" \
"Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n" \
"*/\n" \
"\n" \
"#define	R64		(RED_BLK * 64 / 4)\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(R64, 1, 1)))\n" \
"void reduce_upsweep64(__global uint * restrict const t, const uint s, const uint j)\n" \
"{\n" \
"	__local uint4 T_4[R64 / 4 + R64 / 16];	// alignment\n" \
"	__local uint4 * const T2_4 = &T_4[R64 / 4];\n" \
"	__local uint * const T = (__local uint *)T_4;\n" \
"	__local uint * const T_2 = (__local uint *)T2_4;\n" \
"\n" \
"	const size_t i = get_local_id(0), blk = get_group_id(0), k = get_global_id(0);	// blk * R64 + i;\n" \
"	__global uint * const tj = &t[j];\n" \
"\n" \
"	__global const uint4 * const tj1_4 = (__global const uint4 *)&tj[0];\n" \
"	const uint4 u = tj1_4[k];\n" \
"	const uint u01 = addmod_d(u.s0, u.s1), u23 = addmod_d(u.s2, u.s3), u0123 = addmod_d(u01, u23);\n" \
"	T[i] = u0123;\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	tj[4 * (16 * s) + k] = u23; tj[5 * (16 * s) + k] = u0123; \n" \
"\n" \
"	if (i < R64 / 4)\n" \
"	{\n" \
"		const size_t k_4 = blk * R64 / 4 + i;\n" \
"		const uint4 u = T_4[i];\n" \
"		const uint u01 = addmod_d(u.s0, u.s1), u23 = addmod_d(u.s2, u.s3), u0123 = addmod_d(u01, u23);\n" \
"		tj[5 * (16 * s) + 4 * (4 * s) + k_4] = u23; tj[5 * (16 * s) + 5 * (4 * s) + k_4] = u0123; T_2[i] = u0123;\n" \
"	}\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	if (i < R64 / 16)\n" \
"	{\n" \
"		const size_t k_16 = blk * R64 / 16 + i;\n" \
"		const uint4 u = T2_4[i];\n" \
"		const uint u01 = addmod_d(u.s0, u.s1), u23 = addmod_d(u.s2, u.s3), u0123 = addmod_d(u01, u23);\n" \
"		tj[5 * (16 * s) + 5 * (4 * s) + 4 * s + k_16] = u23; tj[5 * (16 * s) + 5 * (4 * s) + 5 * s + k_16] = u0123;\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(R64, 1, 1)))\n" \
"void reduce_downsweep64(__global uint * restrict const t, const uint s, const uint j)\n" \
"{\n" \
"	__local uint4 T_4[R64 / 4 + R64 / 16];	// alignment\n" \
"	__local uint4 * const T2_4 = &T_4[R64 / 4];\n" \
"	__local uint * const T = (__local uint *)T_4;\n" \
"	__local uint * const T_2 = (__local uint *)T2_4;\n" \
"\n" \
"	const size_t i = get_local_id(0), blk = get_group_id(0), k = get_global_id(0);	// blk * R64 + i;\n" \
"	__global uint * const tj = &t[j];\n" \
"\n" \
"	if (i < R64 / 16)\n" \
"	{\n" \
"		const size_t k_16 = blk * R64 / 16 + i;\n" \
"		__global const uint4 * const tj16_4 = (__global uint4 *)&tj[5 * (16 * s) + 5 * (4 * s)];\n" \
"		const uint u2 = tj[5 * (16 * s) + 5 * (4 * s) + 4 * s + k_16], u0 = tj[5 * (16 * s) + 5 * (4 * s) + 5 * s + k_16], u02 = addmod_d(u0, u2);\n" \
"		const uint4 u13 = tj16_4[k_16];\n" \
"		const uint u012 = addmod_d(u02, u13.s1), u03 = addmod_d(u0, u13.s3);\n" \
"		T2_4[i] = (uint4)(u012, u02, u03, u0);\n" \
"	}\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	if (i < R64 / 4)\n" \
"	{\n" \
"		const size_t k_4 = blk * R64 / 4 + i;\n" \
"		__global const uint4 * const tj4_4 = (__global uint4 *)&tj[5 * (16 * s)];\n" \
"		const uint u2 = tj[5 * (16 * s) + 4 * (4 * s) + k_4], u0 = T_2[i], u02 = addmod_d(u0, u2);\n" \
"		const uint4 u13 = tj4_4[k_4];\n" \
"		const uint u012 = addmod_d(u02, u13.s1), u03 = addmod_d(u0, u13.s3);\n" \
"		T_4[i] = (uint4)(u012, u02, u03, u0);\n" \
"	}\n" \
"\n" \
"	__global uint4 * const tj1_4 = (__global uint4 *)&tj[0];\n" \
"	const uint u2 = tj[4 * (16 * s) + k];\n" \
"	const uint4 u13 = tj1_4[k];\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	const uint u0 = T[i], u02 = addmod_d(u0, u2);\n" \
"	const uint u012 = addmod_d(u02, u13.s1), u03 = addmod_d(u0, u13.s3);\n" \
"	tj1_4[k] = (uint4)(u012, u02, u03, u0);\n" \
"}\n" \
"\n" \
"inline void _reduce_upsweep4i(__local uint * restrict const T, __global const uint * restrict const t, const uint s, const size_t k)\n" \
"{\n" \
"	__global const uint4 * const ti = (__global const uint4 *)&t[4 * k];\n" \
"	__local uint * const To = &T[k];\n" \
"\n" \
"	const uint4 u = ti[0];\n" \
"	const uint u01 = addmod_d(u.s0, u.s1), u23 = addmod_d(u.s2, u.s3), u0123 = addmod_d(u01, u23);\n" \
"	To[0] = u23; To[s] = u0123;\n" \
"}\n" \
"\n" \
"inline void _reduce_upsweep4(__local uint * restrict const T, const uint s, const size_t k)\n" \
"{\n" \
"	__local const uint * const Ti = &T[0 * s + 4 * k];\n" \
"	__local uint * const To = &T[4 * s + 1 * k];\n" \
"\n" \
"	const uint u0 = Ti[0], u1 = Ti[1], u2 = Ti[2], u3 = Ti[3];\n" \
"	const uint u01 = addmod_d(u0, u1), u23 = addmod_d(u2, u3), u0123 = addmod_d(u01, u23);\n" \
"	To[0] = u23; To[s] = u0123;\n" \
"}\n" \
"\n" \
"inline void _reduce_downsweep4(__local uint * restrict const T, const uint s, const size_t k)\n" \
"{\n" \
"	__local const uint * const Ti = &T[4 * s + 1 * k];\n" \
"	__local uint * const To = &T[0 * s + 4 * k];\n" \
"\n" \
"	const uint u2 = Ti[0], u0 = Ti[s], u02 = addmod_d(u0, u2);\n" \
"	const uint u1 = To[1], u3 = To[3];\n" \
"	const uint u012 = addmod_d(u02, u1), u03 = addmod_d(u0, u3);\n" \
"	To[0] = u012; To[1] = u02; To[2] = u03; To[3] = u0;\n" \
"}\n" \
"\n" \
"inline void _reduce_downsweep4o(__global uint * restrict const t, __local const uint * restrict const T, const uint s, const size_t k)\n" \
"{\n" \
"	__local const uint * const Ti = &T[k];\n" \
"	__global uint4 * const to = (__global uint4 *)&t[4 * k];\n" \
"\n" \
"	const uint u2 = Ti[0], u0 = Ti[s], u02 = addmod_d(u0, u2);\n" \
"	const uint4 u13 = to[0];\n" \
"	const uint u012 = addmod_d(u02, u13.s1), u03 = addmod_d(u0, u13.s3);\n" \
"	to[0] = (uint4)(u012, u02, u03, u0);\n" \
"}\n" \
"\n" \
"inline void _reduce_topsweep2(__global uint * restrict const t, __local uint * restrict const T)\n" \
"{\n" \
"	const uint u0 = T[0], u1 = T[1];\n" \
"	const uint u01 = addmod_d(u0, u1);\n" \
"	t[0] = u01;\n" \
"	T[0] = u1; T[1] = 0;\n" \
"}\n" \
"\n" \
"inline void _reduce_topsweep4(__global uint * restrict const t, __local uint * restrict const T)\n" \
"{\n" \
"	const uint u0 = T[0], u1 = T[1], u2 = T[2], u3 = T[3];\n" \
"	const uint u01 = addmod_d(u0, u1), u23 = addmod_d(u2, u3);\n" \
"	const uint u123 = addmod_d(u1, u23), u0123 = addmod_d(u01, u23);\n" \
"	t[0] = u0123;\n" \
"	T[0] = u123; T[1] = u23; T[2] = u3; T[3] = 0;\n" \
"}\n" \
"\n" \
"#define	S32		(32 / 4)\n" \
"__kernel __attribute__((reqd_work_group_size(S32, 1, 1)))\n" \
"void reduce_topsweep32(__global uint * restrict const t, const uint j)\n" \
"{\n" \
"	__local uint T[64];	// 20\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"\n" \
"	_reduce_upsweep4i(T, &t[j], S32, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S32 / 4) _reduce_upsweep4(&T[S32], S32 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i == 0) _reduce_topsweep2(t, &T[S32 + 5 * (S32 / 4)]);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S32 / 4) _reduce_downsweep4(&T[S32], S32 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	_reduce_downsweep4o(&t[j], T, S32, i);\n" \
"}\n" \
"\n" \
"#define	S64		(64 / 4)\n" \
"__kernel __attribute__((reqd_work_group_size(S64, 1, 1)))\n" \
"void reduce_topsweep64(__global uint * restrict const t, const uint j)\n" \
"{\n" \
"	__local uint T[64];	// 40\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"\n" \
"	_reduce_upsweep4i(T, &t[j], S64, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S64 / 4) _reduce_upsweep4(&T[S64], S64 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i == 0) _reduce_topsweep4(t, &T[S64 + 5 * (S64 / 4)]);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S64 / 4) _reduce_downsweep4(&T[S64], S64 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	_reduce_downsweep4o(&t[j], T, S64, i);\n" \
"}\n" \
"\n" \
"#define	S128	(128 / 4)\n" \
"__kernel __attribute__((reqd_work_group_size(S128, 1, 1)))\n" \
"void reduce_topsweep128(__global uint * restrict const t, const uint j)\n" \
"{\n" \
"	__local uint T[128];	// 82\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"\n" \
"	_reduce_upsweep4i(T, &t[j], S128, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S128 / 4) _reduce_upsweep4(&T[S128], S128 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S128 / 16) _reduce_upsweep4(&T[S128 + 5 * (S128 / 4)], S128 / 16, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i == 0) _reduce_topsweep2(t, &T[S128 + 5 * (S128 / 4) + 5 * (S128 / 16)]);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S128 / 16) _reduce_downsweep4(&T[S128 + 5 * (S128 / 4)], S128 / 16, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S128 / 4) _reduce_downsweep4(&T[S128], S128 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	_reduce_downsweep4o(&t[j], T, S128, i);\n" \
"}\n" \
"\n" \
"#define	S256	(256 / 4)\n" \
"__kernel __attribute__((reqd_work_group_size(S256, 1, 1)))\n" \
"void reduce_topsweep256(__global uint * restrict const t, const uint j)\n" \
"{\n" \
"	__local uint T[256];	// 168\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"\n" \
"	_reduce_upsweep4i(T, &t[j], S256, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S256 / 4) _reduce_upsweep4(&T[S256], S256 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S256 / 16) _reduce_upsweep4(&T[S256 + 5 * (S256 / 4)], S256 / 16, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i == 0) _reduce_topsweep4(t, &T[S256 + 5 * (S256 / 4) + 5 * (S256 / 16)]);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S256 / 16) _reduce_downsweep4(&T[S256 + 5 * (S256 / 4)], S256 / 16, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S256 / 4) _reduce_downsweep4(&T[S256], S256 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	_reduce_downsweep4o(&t[j], T, S256, i);\n" \
"}\n" \
"\n" \
"#define	S512	(512 / 4)\n" \
"__kernel __attribute__((reqd_work_group_size(S512, 1, 1)))\n" \
"void reduce_topsweep512(__global uint * restrict const t, const uint j)\n" \
"{\n" \
"	__local uint T[512];	// 340\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"\n" \
"	_reduce_upsweep4i(T, &t[j], S512, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S512 / 4) _reduce_upsweep4(&T[S512], S512 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S512 / 16) _reduce_upsweep4(&T[S512 + 5 * (S512 / 4)], S512 / 16, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S512 / 64) _reduce_upsweep4(&T[S512 + 5 * (S512 / 4) + 5 * (S512 / 16)], S512 / 64, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i == 0) _reduce_topsweep2(t, &T[S512 + 5 * (S512 / 4) + 5 * (S512 / 16) + 5 * (S512 / 64)]);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S512 / 64) _reduce_downsweep4(&T[S512 + 5 * (S512 / 4) + 5 * (S512 / 16)], S512 / 64, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S512 / 16) _reduce_downsweep4(&T[S512 + 5 * (S512 / 4)], S512 / 16, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S512 / 4) _reduce_downsweep4(&T[S512], S512 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	_reduce_downsweep4o(&t[j], T, S512, i);\n" \
"}\n" \
"\n" \
"#define	S1024	(1024 / 4)\n" \
"__kernel __attribute__((reqd_work_group_size(S1024, 1, 1)))\n" \
"void reduce_topsweep1024(__global uint * restrict const t, const uint j)\n" \
"{\n" \
"	__local uint T[1024];	// 680\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"\n" \
"	_reduce_upsweep4i(T, &t[j], S1024, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S1024 / 4) _reduce_upsweep4(&T[S1024], S1024 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S1024 / 16) _reduce_upsweep4(&T[S1024 + 5 * (S1024 / 4)], S1024 / 16, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S1024 / 64) _reduce_upsweep4(&T[S1024 + 5 * (S1024 / 4) + 5 * (S1024 / 16)], S1024 / 64, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i == 0) _reduce_topsweep4(t, &T[S1024 + 5 * (S1024 / 4) + 5 * (S1024 / 16) + 5 * (S1024 / 64)]);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S1024 / 64) _reduce_downsweep4(&T[S1024 + 5 * (S1024 / 4) + 5 * (S1024 / 16)], S1024 / 64, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S1024 / 16) _reduce_downsweep4(&T[S1024 + 5 * (S1024 / 4)], S1024 / 16, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S1024 / 4) _reduce_downsweep4(&T[S1024], S1024 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	_reduce_downsweep4o(&t[j], T, S1024, i);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void reduce_i(__global const uint2 * restrict const x, __global uint * restrict const y, __global uint * restrict const t,\n" \
"	__global const uint * restrict const bp)\n" \
"{\n" \
"	const size_t k = get_global_id(0);\n" \
"\n" \
"	const uint xs = ((x[pconst_e + k].s0 >> pconst_s) | (x[pconst_e + k + 1].s0 << (digit_bit - pconst_s))) & digit_mask;\n" \
"	const uint u = rem_d(xs * (ulong)(bp[k]));\n" \
"\n" \
"	y[k] = xs;\n" \
"	t[k + 4] = u;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void reduce_o(__global uint2 * restrict const x, __global const uint * restrict const y, __global const uint * restrict const t,\n" \
"	__global const uint * restrict const ibp)\n" \
"{\n" \
"	const size_t k = get_global_id(0);\n" \
"\n" \
"	const uint tk = t[k + 4];\n" \
"	//const uint rbk_prev = (k != pconst_size / 2 - 1) ? tk : 0;	// NVidia compiler generates a conditionnal branch instruction then the code must be written with a mask\n" \
"	const uint mask = (k != pconst_size / 2 - 1) ? (uint)(-1) : 0;\n" \
"	const uint rbk_prev = tk & mask;\n" \
"	const uint r_prev = rem_d(rbk_prev * (ulong)(ibp[k]));\n" \
"\n" \
"	const ulong q = ((ulong)(r_prev) << digit_bit) | y[k];\n" \
"\n" \
"	const uint q_d = mul_hi((uint)(q >> pconst_d_shift), pconst_d_inv);	// d < 2^29\n" \
"	const uint r = (uint)(q) - q_d * pconst_d;\n" \
"	const uint c = (r >= pconst_d) ? 1 : 0;\n" \
"\n" \
"	x[k] = (uint2)((k > pconst_e) ? 0 : x[k].s0, q_d + c);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void reduce_f(__global uint2 * restrict const x, __global const uint * restrict const t)\n" \
"{\n" \
"	const uint rs = x[pconst_e].s0 & ((1u << pconst_s) - 1);\n" \
"	ulong l = ((ulong)(t[0]) << pconst_s) | rs;		// rds < 2^(29 + digit_bit - 1)\n" \
"\n" \
"	x[pconst_e].s0 = (uint)(l) & digit_mask;\n" \
"	l >>= digit_bit;\n" \
"\n" \
"	for (size_t k = pconst_e + 1; l != 0; ++k)\n" \
"	{\n" \
"		x[k].s0 = (uint)(l) & digit_mask;\n" \
"		l >>= digit_bit;\n" \
"	}\n" \
"}\n" \
"\n" \
"inline void _reduce_x(__global uint2 * restrict const x, __global int * const err)\n" \
"{\n" \
"	int c = 0;\n" \
"	for (size_t k = 0; k < pconst_size / 2; ++k)\n" \
"	{\n" \
"		const uint2 x_k = x[k];\n" \
"		c += x_k.s0 - x_k.s1;\n" \
"		x[k] = (uint2)((uint)(c) & digit_mask, 0);\n" \
"		c >>= digit_bit;\n" \
"	}\n" \
"\n" \
"	if (c != 0) atomic_or(err, c);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void reduce_x(__global uint2 * restrict const x, __global int * const err)\n" \
"{\n" \
"	_reduce_x(x, err);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void reduce_z(__global uint2 * restrict const x, __global int * const err)\n" \
"{\n" \
"	// s0 = x, s1 = k.2^n + 1\n" \
"	// if s0 >= s1 then s0 -= s1;\n" \
"\n" \
"	for (size_t i = 0; i < pconst_size / 2; ++i)\n" \
"	{\n" \
"		const uint2 x_k = x[pconst_size / 2 - 1 - i];\n" \
"		if (x_k.s0 < x_k.s1) return;\n" \
"		if (x_k.s0 > x_k.s1) break;\n" \
"	}\n" \
"\n" \
"	_reduce_x(x, err);\n" \
"}\n" \
"";
