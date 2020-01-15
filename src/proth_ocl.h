/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

static const char * const src_proth_ocl = \
"/*\n" \
"Copyright 2020, Yves Gallot\n" \
"\n" \
"proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\n" \
"Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n" \
"*/\n" \
"\n" \
"// __global is faster than __constant on NVIDIA GPU\n" \
"//#define __constmem	__constant\n" \
"#define __constmem	__global\n" \
"\n" \
"#define	digit_bit		21\n" \
"#define digit_mask		((1u << digit_bit) - 1)\n" \
"\n" \
"#define	P1			2130706433u		// 127 * 2^24 + 1 = 2^31 - 2^24 + 1\n" \
"#define	P2			2013265921u		//  15 * 2^27 + 1 = 2^31 - 2^27 + 1\n" \
"#define	P1_INV		2164392967u		// 2^62 / P1\n" \
"#define	P2_INV 		2290649223u		// 2^62 / P2\n" \
"#define	P1_I		2113994754u		// P1_PRIM_ROOT^((P1 - 1) / 4)\n" \
"#define	P2_I		1728404513u		// P2_PRIM_ROOT^((P2 - 1) / 4)\n" \
"#define	InvP2_P1	913159918u		// 1 / P2 mod P1\n" \
"#define	P1P2		(P1 * (ulong)(P2))\n" \
"\n" \
"#define CHUNK64		16\n" \
"#define BLK32		8\n" \
"#define BLK64		4\n" \
"#define BLK128		2\n" \
"#define BLK256		1\n" \
"#define	P2I_WGS		16\n" \
"#define	P2I_BLK		16\n" \
"#define	RED_BLK		4\n" \
"\n" \
"/*\n" \
"Barrett's product/reduction, where P is such that h (the number of iterations in the 'while loop') is 0 or 1.\n" \
"\n" \
"Let m < P^2 < 2^62, R = [2^62/P] and h = [m/P] - [[m/2^30] R / 2^32].\n" \
"\n" \
"We have h = ([m/P] - m/P) + m/2^62 (2^62/P - R) + R/2^32 (m/2^30 - [m/2^30]) + ([m/2^30] R / 2^32 - [[m/2^30] * R / 2^32]).\n" \
"Then -1 + 0 + 0 + 0 < h < 0 + (2^62/P - R) + R/2^32 + 1,\n" \
"0 <= h < 1 + (2^62/P - R) + R/2^32.\n" \
"\n" \
"P = 127 * 2^24 + 1 = 2130706433 => R = 2164392967, h < 1.56\n" \
"P =  63 * 2^25 + 1 = 2113929217 => R = 2181570688, h < 2.51 NOK\n" \
"P =  15 * 2^27 + 1 = 2013265921 => R = 2290649223, h < 1.93\n" \
"*/\n" \
"\n" \
"inline uint _addmod(const uint lhs, const uint rhs, const uint d)\n" \
"{\n" \
"	const uint r = lhs + rhs;\n" \
"	const uint t = (lhs >= d - rhs) ? d : 0;\n" \
"	return r - t;\n" \
"}\n" \
"\n" \
"inline uint _rem(const ulong q, const uint d, const uint d_inv, const int d_shift)\n" \
"{\n" \
"	const uint q_d = mul_hi((uint)(q >> d_shift), d_inv);\n" \
"	const uint r = (uint)(q) - q_d * d;\n" \
"	const uint t = (r >= d) ? d : 0;\n" \
"	return r - t;\n" \
"}\n" \
"\n" \
"inline uint _mulMod(const uint a, const uint b, const uint p, const uint p_inv)\n" \
"{\n" \
"	return _rem(a * (ulong)(b), p, p_inv, 30);\n" \
"}\n" \
"\n" \
"inline uint mulModP1(const uint a, const uint b) { return _mulMod(a, b, P1, P1_INV); }\n" \
"inline uint mulModP2(const uint a, const uint b) { return _mulMod(a, b, P2, P2_INV); }\n" \
"\n" \
"inline long getlong(const uint2 lhs)\n" \
"{\n" \
"	// Garner Algorithm\n" \
"	uint d = lhs.s0 - lhs.s1; const uint t = (lhs.s0 < lhs.s1) ? P1 : 0; d += t;	// mod P1\n" \
"	const uint u = mulModP1(d, InvP2_P1);		// P2 < P1\n" \
"	const ulong r = u * (ulong)(P2) + lhs.s1;\n" \
"	const ulong s = (r > P1P2 / 2) ? P1P2 : 0;\n" \
"	return (long)(r - s);\n" \
"}\n" \
"\n" \
"inline uint2 addmod(const uint2 lhs, const uint2 rhs)\n" \
"{\n" \
"	const uint2 r = lhs + rhs;\n" \
"	const uint2 t = (uint2)((lhs.s0 >= P1 - rhs.s0) ? P1 : 0, (lhs.s1 >= P2 - rhs.s1) ? P2 : 0);\n" \
"	return r - t;\n" \
"}\n" \
"\n" \
"inline uint2 submod(const uint2 lhs, const uint2 rhs)\n" \
"{\n" \
"	const uint2 r = lhs - rhs;\n" \
"	const uint2 t = (uint2)((lhs.s0 < rhs.s0) ? P1 : 0, (lhs.s1 < rhs.s1) ? P2 : 0);\n" \
"	return r + t;\n" \
"}\n" \
"\n" \
"inline uint2 mulmod(const uint2 lhs, const uint2 rhs)\n" \
"{\n" \
"	return (uint2)(mulModP1(lhs.s0, rhs.s0), mulModP2(lhs.s1, rhs.s1));\n" \
"}\n" \
"\n" \
"inline uint2 sqrmod(const uint2 lhs) { return mulmod(lhs, lhs); }\n" \
"\n" \
"inline uint2 mulI(const uint2 lhs) { return mulmod(lhs, (uint2)(P1_I, P2_I)); }\n" \
"\n" \
"inline void _sub_forward4i(const size_t ml, __local uint2 * restrict const X, const size_t mg, __global const uint2 * restrict const x, const uint2 r2, const uint4 r1ir1)\n" \
"{\n" \
"	const uint2 abi = x[0 * mg], abim = x[1 * mg];\n" \
"	const uint2 abi0 = (uint2)(abi.s0, abi.s0), abi1 = (uint2)(abi.s1, abi.s1);\n" \
"	const uint2 abim0 = (uint2)(abim.s0, abim.s0), abim1 = (uint2)(abim.s1, abim.s1);\n" \
"	const uint2 u0 = submod(abi0, abi1), u1 = submod(abim0, abim1), u3 = mulI(u1);\n" \
"	X[0 * ml] = addmod(u0, u1); X[1 * ml] = mulmod(submod(u0, u1), r2);\n" \
"	X[2 * ml] = mulmod(submod(u0, u3), r1ir1.s23); X[3 * ml] = mulmod(addmod(u0, u3), r1ir1.s01);\n" \
"}\n" \
"\n" \
"inline void _forward4i(const size_t ml, __local uint2 * restrict const X, const size_t mg, __global const uint2 * restrict const x, const uint2 r2, const uint4 r1ir1)\n" \
"{\n" \
"	const uint2 u0 = x[0 * mg], u2 = x[2 * mg], u1 = x[1 * mg], u3 = x[3 * mg];\n" \
"	const uint2 v0 = addmod(u0, u2), v2 = submod(u0, u2), v1 = addmod(u1, u3), v3 = mulI(submod(u3, u1));\n" \
"	X[0 * ml] = addmod(v0, v1); X[1 * ml] = mulmod(submod(v0, v1), r2);\n" \
"	X[2 * ml] = mulmod(addmod(v2, v3), r1ir1.s23); X[3 * ml] = mulmod(submod(v2, v3), r1ir1.s01);\n" \
"}\n" \
"\n" \
"inline void _forward4(const size_t m, __local uint2 * restrict const X, const uint2 r2, const uint4 r1ir1)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	const uint2 u0 = X[0 * m], u2 = X[2 * m], u1 = X[1 * m], u3 = X[3 * m];\n" \
"	const uint2 v0 = addmod(u0, u2), v2 = submod(u0, u2), v1 = addmod(u1, u3), v3 = mulI(submod(u3, u1));\n" \
"	X[0 * m] = addmod(v0, v1); X[1 * m] = mulmod(submod(v0, v1), r2);\n" \
"	X[2 * m] = mulmod(addmod(v2, v3), r1ir1.s23); X[3 * m] = mulmod(submod(v2, v3), r1ir1.s01);\n" \
"}\n" \
"\n" \
"inline void _forward4o(const size_t mg, __global uint2 * restrict const x, const size_t ml, __local const uint2 * restrict const X, const uint2 r2, const uint4 r1ir1)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	const uint2 u0 = X[0 * ml], u2 = X[2 * ml], u1 = X[1 * ml], u3 = X[3 * ml];\n" \
"	const uint2 v0 = addmod(u0, u2), v2 = submod(u0, u2), v1 = addmod(u1, u3), v3 = mulI(submod(u3, u1));\n" \
"	x[0 * mg] = addmod(v0, v1); x[1 * mg] = mulmod(submod(v0, v1), r2);\n" \
"	x[2 * mg] = mulmod(addmod(v2, v3), r1ir1.s23); x[3 * mg] = mulmod(submod(v2, v3), r1ir1.s01);\n" \
"}\n" \
"\n" \
"inline void _backward4i(const size_t ml, __local uint2 * restrict const X, const size_t mg, __global const uint2 * restrict const x, const uint2 ir2, const uint4 r1ir1)\n" \
"{\n" \
"	const uint2 v0 = x[0 * mg], v1 = mulmod(x[1 * mg], ir2), v2 = mulmod(x[2 * mg], r1ir1.s01), v3 = mulmod(x[3 * mg], r1ir1.s23);\n" \
"	const uint2 u0 = addmod(v0, v1), u2 = addmod(v2, v3), u1 = submod(v0, v1), u3 = mulI(submod(v2, v3));\n" \
"	X[0 * ml] = addmod(u0, u2); X[2 * ml] = submod(u0, u2); X[1 * ml] = addmod(u1, u3); X[3 * ml] = submod(u1, u3);\n" \
"}\n" \
"\n" \
"inline void _backward4(const size_t m, __local uint2 * restrict const X, const uint2 ir2, const uint4 r1ir1)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	const uint2 v0 = X[0 * m], v1 = mulmod(X[1 * m], ir2), v2 = mulmod(X[2 * m], r1ir1.s01), v3 = mulmod(X[3 * m], r1ir1.s23);\n" \
"	const uint2 u0 = addmod(v0, v1), u2 = addmod(v2, v3), u1 = submod(v0, v1), u3 = mulI(submod(v2, v3));\n" \
"	X[0 * m] = addmod(u0, u2); X[2 * m] = submod(u0, u2); X[1 * m] = addmod(u1, u3); X[3 * m] = submod(u1, u3);\n" \
"}\n" \
"\n" \
"inline void _backward4o(const size_t mg, __global uint2 * restrict const x, const size_t ml, __local const uint2 * restrict const X, const uint2 ir2, const uint4 r1ir1)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	const uint2 v0 = X[0 * ml], v1 = mulmod(X[1 * ml], ir2), v2 = mulmod(X[2 * ml], r1ir1.s01), v3 = mulmod(X[3 * ml], r1ir1.s23);\n" \
"	const uint2 u0 = addmod(v0, v1), u2 = addmod(v2, v3), u1 = submod(v0, v1), u3 = mulI(submod(v2, v3));\n" \
"	x[0 * mg] = addmod(u0, u2); x[2 * mg] = submod(u0, u2); x[1 * mg] = addmod(u1, u3); x[3 * mg] = submod(u1, u3);\n" \
"}\n" \
"\n" \
"inline void _square2(__local uint2 * restrict const X)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	// TODO X01 & X45\n" \
"	const uint2 u0 = X[0], u1 = X[1], u2 = X[2], u3 = X[3];\n" \
"	const uint2 v0 = addmod(u0, u1), v1 = submod(u0, u1), v2 = addmod(u2, u3), v3 = submod(u2, u3);\n" \
"	const uint2 s0 = sqrmod(v0), s1 = sqrmod(v1), s2 = sqrmod(v2), s3 = sqrmod(v3);\n" \
"	X[0] = addmod(s0, s1); X[1] = submod(s0, s1); X[2] = addmod(s2, s3); X[3] = submod(s2, s3);\n" \
"}\n" \
"\n" \
"inline void _square4(__local uint2 * restrict const X)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	const uint2 u0 = X[0], u2 = X[2], u1 = X[1], u3 = X[3];\n" \
"	const uint2 v0 = addmod(u0, u2), v2 = submod(u0, u2), v1 = addmod(u1, u3), v3 = mulI(submod(u3, u1));\n" \
"	const uint2 s0 = sqrmod(addmod(v0, v1)), s1 = sqrmod(submod(v0, v1)), s2 = sqrmod(addmod(v2, v3)), s3 = sqrmod(submod(v2, v3));\n" \
"	const uint2 t0 = addmod(s0, s1), t2 = addmod(s2, s3), t1 = submod(s0, s1), t3 = mulI(submod(s2, s3));\n" \
"	X[0] = addmod(t0, t2); X[2] = submod(t0, t2); X[1] = addmod(t1, t3); X[3] = submod(t1, t3);\n" \
"}\n" \
"\n" \
"\n" \
"__kernel  __attribute__((reqd_work_group_size(64 / 4 * CHUNK64, 1, 1)))\n" \
"void sub_ntt64(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2)\n" \
"{\n" \
"	__local uint2 X[64 * CHUNK64];\n" \
"\n" \
"	const size_t m = get_global_size(0) / 16;\n" \
"\n" \
"	const size_t local_id = get_local_id(0), chunk_idx = local_id % CHUNK64, threadIdx = local_id / CHUNK64, block_idx = get_group_id(0) * CHUNK64;	// threadIdx < 64 / 4\n" \
"\n" \
"	const size_t bl_i = block_idx | chunk_idx;\n" \
"\n" \
"	const size_t _i_16m = threadIdx;											// [+ 16, 32, 48] 0, 1, 2, 3, 4, ..., 15\n" \
"	const size_t _i_4m = ((4 * threadIdx) & ~(4 * 4 - 1)) | (threadIdx % 4);	// [+ 4, 8, 12] 0, 1, 2, 3, 16, 17, ...\n" \
"	const size_t _i_m = 4 * threadIdx;											// [+ 1, 2, 3] 0, 4, 8, 12, ...\n" \
"\n" \
"	const size_t i_16m = _i_16m * CHUNK64 | chunk_idx;\n" \
"	const size_t i_4m = _i_4m * CHUNK64 | chunk_idx;\n" \
"	const size_t i_m = _i_m * CHUNK64 | chunk_idx;\n" \
"\n" \
"	const size_t k_16m = _i_16m * m | bl_i;	// TODO shift m ?\n" \
"	const size_t k_4m = _i_4m * m | bl_i;\n" \
"	const size_t k_m = _i_m * m | bl_i;\n" \
"\n" \
"	const size_t j_16m = k_16m;	// & (16 * m - 1); We have _i_16m < 16 and bl_i < m\n" \
"	const size_t j_4m = k_4m & (4 * m - 1);\n" \
"	const size_t j_m = bl_i;	// k_m & (m - 1); We have bl_i < m\n" \
"\n" \
"	_sub_forward4i(16 * CHUNK64, &X[i_16m], 16 * m, &x[k_16m], r2[j_16m], r1ir1[j_16m]);\n" \
"	_forward4(4 * CHUNK64, &X[i_4m], r2[16 * m + j_4m], r1ir1[16 * m + j_4m]);\n" \
"	_forward4o(m, &x[k_m], CHUNK64, &X[i_m], r2[16 * m + 4 * m + j_m], r1ir1[16 * m + 4 * m + j_m]);\n" \
"}\n" \
"\n" \
"__kernel  __attribute__((reqd_work_group_size(64 / 4 * CHUNK64, 1, 1)))\n" \
"void ntt64(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)\n" \
"{\n" \
"	__local uint2 X[64 * CHUNK64];\n" \
"\n" \
"	const size_t local_id = get_local_id(0), chunk_idx = local_id % CHUNK64, threadIdx = local_id / CHUNK64, block_idx = get_group_id(0) * CHUNK64;	// threadIdx < 64 / 4\n" \
"\n" \
"	__global uint2 * const xo = &x[64 * (block_idx & ~(m - 1))];		// m-block offset\n" \
"	const size_t bl_i = (block_idx & (m - 1)) | chunk_idx;\n" \
"\n" \
"	const size_t _i_16m = threadIdx;											// [+ 16, 32, 48] 0, 1, 2, 3, 4, ..., 15\n" \
"	const size_t _i_4m = ((4 * threadIdx) & ~(4 * 4 - 1)) | (threadIdx % 4);	// [+ 4, 8, 12] 0, 1, 2, 3, 16, 17, ...\n" \
"	const size_t _i_m = 4 * threadIdx;											// [+ 1, 2, 3] 0, 4, 8, 12, ...\n" \
"\n" \
"	const size_t i_16m = _i_16m * CHUNK64 | chunk_idx;\n" \
"	const size_t i_4m = _i_4m * CHUNK64 | chunk_idx;\n" \
"	const size_t i_m = _i_m * CHUNK64 | chunk_idx;\n" \
"\n" \
"	const size_t k_16m = _i_16m * m | bl_i;	// TODO shift m ?\n" \
"	const size_t k_4m = _i_4m * m | bl_i;\n" \
"	const size_t k_m = _i_m * m | bl_i;\n" \
"\n" \
"	const size_t j_16m = k_16m;	// & (16 * m - 1); We have _i_16m < 16 and bl_i < m\n" \
"	const size_t j_4m = k_4m & (4 * m - 1);\n" \
"	const size_t j_m = bl_i;	// k_m & (m - 1); We have bl_i < m\n" \
"\n" \
"	_forward4i(16 * CHUNK64, &X[i_16m], 16 * m, &xo[k_16m], r2[rindex + j_16m], r1ir1[rindex + j_16m]);\n" \
"	_forward4(4 * CHUNK64, &X[i_4m], r2[rindex + 16 * m + j_4m], r1ir1[rindex + 16 * m + j_4m]);\n" \
"	_forward4o(m, &xo[k_m], CHUNK64, &X[i_m], r2[rindex + 16 * m + 4 * m + j_m], r1ir1[rindex + 16 * m + 4 * m + j_m]);\n" \
"}\n" \
"\n" \
"__kernel  __attribute__((reqd_work_group_size(64 / 4 * CHUNK64, 1, 1)))\n" \
"void intt64(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)\n" \
"{\n" \
"	__local uint2 X[64 * CHUNK64];\n" \
"\n" \
"	const size_t local_id = get_local_id(0), chunk_idx = local_id % CHUNK64, threadIdx = local_id / CHUNK64, block_idx = get_group_id(0) * CHUNK64;	// threadIdx < 64 / 4\n" \
"\n" \
"	__global uint2 * const xo = &x[64 * (block_idx & ~(m - 1))];		// m-block offset\n" \
"	const size_t bl_i = (block_idx & (m - 1)) | chunk_idx;\n" \
"\n" \
"	const size_t _i_16m = threadIdx;											// [+ 16, 32, 48] 0, 1, 2, 3, 4, ..., 15\n" \
"	const size_t _i_4m = ((4 * threadIdx) & ~(4 * 4 - 1)) | (threadIdx % 4);	// [+ 4, 8, 12] 0, 1, 2, 3, 16, 17, ...\n" \
"	const size_t _i_m = 4 * threadIdx;											// [+ 1, 2, 3] 0, 4, 8, 12, ...\n" \
"\n" \
"	const size_t i_16m = _i_16m * CHUNK64 | chunk_idx;\n" \
"	const size_t i_4m = _i_4m * CHUNK64 | chunk_idx;\n" \
"	const size_t i_m = _i_m * CHUNK64 | chunk_idx;\n" \
"\n" \
"	const size_t k_16m = _i_16m * m | bl_i;	// TODO shift m ?\n" \
"	const size_t k_4m = _i_4m * m | bl_i;\n" \
"	const size_t k_m = _i_m * m | bl_i;\n" \
"\n" \
"	const size_t j_16m = k_16m;	// & (16 * m - 1); We have _i_16m < 16 and bl_i < m\n" \
"	const size_t j_4m = k_4m & (4 * m - 1);\n" \
"	const size_t j_m = bl_i;	// k_m & (m - 1); We have bl_i < m\n" \
"\n" \
"	_backward4i(CHUNK64, &X[i_m], m, &xo[k_m], ir2[rindex + 16 * m + 4 * m + j_m], r1ir1[rindex + 16 * m + 4 * m + j_m]);\n" \
"	_backward4(4 * CHUNK64, &X[i_4m], ir2[rindex + 16 * m + j_4m], r1ir1[rindex + 16 * m + j_4m]);\n" \
"	_backward4o(16 * m, &xo[k_16m], 16 * CHUNK64, &X[i_16m], ir2[rindex + j_16m], r1ir1[rindex + j_16m]);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(32 / 4 * BLK32, 1, 1)))\n" \
"void square32(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)\n" \
"{\n" \
"	__local uint2 X[32 * BLK32];\n" \
"\n" \
"	// copy mem first ?\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"	const size_t i_8 = i % 8, i8 = ((4 * i) & (size_t)~(4 * 8 - 1)) | i_8, j8 = i_8 + 2;\n" \
"	const size_t i_2 = i % 2, i2 = ((4 * i) & (size_t)~(4 * 2 - 1)) | i_2, j2 = i_2;\n" \
"	const size_t k8 = get_group_id(0) * 32 * BLK32 | i8;\n" \
"\n" \
"	_forward4i(8, &X[i8], 8, &x[k8], r2ir2[j8].s01, r1ir1[j8]);\n" \
"	_forward4(2, &X[i2], r2ir2[j2].s01, r1ir1[j2]);\n" \
"	_square2(&X[4 * i]);\n" \
"	_backward4(2, &X[i2], r2ir2[j2].s23, r1ir1[j2]);\n" \
"	_backward4o(8, &x[k8], 8, &X[i8], r2ir2[j8].s23, r1ir1[j8]);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(64 / 4 * BLK64, 1, 1)))\n" \
"void square64(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)\n" \
"{\n" \
"	__local uint2 X[64 * BLK64];\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"	const size_t i_16 = i % 16, i16 = ((4 * i) & (size_t)~(4 * 16 - 1)) | i_16, j16 = i_16 + 4;\n" \
"	const size_t i_4 = i % 4, i4 = ((4 * i) & (size_t)~(4 * 4 - 1)) | i_4, j4 = i_4;\n" \
"	const size_t k16 = get_group_id(0) * 64 * BLK64 | i16;\n" \
"\n" \
"	_forward4i(16, &X[i16], 16, &x[k16], r2ir2[j16].s01, r1ir1[j16]);\n" \
"	_forward4(4, &X[i4], r2ir2[j4].s01, r1ir1[j4]);\n" \
"	_square4(&X[4 * i]);\n" \
"	_backward4(4, &X[i4], r2ir2[j4].s23, r1ir1[j4]);\n" \
"	_backward4o(16, &x[k16], 16, &X[i16], r2ir2[j16].s23, r1ir1[j16]);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(128 / 4 * BLK128, 1, 1)))\n" \
"void square128(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)\n" \
"{\n" \
"	__local uint2 X[128 * BLK128];\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"	const size_t i_32 = i % 32, i32 = ((4 * i) & (size_t)~(4 * 32 - 1)) | i_32, j32 = i_32 + 2 + 8;\n" \
"	const size_t i_8 = i % 8, i8 = ((4 * i) & (size_t)~(4 * 8 - 1)) | i_8, j8 = i_8 + 2;\n" \
"	const size_t i_2 = i % 2, i2 = ((4 * i) & (size_t)~(4 * 2 - 1)) | i_2, j2 = i_2;\n" \
"	const size_t k32 = get_group_id(0) * 128 * BLK128 | i32;\n" \
"\n" \
"	_forward4i(32, &X[i32], 32, &x[k32], r2ir2[j32].s01, r1ir1[j32]);\n" \
"	_forward4(8, &X[i8], r2ir2[j8].s01, r1ir1[j8]);\n" \
"	_forward4(2, &X[i2], r2ir2[j2].s01, r1ir1[j2]);\n" \
"	_square2(&X[4 * i]);\n" \
"	_backward4(2, &X[i2], r2ir2[j2].s23, r1ir1[j2]);\n" \
"	_backward4(8, &X[i8], r2ir2[j8].s23, r1ir1[j8]);\n" \
"	_backward4o(32, &x[k32], 32, &X[i32], r2ir2[j32].s23, r1ir1[j32]);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(256 / 4 * BLK256, 1, 1)))\n" \
"void square256(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)\n" \
"{\n" \
"	__local uint2 X[256 * BLK256];\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"	const size_t i_64 = i % 64, i64 = ((4 * i) & (size_t)~(4 * 64 - 1)) | i_64, j64 = i_64 + 4 + 16;\n" \
"	const size_t i_16 = i % 16, i16 = ((4 * i) & (size_t)~(4 * 16 - 1)) | i_16, j16 = i_16 + 4;\n" \
"	const size_t i_4 = i % 4, i4 = ((4 * i) & (size_t)~(4 * 4 - 1)) | i_4, j4 = i_4;\n" \
"	const size_t k64 = get_group_id(0) * 256 * BLK256 | i64;\n" \
"\n" \
"	_forward4i(64, &X[i64], 64, &x[k64], r2ir2[j64].s01, r1ir1[j64]);\n" \
"	_forward4(16, &X[i16], r2ir2[j16].s01, r1ir1[j16]);\n" \
"	_forward4(4, &X[i4], r2ir2[j4].s01, r1ir1[j4]);\n" \
"	_square4(&X[4 * i]);\n" \
"	_backward4(4, &X[i4], r2ir2[j4].s23, r1ir1[j4]);\n" \
"	_backward4(16, &X[i16], r2ir2[j16].s23, r1ir1[j16]);\n" \
"	_backward4o(64, &x[k64], 64, &X[i64], r2ir2[j64].s23, r1ir1[j64]);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(512 / 4, 1, 1)))\n" \
"void square512(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)\n" \
"{\n" \
"	__local uint2 X[512];\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"	const size_t i128 = i, j128 = i + 2 + 8 + 32;\n" \
"	const size_t i_32 = i % 32, i32 = ((4 * i) & (size_t)~(4 * 32 - 1)) | i_32, j32 = i_32 + 2 + 8;\n" \
"	const size_t i_8 = i % 8, i8 = ((4 * i) & (size_t)~(4 * 8 - 1)) | i_8, j8 = i_8 + 2;\n" \
"	const size_t i_2 = i % 2, i2 = ((4 * i) & (size_t)~(4 * 2 - 1)) | i_2, j2 = i_2;\n" \
"	const size_t k128 = get_group_id(0) * 512 | i128;\n" \
"\n" \
"	_forward4i(128, &X[i128], 128, &x[k128], r2ir2[j128].s01, r1ir1[j128]);\n" \
"	_forward4(32, &X[i32], r2ir2[j32].s01, r1ir1[j32]);\n" \
"	_forward4(8, &X[i8], r2ir2[j8].s01, r1ir1[j8]);\n" \
"	_forward4(2, &X[i2], r2ir2[j2].s01, r1ir1[j2]);\n" \
"	_square2(&X[4 * i]);\n" \
"	_backward4(2, &X[i2], r2ir2[j2].s23, r1ir1[j2]);\n" \
"	_backward4(8, &X[i8], r2ir2[j8].s23, r1ir1[j8]);\n" \
"	_backward4(32, &X[i32], r2ir2[j32].s23, r1ir1[j32]);\n" \
"	_backward4o(128, &x[k128], 128, &X[i128], r2ir2[j128].s23, r1ir1[j128]);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 4, 1, 1)))\n" \
"void square1024(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)\n" \
"{\n" \
"	__local uint2 X[1024];\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"	const size_t i256 = i, j256 = i + 4 + 16 + 64;\n" \
"	const size_t i_64 = i % 64, i64 = ((4 * i) & (size_t)~(4 * 64 - 1)) | i_64, j64 = i_64 + 4 + 16;\n" \
"	const size_t i_16 = i % 16, i16 = ((4 * i) & (size_t)~(4 * 16 - 1)) | i_16, j16 = i_16 + 4;\n" \
"	const size_t i_4 = i % 4, i4 = ((4 * i) & (size_t)~(4 * 4 - 1)) | i_4, j4 = i_4;\n" \
"	const size_t k256 = get_group_id(0) * 1024 | i256;\n" \
"\n" \
"	_forward4i(256, &X[i256], 256, &x[k256], r2ir2[j256].s01, r1ir1[j256]);\n" \
"	_forward4(64, &X[i64], r2ir2[j64].s01, r1ir1[j64]);\n" \
"	_forward4(16, &X[i16], r2ir2[j16].s01, r1ir1[j16]);\n" \
"	_forward4(4, &X[i4], r2ir2[j4].s01, r1ir1[j4]);\n" \
"	_square4(&X[4 * i]);\n" \
"	_backward4(4, &X[i4], r2ir2[j4].s23, r1ir1[j4]);\n" \
"	_backward4(16, &X[i16], r2ir2[j16].s23, r1ir1[j16]);\n" \
"	_backward4(64, &X[i64], r2ir2[j64].s23, r1ir1[j64]);\n" \
"	_backward4o(256, &x[k256], 256, &X[i256], r2ir2[j256].s23, r1ir1[j256]);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(P2I_WGS, 1, 1)))\n" \
"void poly2int0(__global uint2 * restrict const x, __global long * restrict const cr, const uint2 norm)\n" \
"{\n" \
"	__local long L[P2I_WGS * P2I_BLK];\n" \
"	__local uint X[P2I_WGS * P2I_BLK];\n" \
"\n" \
"	const size_t i = get_local_id(0), blk = get_group_id(0);\n" \
"	const size_t kc = (get_global_id(0) + 1) & (get_global_size(0) - 1);\n" \
"\n" \
"	__global uint2 * const xo = &x[P2I_WGS * P2I_BLK * blk];\n" \
"\n" \
"	for (size_t j = 0; j < P2I_BLK; ++j)\n" \
"	{\n" \
"		const size_t k = P2I_WGS * j + i;\n" \
"		L[P2I_WGS * (k % P2I_BLK) + (k / P2I_BLK)] = getlong(mulmod(xo[k], norm));	// -n/2 . (B-1)^2 <= l <= n/2 . (B-1)^2\n" \
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
"		//xo[k] = (uint2)(X[P2I_WGS * (k % P2I_BLK) + (k / P2I_BLK)], 0);	faster?\n" \
"		xo[k].s0 = X[P2I_WGS * (k % P2I_BLK) + (k / P2I_BLK)];\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void poly2int1(__global uint2 * restrict const x, __global const long * restrict const cr, __global int * const err)\n" \
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
"#pragma unroll\n" \
"	for (size_t j = 1; j < P2I_BLK; ++j)\n" \
"	{\n" \
"		f += xi[j].s0;\n" \
"		xi[j].s0 = (uint)(f) & digit_mask;\n" \
"		f >>= digit_bit;					// f = -1, 0 or 1\n" \
"		if (f == 0) break;\n" \
"	}\n" \
"\n" \
"	if (f != 0) atomic_or(err, 1);\n" \
"}\n" \
"\n" \
"#define	R64		(RED_BLK * 64 / 4)\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(R64, 1, 1)))\n" \
"void reduce_upsweep64(__global uint * restrict const t, const uint d, const uint s, const uint j)\n" \
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
"	const uint u01 = _addmod(u.s0, u.s1, d), u23 = _addmod(u.s2, u.s3, d), u0123 = _addmod(u01, u23, d);\n" \
"	tj[4 * (16 * s) + k] = u23; tj[5 * (16 * s) + k] = u0123; T[i] = u0123;\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	if (i < R64 / 4)\n" \
"	{\n" \
"		const size_t k_4 = blk * R64 / 4 + i;\n" \
"		const uint4 u = T_4[i];\n" \
"		const uint u01 = _addmod(u.s0, u.s1, d), u23 = _addmod(u.s2, u.s3, d), u0123 = _addmod(u01, u23, d);\n" \
"		tj[5 * (16 * s) + 4 * (4 * s) + k_4] = u23; tj[5 * (16 * s) + 5 * (4 * s) + k_4] = u0123; T_2[i] = u0123;\n" \
"	}\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	if (i < R64 / 16)\n" \
"	{\n" \
"		const size_t k_16 = blk * R64 / 16 + i;\n" \
"		const uint4 u = T2_4[i];\n" \
"		const uint u01 = _addmod(u.s0, u.s1, d), u23 = _addmod(u.s2, u.s3, d), u0123 = _addmod(u01, u23, d);\n" \
"		tj[5 * (16 * s) + 5 * (4 * s) + 4 * s + k_16] = u23; tj[5 * (16 * s) + 5 * (4 * s) + 5 * s + k_16] = u0123;\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(R64, 1, 1)))\n" \
"void reduce_downsweep64(__global uint * restrict const t, const uint d, const uint s, const uint j)\n" \
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
"		const uint u2 = tj[5 * (16 * s) + 5 * (4 * s) + 4 * s + k_16], u0 = tj[5 * (16 * s) + 5 * (4 * s) + 5 * s + k_16], u02 = _addmod(u0, u2, d);\n" \
"		const uint4 u13 = tj16_4[k_16];\n" \
"		const uint u012 = _addmod(u02, u13.s1, d), u03 = _addmod(u0, u13.s3, d);\n" \
"		T2_4[i] = (uint4)(u012, u02, u03, u0);\n" \
"	}\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	if (i < R64 / 4)\n" \
"	{\n" \
"		const size_t k_4 = blk * R64 / 4 + i;\n" \
"		__global const uint4 * const tj4_4 = (__global uint4 *)&tj[5 * (16 * s)];\n" \
"		const uint u2 = tj[5 * (16 * s) + 4 * (4 * s) + k_4], u0 = T_2[i], u02 = _addmod(u0, u2, d);\n" \
"		const uint4 u13 = tj4_4[k_4];\n" \
"		const uint u012 = _addmod(u02, u13.s1, d), u03 = _addmod(u0, u13.s3, d);\n" \
"		T_4[i] = (uint4)(u012, u02, u03, u0);\n" \
"	}\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	__global uint4 * const tj1_4 = (__global const uint4 *)&tj[0];\n" \
"	const uint u2 = tj[4 * (16 * s) + k], u0 = T[i], u02 = _addmod(u0, u2, d);\n" \
"	const uint4 u13 = tj1_4[k];\n" \
"	const uint u012 = _addmod(u02, u13.s1, d), u03 = _addmod(u0, u13.s3, d);\n" \
"	tj1_4[k] = (uint4)(u012, u02, u03, u0);\n" \
"}\n" \
"\n" \
"inline void _reduce_upsweep4i(__local uint * restrict const T, __global const uint * restrict const t, const uint d, const uint s, const size_t k)\n" \
"{\n" \
"	__global const uint4 * const ti = (__global const uint4 *)&t[4 * k];\n" \
"	__local uint * const To = &T[k];\n" \
"\n" \
"	const uint4 u = ti[0];\n" \
"	const uint u01 = _addmod(u.s0, u.s1, d), u23 = _addmod(u.s2, u.s3, d), u0123 = _addmod(u01, u23, d);\n" \
"	To[0] = u23; To[s] = u0123;\n" \
"}\n" \
"\n" \
"inline void _reduce_upsweep4(__local uint * restrict const T, const uint d, const uint s, const size_t k)\n" \
"{\n" \
"	__local const uint * const Ti = &T[0 * s + 4 * k];\n" \
"	__local uint * const To = &T[4 * s + 1 * k];\n" \
"\n" \
"	const uint u0 = Ti[0], u1 = Ti[1], u2 = Ti[2], u3 = Ti[3];\n" \
"	const uint u01 = _addmod(u0, u1, d), u23 = _addmod(u2, u3, d), u0123 = _addmod(u01, u23, d);\n" \
"	To[0] = u23; To[s] = u0123;\n" \
"}\n" \
"\n" \
"inline void _reduce_downsweep4(__local uint * restrict const T, const uint d, const uint s, const size_t k)\n" \
"{\n" \
"	__local const uint * const Ti = &T[4 * s + 1 * k];\n" \
"	__local uint * const To = &T[0 * s + 4 * k];\n" \
"\n" \
"	const uint u2 = Ti[0], u0 = Ti[s], u02 = _addmod(u0, u2, d);\n" \
"	const uint u1 = To[1], u3 = To[3];\n" \
"	const uint u012 = _addmod(u02, u1, d), u03 = _addmod(u0, u3, d);\n" \
"	To[0] = u012; To[1] = u02; To[2] = u03; To[3] = u0;\n" \
"}\n" \
"\n" \
"inline void _reduce_downsweep4o(__global uint * restrict const t, __local const uint * restrict const T, const uint d, const uint s, const size_t k)\n" \
"{\n" \
"	__local const uint * const Ti = &T[k];\n" \
"	__global uint4 * const to = (__global uint4 *)&t[4 * k];\n" \
"\n" \
"	const uint u2 = Ti[0], u0 = Ti[s], u02 = _addmod(u0, u2, d);\n" \
"	const uint4 u13 = to[0];\n" \
"	const uint u012 = _addmod(u02, u13.s1, d), u03 = _addmod(u0, u13.s3, d);\n" \
"	to[0] = (uint4)(u012, u02, u03, u0);\n" \
"}\n" \
"\n" \
"inline void _reduce_topsweep2(__global uint * restrict const t, __local uint * restrict const T, const uint d)\n" \
"{\n" \
"	const uint u0 = T[0], u1 = T[1];\n" \
"	const uint u01 = _addmod(u0, u1, d);\n" \
"	t[0] = u01;\n" \
"	T[0] = u1; T[1] = 0;\n" \
"}\n" \
"\n" \
"inline void _reduce_topsweep4(__global uint * restrict const t, __local uint * restrict const T, const uint d)\n" \
"{\n" \
"	const uint u0 = T[0], u1 = T[1], u2 = T[2], u3 = T[3];\n" \
"	const uint u01 = _addmod(u0, u1, d), u23 = _addmod(u2, u3, d);\n" \
"	const uint u123 = _addmod(u1, u23, d), u0123 = _addmod(u01, u23, d);\n" \
"	t[0] = u0123;\n" \
"	T[0] = u123; T[1] = u23; T[2] = u3; T[3] = 0;\n" \
"}\n" \
"\n" \
"#define	S32		(32 / 4)\n" \
"__kernel __attribute__((reqd_work_group_size(S32, 1, 1)))\n" \
"void reduce_topsweep32(__global uint * restrict const t, const uint d, const uint j)\n" \
"{\n" \
"	__local uint T[32];	// 20\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"\n" \
"	_reduce_upsweep4i(T, &t[j], d, S32, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S32 / 4) _reduce_upsweep4(&T[S32], d, S32 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i == 0) _reduce_topsweep2(t, &T[S32 + 5 * (S32 / 4)], d);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S32 / 4) _reduce_downsweep4(&T[S32], d, S32 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	_reduce_downsweep4o(&t[j], T, d, S32, i);\n" \
"}\n" \
"\n" \
"#define	S64		(64 / 4)\n" \
"__kernel __attribute__((reqd_work_group_size(S64, 1, 1)))\n" \
"void reduce_topsweep64(__global uint * restrict const t, const uint d, const uint j)\n" \
"{\n" \
"	__local uint T[64];	// 40\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"\n" \
"	_reduce_upsweep4i(T, &t[j], d, S64, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S64 / 4) _reduce_upsweep4(&T[S64], d, S64 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i == 0) _reduce_topsweep4(t, &T[S64 + 5 * (S64 / 4)], d);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S64 / 4) _reduce_downsweep4(&T[S64], d, S64 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	_reduce_downsweep4o(&t[j], T, d, S64, i);\n" \
"}\n" \
"\n" \
"#define	S128	(128 / 4)\n" \
"__kernel __attribute__((reqd_work_group_size(S128, 1, 1)))\n" \
"void reduce_topsweep128(__global uint * restrict const t, const uint d, const uint j)\n" \
"{\n" \
"	__local uint T[128];	// 82\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"\n" \
"	_reduce_upsweep4i(T, &t[j], d, S128, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S128 / 4) _reduce_upsweep4(&T[S128], d, S128 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S128 / 16) _reduce_upsweep4(&T[S128 + 5 * (S128 / 4)], d, S128 / 16, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i == 0) _reduce_topsweep2(t, &T[S128 + 5 * (S128 / 4) + 5 * (S128 / 16)], d);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S128 / 16) _reduce_downsweep4(&T[S128 + 5 * (S128 / 4)], d, S128 / 16, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S128 / 4) _reduce_downsweep4(&T[S128], d, S128 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	_reduce_downsweep4o(&t[j], T, d, S128, i);\n" \
"}\n" \
"\n" \
"#define	S256	(256 / 4)\n" \
"__kernel __attribute__((reqd_work_group_size(S256, 1, 1)))\n" \
"void reduce_topsweep256(__global uint * restrict const t, const uint d, const uint j)\n" \
"{\n" \
"	__local uint T[256];	// 168\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"\n" \
"	_reduce_upsweep4i(T, &t[j], d, S256, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S256 / 4) _reduce_upsweep4(&T[S256], d, S256 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S256 / 16) _reduce_upsweep4(&T[S256 + 5 * (S256 / 4)], d, S256 / 16, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i == 0) _reduce_topsweep4(t, &T[S256 + 5 * (S256 / 4) + 5 * (S256 / 16)], d);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S256 / 16) _reduce_downsweep4(&T[S256 + 5 * (S256 / 4)], d, S256 / 16, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S256 / 4) _reduce_downsweep4(&T[S256], d, S256 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	_reduce_downsweep4o(&t[j], T, d, S256, i);\n" \
"}\n" \
"\n" \
"#define	S512	(512 / 4)\n" \
"__kernel __attribute__((reqd_work_group_size(S512, 1, 1)))\n" \
"void reduce_topsweep512(__global uint * restrict const t, const uint d, const uint j)\n" \
"{\n" \
"	__local uint T[512];	// 340\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"\n" \
"	_reduce_upsweep4i(T, &t[j], d, S512, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S512 / 4) _reduce_upsweep4(&T[S512], d, S512 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S512 / 16) _reduce_upsweep4(&T[S512 + 5 * (S512 / 4)], d, S512 / 16, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S512 / 64) _reduce_upsweep4(&T[S512 + 5 * (S512 / 4) + 5 * (S512 / 16)], d, S512 / 64, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i == 0) _reduce_topsweep2(t, &T[S512 + 5 * (S512 / 4) + 5 * (S512 / 16) + 5 * (S512 / 64)], d);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S512 / 64) _reduce_downsweep4(&T[S512 + 5 * (S512 / 4) + 5 * (S512 / 16)], d, S512 / 64, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S512 / 16) _reduce_downsweep4(&T[S512 + 5 * (S512 / 4)], d, S512 / 16, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S512 / 4) _reduce_downsweep4(&T[S512], d, S512 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	_reduce_downsweep4o(&t[j], T, d, S512, i);\n" \
"}\n" \
"\n" \
"#define	S1024	(1024 / 4)\n" \
"__kernel __attribute__((reqd_work_group_size(S1024, 1, 1)))\n" \
"void reduce_topsweep1024(__global uint * restrict const t, const uint d, const uint j)\n" \
"{\n" \
"	__local uint T[1024];	// 680\n" \
"\n" \
"	const size_t i = get_local_id(0);\n" \
"\n" \
"	_reduce_upsweep4i(T, &t[j], d, S1024, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S1024 / 4) _reduce_upsweep4(&T[S1024], d, S1024 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S1024 / 16) _reduce_upsweep4(&T[S1024 + 5 * (S1024 / 4)], d, S1024 / 16, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S1024 / 64) _reduce_upsweep4(&T[S1024 + 5 * (S1024 / 4) + 5 * (S1024 / 16)], d, S1024 / 64, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i == 0) _reduce_topsweep4(t, &T[S1024 + 5 * (S1024 / 4) + 5 * (S1024 / 16) + 5 * (S1024 / 64)], d);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S1024 / 64) _reduce_downsweep4(&T[S1024 + 5 * (S1024 / 4) + 5 * (S1024 / 16)], d, S1024 / 64, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S1024 / 16) _reduce_downsweep4(&T[S1024 + 5 * (S1024 / 4)], d, S1024 / 16, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	if (i < S1024 / 4) _reduce_downsweep4(&T[S1024], d, S1024 / 4, i);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	_reduce_downsweep4o(&t[j], T, d, S1024, i);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void reduce_i(__global const uint2 * restrict const x, __global uint * restrict const y, __global uint * restrict const t,\n" \
"	__global const uint * restrict const bp, const uint3 e_d_d_inv, const int2 s_d_shift)\n" \
"{\n" \
"	const size_t k = get_global_id(0);\n" \
"\n" \
"	const uint e = e_d_d_inv.s0, d = e_d_d_inv.s1, d_inv = e_d_d_inv.s2;\n" \
"	const int s = s_d_shift.s0, d_shift = s_d_shift.s1;\n" \
"\n" \
"	const uint xs = ((x[e + k + 1].s0 << (digit_bit - s)) | (x[e + k].s0 >> s)) & digit_mask;\n" \
"	const uint u = _rem(xs * (ulong)(bp[k]), d, d_inv, d_shift);\n" \
"\n" \
"	y[k] = xs;\n" \
"	t[k + 4] = u;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void reduce_o(__global uint2 * restrict const x, __global const uint * restrict const y, __global const uint * restrict const t,\n" \
"	__global const uint * restrict const ibp, const uint3 e_d_d_inv, const int2 s_d_shift)\n" \
"{\n" \
"	const size_t n = get_global_size(0), k = get_global_id(0);\n" \
"\n" \
"	const uint e = e_d_d_inv.s0, d = e_d_d_inv.s1, d_inv = e_d_d_inv.s2;\n" \
"	const int s = s_d_shift.s0, d_shift = s_d_shift.s1;\n" \
"\n" \
"	const uint tk = t[k + 4];\n" \
"	//const uint rbk_prev = (k + 1 != n) ? tk : 0;	// NVidia compiler generates a conditionnal branch instruction then the code must be written with a mask\n" \
"	const uint mask = (k + 1 != n) ? (uint)(-1) : 0;\n" \
"	const uint rbk_prev = tk & mask;\n" \
"	const uint r_prev = _rem(rbk_prev * (ulong)(ibp[k]), d, d_inv, d_shift);\n" \
"\n" \
"	const ulong q = ((ulong)(r_prev) << digit_bit) | y[k];\n" \
"\n" \
"	const uint q_d = mul_hi((uint)(q >> d_shift), d_inv);	// d < 2^29\n" \
"	const uint r = (uint)(q) - q_d * d;\n" \
"	const uint c = (r >= d) ? 1 : 0;\n" \
"\n" \
"	const uint2 x_k = x[k];\n" \
"	x[k] = (uint2)((k > e) ? 0 : x_k.s0, q_d + c);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void reduce_f(__global uint2 * restrict const x, __global const uint * restrict const t, const uint n, const uint e, const int s)\n" \
"{\n" \
"	const uint rs = x[e].s0 & ((1u << s) - 1);\n" \
"	ulong l = ((ulong)(t[0]) << s) | rs;		// rds < 2^(29 + digit_bit - 1)\n" \
"\n" \
" 	x[e].s0 = (uint)(l) & digit_mask;\n" \
"	l >>= digit_bit;\n" \
"\n" \
"	for (size_t i = e + 1; l != 0; ++i)\n" \
"	{\n" \
"		x[i].s0 = (uint)(l) & digit_mask;\n" \
"		l >>= digit_bit;\n" \
"	}\n" \
"}\n" \
;