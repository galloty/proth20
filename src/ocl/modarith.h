/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

static const char * const src_ocl_modarith = \
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
"#define digit_mask		((1u << digit_bit) - 1)\n" \
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
"#define	P1			2130706433u		// 127 * 2^24 + 1 = 2^31 - 2^24 + 1\n" \
"#define	P2			2013265921u		//  15 * 2^27 + 1 = 2^31 - 2^27 + 1\n" \
"#define	P1_INV		2164392967u		// 2^62 / P1\n" \
"#define	P2_INV 		2290649223u		// 2^62 / P2\n" \
"#define	P1_I		2113994754u		// P1_PRIM_ROOT = 3,  P1_PRIM_ROOT^((P1 - 1) / 4)\n" \
"#define	P2_I		1728404513u		// P2_PRIM_ROOT = 31, P2_PRIM_ROOT^((P2 - 1) / 4)\n" \
"#define	P1_Ip		4261280761u		// (P1_I * 2^32) / P1\n" \
"#define	P2_Ip		3687262959u		// (P2_I * 2^32) / P2\n" \
"#define	InvP2_P1	913159918u		// 1 / P2 mod P1\n" \
"#define	InvP2_P1p	1840700306u		// (InvP2_P1 * 2^32) / P1\n" \
"#define	P1P2		(P1 * (ulong)(P2))\n" \
"\n" \
"inline uint _rem(const ulong q, const uint d, const uint d_inv, const int d_shift)\n" \
"{\n" \
"	const uint q_d = mul_hi((uint)(q >> d_shift), d_inv);\n" \
"	const uint r = (uint)(q) - q_d * d;\n" \
"	const uint t = (r >= d) ? d : 0;\n" \
"	return r - t;\n" \
"}\n" \
"\n" \
"inline uint addmod_d(const uint lhs, const uint rhs)\n" \
"{\n" \
"	const uint r = lhs + rhs;\n" \
"	const uint t = (r >= pconst_d) ? pconst_d : 0;\n" \
"	return r - t;\n" \
"}\n" \
"\n" \
"inline uint rem_d(const ulong q)\n" \
"{\n" \
"	return _rem(q, pconst_d, pconst_d_inv, pconst_d_shift);\n" \
"}\n" \
"\n" \
"inline uint _mulmodP1(const uint a, const uint b) { return _rem(a * (ulong)(b), P1, P1_INV, 30); }\n" \
"inline uint _mulmodP2(const uint a, const uint b) { return _rem(a * (ulong)(b), P2, P2_INV, 30); }\n" \
"\n" \
"inline uint _mulmodp(const uint lhs, const uint p, const uint c, const uint cp)\n" \
"{\n" \
"	// Shoup's modular multiplication: Faster arithmetic for number-theoretic transforms, David Harvey, J.Symb.Comp. 60 (2014) 113-119\n" \
"	const uint r = lhs * c - mul_hi(lhs, cp) * p;\n" \
"	const uint t = (r >= p) ? p : 0;\n" \
"	return r - t;\n" \
"}\n" \
"\n" \
"inline long getlong(const uint2 lhs)\n" \
"{\n" \
"	// Garner Algorithm\n" \
"	uint d = lhs.s0 - lhs.s1; const uint t = (lhs.s0 < lhs.s1) ? P1 : 0; d += t;	// mod P1\n" \
"	const uint u = _mulmodp(d, P1, InvP2_P1, InvP2_P1p);		// P2 < P1\n" \
"	const ulong r = u * (ulong)(P2) + lhs.s1;\n" \
"	const ulong s = (r > P1P2 / 2) ? P1P2 : 0;\n" \
"	return (long)(r - s);\n" \
"}\n" \
"\n" \
"inline uint2 addmod(const uint2 lhs, const uint2 rhs)\n" \
"{\n" \
"	const uint2 r = lhs + rhs;\n" \
"	const uint2 t = (uint2)((lhs.s0 >= P1 - rhs.s0) ? P1 : 0, (lhs.s1 >= P2 - rhs.s1) ? P2 : 0);\n" \
"	// const uint2 t = (uint2)((r.s0 >= P1) ? P1 : 0, (r.s1 >= P2) ? P2 : 0);\n" \
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
"	return (uint2)(_mulmodP1(lhs.s0, rhs.s0), _mulmodP2(lhs.s1, rhs.s1));\n" \
"}\n" \
"\n" \
"inline uint2 mulmodp(const uint2 lhs, const uint4 rhs)\n" \
"{\n" \
"	return (uint2)(_mulmodp(lhs.s0, P1, rhs.s0, rhs.s2), _mulmodp(lhs.s1, P2, rhs.s1, rhs.s3));\n" \
"}\n" \
"\n" \
"inline uint2 sqrmod(const uint2 lhs) { return mulmod(lhs, lhs); }\n" \
"\n" \
"inline uint2 mulI(const uint2 lhs)\n" \
"{\n" \
"	return (uint2)(_mulmodp(lhs.s0, P1, P1_I, P1_Ip), _mulmodp(lhs.s1, P2, P2_I, P2_Ip));\n" \
"}\n" \
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
"inline void _forward4pi(const size_t ml, __local uint2 * restrict const X, const size_t mg, __global const uint2 * restrict const x,\n" \
"	const uint4 r2, const uint4 r1, const uint4 ir1)\n" \
"{\n" \
"	const uint2 u0 = x[0 * mg], u2 = x[2 * mg], u1 = x[1 * mg], u3 = x[3 * mg];\n" \
"	const uint2 v0 = addmod(u0, u2), v2 = submod(u0, u2), v1 = addmod(u1, u3), v3 = mulI(submod(u3, u1));\n" \
"	X[0 * ml] = addmod(v0, v1); X[1 * ml] = mulmodp(submod(v0, v1), r2);\n" \
"	X[2 * ml] = mulmodp(addmod(v2, v3), ir1); X[3 * ml] = mulmodp(submod(v2, v3), r1);\n" \
"}\n" \
"\n" \
"inline void _forward4p(const size_t m, __local uint2 * restrict const X,\n" \
"	const uint4 r2, const uint4 r1, const uint4 ir1)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	const uint2 u0 = X[0 * m], u2 = X[2 * m], u1 = X[1 * m], u3 = X[3 * m];\n" \
"	const uint2 v0 = addmod(u0, u2), v2 = submod(u0, u2), v1 = addmod(u1, u3), v3 = mulI(submod(u3, u1));\n" \
"	X[0 * m] = addmod(v0, v1); X[1 * m] = mulmodp(submod(v0, v1), r2);\n" \
"	X[2 * m] = mulmodp(addmod(v2, v3), ir1); X[3 * m] = mulmodp(submod(v2, v3), r1);\n" \
"}\n" \
"\n" \
"inline void _forward4po(const size_t mg, __global uint2 * restrict const x, const size_t ml, __local const uint2 * restrict const X,\n" \
"	const uint4 r2, const uint4 r1, const uint4 ir1)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	const uint2 u0 = X[0 * ml], u2 = X[2 * ml], u1 = X[1 * ml], u3 = X[3 * ml];\n" \
"	const uint2 v0 = addmod(u0, u2), v2 = submod(u0, u2), v1 = addmod(u1, u3), v3 = mulI(submod(u3, u1));\n" \
"	x[0 * mg] = addmod(v0, v1); x[1 * mg] = mulmodp(submod(v0, v1), r2);\n" \
"	x[2 * mg] = mulmodp(addmod(v2, v3), ir1); x[3 * mg] = mulmodp(submod(v2, v3), r1);\n" \
"}\n" \
"\n" \
"inline void _backward4pi(const size_t ml, __local uint2 * restrict const X, const size_t mg, __global const uint2 * restrict const x,\n" \
"	const uint4 ir2, const uint4 r1, const uint4 ir1)\n" \
"{\n" \
"	const uint2 v0 = x[0 * mg], v1 = mulmodp(x[1 * mg], ir2), v2 = mulmodp(x[2 * mg], r1), v3 = mulmodp(x[3 * mg], ir1);\n" \
"	const uint2 u0 = addmod(v0, v1), u2 = addmod(v2, v3), u1 = submod(v0, v1), u3 = mulI(submod(v2, v3));\n" \
"	X[0 * ml] = addmod(u0, u2); X[2 * ml] = submod(u0, u2); X[1 * ml] = addmod(u1, u3); X[3 * ml] = submod(u1, u3);\n" \
"}\n" \
"\n" \
"inline void _backward4p(const size_t m, __local uint2 * restrict const X,\n" \
"	const uint4 ir2, const uint4 r1, const uint4 ir1)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	const uint2 v0 = X[0 * m], v1 = mulmodp(X[1 * m], ir2), v2 = mulmodp(X[2 * m], r1), v3 = mulmodp(X[3 * m], ir1);\n" \
"	const uint2 u0 = addmod(v0, v1), u2 = addmod(v2, v3), u1 = submod(v0, v1), u3 = mulI(submod(v2, v3));\n" \
"	X[0 * m] = addmod(u0, u2); X[2 * m] = submod(u0, u2); X[1 * m] = addmod(u1, u3); X[3 * m] = submod(u1, u3);\n" \
"}\n" \
"\n" \
"inline void _backward4po(const size_t mg, __global uint2 * restrict const x, const size_t ml, __local const uint2 * restrict const X,\n" \
"	const uint4 ir2, const uint4 r1, const uint4 ir1)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	const uint2 v0 = X[0 * ml], v1 = mulmodp(X[1 * ml], ir2), v2 = mulmodp(X[2 * ml], r1), v3 = mulmodp(X[3 * ml], ir1);\n" \
"	const uint2 u0 = addmod(v0, v1), u2 = addmod(v2, v3), u1 = submod(v0, v1), u3 = mulI(submod(v2, v3));\n" \
"	x[0 * mg] = addmod(u0, u2); x[2 * mg] = submod(u0, u2); x[1 * mg] = addmod(u1, u3); x[3 * mg] = submod(u1, u3);\n" \
"}\n" \
"\n" \
"inline void _square2(__local uint2 * restrict const X)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	const uint2 u0 = X[0], u1 = X[1], u4 = X[4], u5 = X[5];\n" \
"	const uint2 v0 = addmod(u0, u1), v1 = submod(u0, u1), v4 = addmod(u4, u5), v5 = submod(u4, u5);\n" \
"	const uint2 s0 = sqrmod(v0), s1 = sqrmod(v1), s4 = sqrmod(v4), s5 = sqrmod(v5);\n" \
"	X[0] = addmod(s0, s1); X[1] = submod(s0, s1); X[4] = addmod(s4, s5); X[5] = submod(s4, s5);\n" \
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
"";
