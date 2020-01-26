/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

static const char * const src_ocl_misc = \
"/*\n" \
"Copyright 2020, Yves Gallot\n" \
"\n" \
"proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\n" \
"Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n" \
"*/\n" \
"\n" \
"__kernel\n" \
"void set_positive(__global uint2 * restrict const x, const uint n, const uint e, const ulong ds)\n" \
"{\n" \
"	// x.s0 = R, x.s1 = Y\n" \
"	// if R < Y then add k.2^n + 1 to R.\n" \
"\n" \
"	for (size_t i = 0; i < n; ++i)\n" \
"	{\n" \
"		const size_t j = n - 1 - i;\n" \
"		const uint2 x_j = x[j];\n" \
"		if (x_j.s0 > x_j.s1) return;\n" \
"		if (x_j.s0 < x_j.s1)\n" \
"		{\n" \
"			// R += 1\n" \
"			uint c = 1;\n" \
"			for (size_t k = 0; c != 0; ++k)\n" \
"			{\n" \
"				c += x[k].s0;\n" \
"				x[k].s0 = c & digit_mask;\n" \
"				c >>= digit_bit;\n" \
"			}\n" \
"\n" \
"			// R += k.2^n\n" \
"			ulong l = ds;\n" \
"			for (size_t k = e; l != 0; ++k)\n" \
"			{\n" \
"				l += x[k].s0;\n" \
"				x[k].s0 = (uint)(l) & digit_mask;\n" \
"				l >>= digit_bit;\n" \
"			}\n" \
"\n" \
"			return;\n" \
"		}\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void add1(__global uint2 * restrict const x, const uint e, const ulong ds)\n" \
"{\n" \
"	uint c = 1;\n" \
"	for (size_t k = 0; c != 0; ++k)\n" \
"	{\n" \
"		c += x[k].s0;\n" \
"		x[k].s0 = (uint)(c) & digit_mask;\n" \
"		c >>= digit_bit;\n" \
"	}\n" \
"\n" \
"	// s1: 0 => k.2^n + 1 for reduce_z step\n" \
"\n" \
"	x[0].s1 = 1;\n" \
"\n" \
"	ulong l = ds;\n" \
"	for (size_t k = e; l != 0; ++k)\n" \
"	{\n" \
"		x[k].s1 = (uint)(l) & digit_mask;\n" \
"		l >>= digit_bit;\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void swap(__global uint2 * restrict const x, __global uint2 * restrict const y)\n" \
"{\n" \
"	const size_t k = get_global_id(0);\n" \
"	const uint2 x_k = x[k], y_k = y[k];\n" \
"	x[k] = y_k; y[k] = x_k;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void copy(__global uint2 * restrict const x, __global const uint2 * restrict const y)\n" \
"{\n" \
"	const size_t k = get_global_id(0);\n" \
"	x[k] = y[k];\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void compare(__global const uint2 * restrict const x, __global const uint2 * restrict const y, __global int * const err)\n" \
"{\n" \
"	const size_t k = get_global_id(0);\n" \
"	const uint2 x_k = x[k], y_k = y[k];\n" \
"	if ((x_k.s0 != y_k.s0) || (x_k.s1 != y_k.s1)) atomic_or(err, 1);\n" \
"}\n" \
;
