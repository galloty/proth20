/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

__kernel
void set_positive(__global uint2 * restrict const x)
{
	// x.s0 = R, x.s1 = Y
	// if R < Y then add k.2^n + 1 to R.

	for (size_t i = 0; i < pconst_size / 2; ++i)
	{
		const size_t j = pconst_size / 2 - 1 - i;
		const uint2 x_j = x[j];
		if (x_j.s0 > x_j.s1) return;
		if (x_j.s0 < x_j.s1)
		{
			// R += 1
			uint c = 1;
			for (size_t k = 0; c != 0; ++k)
			{
				c += x[k].s0;
				x[k].s0 = c & digit_mask;
				c >>= digit_bit;
			}

			// R += k.2^n
			ulong l = (ulong)(pconst_d) << pconst_s;
			for (size_t k = pconst_e; l != 0; ++k)
			{
				l += x[k].s0;
				x[k].s0 = (uint)(l) & digit_mask;
				l >>= digit_bit;
			}

			return;
		}
	}
}

__kernel
void add1(__global uint2 * restrict const x)
{
	// s0: += 1
	// s1: 0 => k.2^n + 1 for reduce_z step

	uint c = x[0].s0 + 1;
	x[0] = (uint2)(c & digit_mask, 1);
	c >>= digit_bit;

	for (size_t k = 1; c != 0; ++k)
	{
		c += x[k].s0;
		x[k].s0 = c & digit_mask;
		c >>= digit_bit;
	}

	ulong l = (ulong)(pconst_d) << pconst_s;
	for (size_t k = pconst_e; l != 0; ++k)
	{
		x[k].s1 = (uint)(l) & digit_mask;
		l >>= digit_bit;
	}
}

__kernel
void swap(__global uint2 * restrict const x, __global uint2 * restrict const y)
{
	const size_t k = get_global_id(0);
	const uint2 x_k = x[k], y_k = y[k];
	x[k] = y_k; y[k] = x_k;
}

__kernel
void copy(__global uint2 * restrict const x, __global const uint2 * restrict const y)
{
	const size_t k = get_global_id(0);
	x[k] = y[k];
}

__kernel
void compare(__global const uint2 * restrict const x, __global const uint2 * restrict const y, __global int * const err)
{
	const size_t k = get_global_id(0);
	const uint2 x_k = x[k], y_k = y[k];
	if ((x_k.s0 != y_k.s0) || (x_k.s1 != y_k.s1)) atomic_or(err, 1);
}
