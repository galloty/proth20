/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

__kernel __attribute__((reqd_work_group_size(256 / 4 * 8, 1, 1)))
void sub_ntt256_8(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2)
{
	SUB_NTT256(8);
}

__kernel __attribute__((reqd_work_group_size(256 / 4 * 8, 1, 1)))
void ntt256_8(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)
{
	NTT256(8);
}

__kernel __attribute__((reqd_work_group_size(256 / 4 * 8, 1, 1)))
void intt256_8(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)
{
	INTT256(8);
}


__kernel __attribute__((reqd_work_group_size(1024 / 4 * 2, 1, 1)))
void sub_ntt1024_2(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2)
{
	SUB_NTT1024(2);
}

__kernel __attribute__((reqd_work_group_size(1024 / 4 * 2, 1, 1)))
void ntt1024_2(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)
{
	NTT1024(2);
}

__kernel __attribute__((reqd_work_group_size(1024 / 4 * 2, 1, 1)))
void intt1024_2(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)
{
	INTT1024(2);
}


__kernel __attribute__((reqd_work_group_size(2048 / 4, 1, 1)))
void square2048(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1, __constmem const uint4 * restrict const ir1,
	__constmem const uint4 * restrict const r2, __constmem const uint4 * restrict const ir2)
{
	__local uint2 X[2048];	// 16k

	const size_t i = get_local_id(0);
	const size_t i512 = i, j512 = i + 2 + 8 + 32 + 128;
	const size_t i_128 = i % 128, i128 = ((4 * i) & (size_t)~(4 * 128 - 1)) | i_128, j128 = i_128 + 2 + 8 + 32;
	const size_t i_32 = i % 32, i32 = ((4 * i) & (size_t)~(4 * 32 - 1)) | i_32, j32 = i_32 + 2 + 8;
	const size_t i_8 = i % 8, i8 = ((4 * i) & (size_t)~(4 * 8 - 1)) | i_8, j8 = i_8 + 2;
	const size_t i_2 = i % 2, _i2 = ((4 * i) & (size_t)~(4 * 2 - 1)), i2 = _i2 | i_2, j2 = i_2, i_0 = _i2 | (2 * i_2);
	const size_t k512 = get_group_id(0) * 2048 | i512;

	const uint4 r1_512 = r1[j512], ir1_512 = ir1[j512];
	_forward4pi(512, &X[i512], 512, &x[k512], r2[j512], r1_512, ir1_512);
	const uint4 r1_128 = r1[j128], ir1_128 = ir1[j128];
	_forward4p(128, &X[i128], r2[j128], r1_128, ir1_128);
	const uint4 r1_32 = r1[j32], ir1_32 = ir1[j32];
	_forward4p(32, &X[i32], r2[j32], r1_32, ir1_32);
	const uint4 r1_8 = r1[j8], ir1_8 = ir1[j8];
	_forward4p(8, &X[i8], r2[j8], r1_8, ir1_8);
	const uint4 r1_2 = r1[j2], ir1_2 = ir1[j2];
	_forward4p(2, &X[i2], r2[j2], r1_2, ir1_2);
	_square2(&X[i_0]);
	_backward4p(2, &X[i2], ir2[j2], r1_2, ir1_2);
	_backward4p(8, &X[i8], ir2[j8], r1_8, ir1_8);
	_backward4p(32, &X[i32], ir2[j32], r1_32, ir1_32);
	_backward4p(128, &X[i128], ir2[j128], r1_128, ir1_128);
	_backward4po(512, &x[k512], 512, &X[i512], ir2[j512], r1_512, ir1_512);
}
