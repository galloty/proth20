/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#define CHUNK1024	4

__kernel __attribute__((reqd_work_group_size(1024 / 4 * CHUNK1024, 1, 1)))
void sub_ntt1024(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2)
{
	SETVAR(1024, CHUNK1024);	// __local 32k
	SETVAR_SUB_NTT(1024);

	SUB_FORWARD4i(256, CHUNK1024);
	FORWARD4(64, CHUNK1024, 256 * m);
	FORWARD4(16, CHUNK1024, 256 * m + 64 * m);
	FORWARD4(4, CHUNK1024, 256 * m + 64 * m + 16 * m);
	FORWARD4o(CHUNK1024, 256 * m + 64 * m + 16 * m + 4 * m);
}

__kernel __attribute__((reqd_work_group_size(1024 / 4 * CHUNK1024, 1, 1)))
void ntt1024(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)
{
	SETVAR(1024, CHUNK1024);	// __local 32k
	SETVAR_NTT(1024);

	FORWARD4i(256, CHUNK1024, rindex);
	FORWARD4(64, CHUNK1024, rindex + 256 * m);
	FORWARD4(16, CHUNK1024, rindex + 256 * m + 64 * m);
	FORWARD4(4, CHUNK1024, rindex + 256 * m + 64 * m + 16 * m);
	FORWARD4o(CHUNK1024, rindex + 256 * m + 64 * m + 16 * m + 4 * m);
}

__kernel __attribute__((reqd_work_group_size(1024 / 4 * CHUNK1024, 1, 1)))
void intt1024(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)
{
	SETVAR(1024, CHUNK1024);	// __local 32k
	SETVAR_NTT(1024);

	BACKWARD4i(CHUNK1024, rindex + 256 * m + 64 * m + 16 * m + 4 * m);
	BACKWARD4(4, CHUNK1024, rindex + 256 * m + 64 * m + 16 * m);
	BACKWARD4(16, CHUNK1024, rindex + 256 * m + 64 * m);
	BACKWARD4(64, CHUNK1024, rindex + 256 * m);
	BACKWARD4o(256, CHUNK1024, rindex);
}


__kernel __attribute__((reqd_work_group_size(4096 / 4, 1, 1)))
void square4096(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)
{
	__local uint2 X[4096];	// 32k

	const size_t i = get_local_id(0);
	const size_t i1024 = i, j1024 = i + 4 + 16 + 64 + 256;
	const size_t i_256 = i % 256, i256 = ((4 * i) & (size_t)~(4 * 256 - 1)) | i_256, j256 = i_256 + 4 + 16 + 64;
	const size_t i_64 = i % 64, i64 = ((4 * i) & (size_t)~(4 * 64 - 1)) | i_64, j64 = i_64 + 4 + 16;
	const size_t i_16 = i % 16, i16 = ((4 * i) & (size_t)~(4 * 16 - 1)) | i_16, j16 = i_16 + 4;
	const size_t i_4 = i % 4, i4 = ((4 * i) & (size_t)~(4 * 4 - 1)) | i_4, j4 = i_4;
	const size_t k1024 = get_group_id(0) * 4096 | i1024;

	_forward4i(1024, &X[i1024], 1024, &x[k1024], r2ir2[j1024].s01, r1ir1[j1024]);
	_forward4(256, &X[i256], r2ir2[j256].s01, r1ir1[j256]);
	_forward4(64, &X[i64], r2ir2[j64].s01, r1ir1[j64]);
	_forward4(16, &X[i16], r2ir2[j16].s01, r1ir1[j16]);
	_forward4(4, &X[i4], r2ir2[j4].s01, r1ir1[j4]);
	_square4(&X[4 * i]);
	_backward4(4, &X[i4], r2ir2[j4].s23, r1ir1[j4]);
	_backward4(16, &X[i16], r2ir2[j16].s23, r1ir1[j16]);
	_backward4(64, &X[i64], r2ir2[j64].s23, r1ir1[j64]);
	_backward4(256, &X[i256], r2ir2[j256].s23, r1ir1[j256]);
	_backward4o(1024, &x[k1024], 1024, &X[i1024], r2ir2[j1024].s23, r1ir1[j1024]);
}