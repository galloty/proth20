/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#define BLK8		32
#define BLK16		16
#define BLK32		8
#define BLK64		4
#define BLK128		2
#define BLK256		1

__kernel
void mul2(__global uint2 * restrict const x, __global const uint2 * restrict const y)
{
	const size_t k = get_global_id(0);

	const size_t i = 4 * k;

	const uint2 ux0 = x[i + 0], ux1 = x[i + 1], ux2 = x[i + 2], ux3 = x[i + 3];
	const uint2 vx0 = addmod(ux0, ux1), vx1 = submod(ux0, ux1), vx2 = addmod(ux2, ux3), vx3 = submod(ux2, ux3);
	const uint2 uy0 = y[i + 0], uy1 = y[i + 1], uy2 = y[i + 2], uy3 = y[i + 3];
	const uint2 vy0 = addmod(uy0, uy1), vy1 = submod(uy0, uy1), vy2 = addmod(uy2, uy3), vy3 = submod(uy2, uy3);
	const uint2 s0 = mulmod(vx0, vy0), s1 = mulmod(vx1, vy1), s2 = mulmod(vx2, vy2), s3 = mulmod(vx3, vy3);
	x[i + 0] = addmod(s0, s1); x[i + 1] = submod(s0, s1); x[i + 2] = addmod(s2, s3); x[i + 3] = submod(s2, s3);
}

__kernel
void mul4(__global uint2 * restrict const x, __global const uint2 * restrict const y)
{
	const size_t k = get_global_id(0);

	const size_t i = 4 * k;

	const uint2 ux0 = x[i + 0], ux2 = x[i + 2], ux1 = x[i + 1], ux3 = x[i + 3];
	const uint2 vx0 = addmod(ux0, ux2), vx2 = submod(ux0, ux2), vx1 = addmod(ux1, ux3), vx3 = mulI(submod(ux3, ux1));
	const uint2 uy0 = y[i + 0], uy2 = y[i + 2], uy1 = y[i + 1], uy3 = y[i + 3];
	const uint2 vy0 = addmod(uy0, uy2), vy2 = submod(uy0, uy2), vy1 = addmod(uy1, uy3), vy3 = mulI(submod(uy3, uy1));
	const uint2 s0 = mulmod(addmod(vx0, vx1), addmod(vy0, vy1)), s1 = mulmod(submod(vx0, vx1), submod(vy0, vy1));
	const uint2 s2 = mulmod(addmod(vx2, vx3), addmod(vy2, vy3)), s3 = mulmod(submod(vx2, vx3), submod(vy2, vy3));
	const uint2 t0 = addmod(s0, s1), t2 = addmod(s2, s3), t1 = submod(s0, s1), t3 = mulI(submod(s2, s3));
	x[i + 0] = addmod(t0, t2); x[i + 2] = submod(t0, t2); x[i + 1] = addmod(t1, t3); x[i + 3] = submod(t1, t3);
}

__kernel __attribute__((reqd_work_group_size(8 / 4 * BLK8, 1, 1)))
void square8(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)
{
	__local uint2 X[8 * BLK8];

	const size_t i = get_local_id(0);
	const size_t i_2 = i % 2, _i2 = ((4 * i) & (size_t)~(4 * 2 - 1)), i2 = _i2 | i_2, j2 = i_2, i_0 = _i2 | (2 * i_2);
	const size_t k2 = get_group_id(0) * 8 * BLK8 | i2;

	_forward4i(2, &X[i2], 2, &x[k2], r2ir2[j2].s01, r1ir1[j2]);
	_square2(&X[i_0]);
	_backward4o(2, &x[k2], 2, &X[i2], r2ir2[j2].s23, r1ir1[j2]);
}

__kernel __attribute__((reqd_work_group_size(16 / 4 * BLK16, 1, 1)))
void square16(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)
{
	__local uint2 X[16 * BLK16];

	const size_t i = get_local_id(0);
	const size_t i_4 = i % 4, i4 = ((4 * i) & (size_t)~(4 * 4 - 1)) | i_4, j4 = i_4;
	const size_t k4 = get_group_id(0) * 16 * BLK16 | i4;

	_forward4i(4, &X[i4], 4, &x[k4], r2ir2[j4].s01, r1ir1[j4]);
	_square4(&X[4 * i]);
	_backward4o(4, &x[k4], 4, &X[i4], r2ir2[j4].s23, r1ir1[j4]);
}

__kernel __attribute__((reqd_work_group_size(32 / 4 * BLK32, 1, 1)))
void square32(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)
{
	__local uint2 X[32 * BLK32];

	// copy mem first ?

	const size_t i = get_local_id(0);
	const size_t i_8 = i % 8, i8 = ((4 * i) & (size_t)~(4 * 8 - 1)) | i_8, j8 = i_8 + 2;
	const size_t i_2 = i % 2, _i2 = ((4 * i) & (size_t)~(4 * 2 - 1)), i2 = _i2 | i_2, j2 = i_2, i_0 = _i2 | (2 * i_2);
	const size_t k8 = get_group_id(0) * 32 * BLK32 | i8;

	_forward4i(8, &X[i8], 8, &x[k8], r2ir2[j8].s01, r1ir1[j8]);
	_forward4(2, &X[i2], r2ir2[j2].s01, r1ir1[j2]);
	_square2(&X[i_0]);
	_backward4(2, &X[i2], r2ir2[j2].s23, r1ir1[j2]);
	_backward4o(8, &x[k8], 8, &X[i8], r2ir2[j8].s23, r1ir1[j8]);
}

__kernel __attribute__((reqd_work_group_size(64 / 4 * BLK64, 1, 1)))
void square64(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)
{
	__local uint2 X[64 * BLK64];

	const size_t i = get_local_id(0);
	const size_t i_16 = i % 16, i16 = ((4 * i) & (size_t)~(4 * 16 - 1)) | i_16, j16 = i_16 + 4;
	const size_t i_4 = i % 4, i4 = ((4 * i) & (size_t)~(4 * 4 - 1)) | i_4, j4 = i_4;
	const size_t k16 = get_group_id(0) * 64 * BLK64 | i16;

	_forward4i(16, &X[i16], 16, &x[k16], r2ir2[j16].s01, r1ir1[j16]);
	_forward4(4, &X[i4], r2ir2[j4].s01, r1ir1[j4]);
	_square4(&X[4 * i]);
	_backward4(4, &X[i4], r2ir2[j4].s23, r1ir1[j4]);
	_backward4o(16, &x[k16], 16, &X[i16], r2ir2[j16].s23, r1ir1[j16]);
}

__kernel __attribute__((reqd_work_group_size(128 / 4 * BLK128, 1, 1)))
void square128(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)
{
	__local uint2 X[128 * BLK128];

	const size_t i = get_local_id(0);
	const size_t i_32 = i % 32, i32 = ((4 * i) & (size_t)~(4 * 32 - 1)) | i_32, j32 = i_32 + 2 + 8;
	const size_t i_8 = i % 8, i8 = ((4 * i) & (size_t)~(4 * 8 - 1)) | i_8, j8 = i_8 + 2;
	const size_t i_2 = i % 2, _i2 = ((4 * i) & (size_t)~(4 * 2 - 1)), i2 = _i2 | i_2, j2 = i_2, i_0 = _i2 | (2 * i_2);
	const size_t k32 = get_group_id(0) * 128 * BLK128 | i32;

	_forward4i(32, &X[i32], 32, &x[k32], r2ir2[j32].s01, r1ir1[j32]);
	_forward4(8, &X[i8], r2ir2[j8].s01, r1ir1[j8]);
	_forward4(2, &X[i2], r2ir2[j2].s01, r1ir1[j2]);
	_square2(&X[i_0]);
	_backward4(2, &X[i2], r2ir2[j2].s23, r1ir1[j2]);
	_backward4(8, &X[i8], r2ir2[j8].s23, r1ir1[j8]);
	_backward4o(32, &x[k32], 32, &X[i32], r2ir2[j32].s23, r1ir1[j32]);
}

__kernel __attribute__((reqd_work_group_size(256 / 4 * BLK256, 1, 1)))
void square256(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)
{
	__local uint2 X[256 * BLK256];

	const size_t i = get_local_id(0);
	const size_t i_64 = i % 64, i64 = ((4 * i) & (size_t)~(4 * 64 - 1)) | i_64, j64 = i_64 + 4 + 16;
	const size_t i_16 = i % 16, i16 = ((4 * i) & (size_t)~(4 * 16 - 1)) | i_16, j16 = i_16 + 4;
	const size_t i_4 = i % 4, i4 = ((4 * i) & (size_t)~(4 * 4 - 1)) | i_4, j4 = i_4;
	const size_t k64 = get_group_id(0) * 256 * BLK256 | i64;

	_forward4i(64, &X[i64], 64, &x[k64], r2ir2[j64].s01, r1ir1[j64]);
	_forward4(16, &X[i16], r2ir2[j16].s01, r1ir1[j16]);
	_forward4(4, &X[i4], r2ir2[j4].s01, r1ir1[j4]);
	_square4(&X[4 * i]);
	_backward4(4, &X[i4], r2ir2[j4].s23, r1ir1[j4]);
	_backward4(16, &X[i16], r2ir2[j16].s23, r1ir1[j16]);
	_backward4o(64, &x[k64], 64, &X[i64], r2ir2[j64].s23, r1ir1[j64]);
}

__kernel __attribute__((reqd_work_group_size(512 / 4, 1, 1)))
void square512(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)
{
	__local uint2 X[512];

	const size_t i = get_local_id(0);
	const size_t i128 = i, j128 = i + 2 + 8 + 32;
	const size_t i_32 = i % 32, i32 = ((4 * i) & (size_t)~(4 * 32 - 1)) | i_32, j32 = i_32 + 2 + 8;
	const size_t i_8 = i % 8, i8 = ((4 * i) & (size_t)~(4 * 8 - 1)) | i_8, j8 = i_8 + 2;
	const size_t i_2 = i % 2, _i2 = ((4 * i) & (size_t)~(4 * 2 - 1)), i2 = _i2 | i_2, j2 = i_2, i_0 = _i2 | (2 * i_2);
	const size_t k128 = get_group_id(0) * 512 | i128;

	_forward4i(128, &X[i128], 128, &x[k128], r2ir2[j128].s01, r1ir1[j128]);
	_forward4(32, &X[i32], r2ir2[j32].s01, r1ir1[j32]);
	_forward4(8, &X[i8], r2ir2[j8].s01, r1ir1[j8]);
	_forward4(2, &X[i2], r2ir2[j2].s01, r1ir1[j2]);
	_square2(&X[i_0]);
	_backward4(2, &X[i2], r2ir2[j2].s23, r1ir1[j2]);
	_backward4(8, &X[i8], r2ir2[j8].s23, r1ir1[j8]);
	_backward4(32, &X[i32], r2ir2[j32].s23, r1ir1[j32]);
	_backward4o(128, &x[k128], 128, &X[i128], r2ir2[j128].s23, r1ir1[j128]);
}

__kernel __attribute__((reqd_work_group_size(1024 / 4, 1, 1)))
void square1024(__global uint2 * restrict const x, __constmem const uint4 * restrict const r1ir1, __constmem const uint4 * restrict const r2ir2)
{
	__local uint2 X[1024];

	const size_t i = get_local_id(0);
	const size_t i256 = i, j256 = i + 4 + 16 + 64;
	const size_t i_64 = i % 64, i64 = ((4 * i) & (size_t)~(4 * 64 - 1)) | i_64, j64 = i_64 + 4 + 16;
	const size_t i_16 = i % 16, i16 = ((4 * i) & (size_t)~(4 * 16 - 1)) | i_16, j16 = i_16 + 4;
	const size_t i_4 = i % 4, i4 = ((4 * i) & (size_t)~(4 * 4 - 1)) | i_4, j4 = i_4;
	const size_t k256 = get_group_id(0) * 1024 | i256;

	_forward4i(256, &X[i256], 256, &x[k256], r2ir2[j256].s01, r1ir1[j256]);
	_forward4(64, &X[i64], r2ir2[j64].s01, r1ir1[j64]);
	_forward4(16, &X[i16], r2ir2[j16].s01, r1ir1[j16]);
	_forward4(4, &X[i4], r2ir2[j4].s01, r1ir1[j4]);
	_square4(&X[4 * i]);
	_backward4(4, &X[i4], r2ir2[j4].s23, r1ir1[j4]);
	_backward4(16, &X[i16], r2ir2[j16].s23, r1ir1[j16]);
	_backward4(64, &X[i64], r2ir2[j64].s23, r1ir1[j64]);
	_backward4o(256, &x[k256], 256, &X[i256], r2ir2[j256].s23, r1ir1[j256]);
}

