/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#define	digit_bit		21
#define digit_mask		((1u << digit_bit) - 1)

#define	P1			2130706433u		// 127 * 2^24 + 1 = 2^31 - 2^24 + 1
#define	P2			2013265921u		//  15 * 2^27 + 1 = 2^31 - 2^27 + 1
#define	P1_INV		2164392967u		// 2^62 / P1
#define	P2_INV 		2290649223u		// 2^62 / P2
#define	P1_I		2113994754u		// P1_PRIM_ROOT^((P1 - 1) / 4)
#define	P2_I		1728404513u		// P2_PRIM_ROOT^((P2 - 1) / 4)
#define	InvP2_P1	913159918u		// 1 / P2 mod P1
#define	P1P2		(P1 * (ulong)(P2))

/*
Barrett's product/reduction, where P is such that h (the number of iterations in the 'while loop') is 0 or 1.

Let m < P^2 < 2^62, R = [2^62/P] and h = [m/P] - [[m/2^30] R / 2^32].

We have h = ([m/P] - m/P) + m/2^62 (2^62/P - R) + R/2^32 (m/2^30 - [m/2^30]) + ([m/2^30] R / 2^32 - [[m/2^30] * R / 2^32]).
Then -1 + 0 + 0 + 0 < h < 0 + (2^62/P - R) + R/2^32 + 1,
0 <= h < 1 + (2^62/P - R) + R/2^32.

P = 127 * 2^24 + 1 = 2130706433 => R = 2164392967, h < 1.56
P =  63 * 2^25 + 1 = 2113929217 => R = 2181570688, h < 2.51 NOK
P =  15 * 2^27 + 1 = 2013265921 => R = 2290649223, h < 1.93
*/

inline uint _addmod(const uint lhs, const uint rhs, const uint d)
{
	const uint r = lhs + rhs;
	const uint t = (lhs >= d - rhs) ? d : 0;
	return r - t;
}

inline uint _rem(const ulong q, const uint d, const uint d_inv, const int d_shift)
{
	const uint q_d = mul_hi((uint)(q >> d_shift), d_inv);
	const uint r = (uint)(q) - q_d * d;
	const uint t = (r >= d) ? d : 0;
	return r - t;
}

inline uint _mulMod(const uint a, const uint b, const uint p, const uint p_inv)
{
	return _rem(a * (ulong)(b), p, p_inv, 30);
}

inline uint mulModP1(const uint a, const uint b) { return _mulMod(a, b, P1, P1_INV); }
inline uint mulModP2(const uint a, const uint b) { return _mulMod(a, b, P2, P2_INV); }

inline long getlong(const uint2 lhs)
{
	// Garner Algorithm
	uint d = lhs.s0 - lhs.s1; const uint t = (lhs.s0 < lhs.s1) ? P1 : 0; d += t;	// mod P1
	const uint u = mulModP1(d, InvP2_P1);		// P2 < P1
	const ulong r = u * (ulong)(P2) + lhs.s1;
	const ulong s = (r > P1P2 / 2) ? P1P2 : 0;
	return (long)(r - s);
}

inline uint2 addmod(const uint2 lhs, const uint2 rhs)
{
	const uint2 r = lhs + rhs;
	const uint2 t = (uint2)((lhs.s0 >= P1 - rhs.s0) ? P1 : 0, (lhs.s1 >= P2 - rhs.s1) ? P2 : 0);
	return r - t;
}

inline uint2 submod(const uint2 lhs, const uint2 rhs)
{
	const uint2 r = lhs - rhs;
	const uint2 t = (uint2)((lhs.s0 < rhs.s0) ? P1 : 0, (lhs.s1 < rhs.s1) ? P2 : 0);
	return r + t;
}

inline uint2 mulmod(const uint2 lhs, const uint2 rhs)
{
	return (uint2)(mulModP1(lhs.s0, rhs.s0), mulModP2(lhs.s1, rhs.s1));
}

inline uint2 sqrmod(const uint2 lhs) { return mulmod(lhs, lhs); }

inline uint2 mulI(const uint2 lhs) { return mulmod(lhs, (uint2)(P1_I, P2_I)); }

__kernel
void sub_ntt4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint rindex)
{
	const size_t n_4 = get_global_size(0), k = get_global_id(0);

	const size_t i = k;

	const uint4 r1ir1_i = r1ir1[rindex + i];
	const uint2 r2_i = r2[rindex + i];

	const uint2 abi = x[i + 0 * n_4], abim = x[i + 1 * n_4];
	const uint2 abi0 = (uint2)(abi.s0, abi.s0), abi1 = (uint2)(abi.s1, abi.s1);
	const uint2 abim0 = (uint2)(abim.s0, abim.s0), abim1 = (uint2)(abim.s1, abim.s1);
	const uint2 u0 = submod(abi0, abi1), u1 = submod(abim0, abim1), u3 = mulI(u1);
	x[i + 0 * n_4] = addmod(u0, u1); x[i + 1 * n_4] = mulmod(submod(u0, u1), r2_i);
	x[i + 2 * n_4] = mulmod(submod(u0, u3), r1ir1_i.s23); x[i + 3 * n_4] = mulmod(addmod(u0, u3), r1ir1_i.s01);
}

__kernel
void ntt4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const r2, const uint m, const uint rindex)
{
	const size_t n_4 = get_global_size(0), k = get_global_id(0);

	const size_t i = k & (m - 1), j = 4 * k - 3 * i;

	const uint2 r2_i = r2[rindex + i];
	const uint4 r1ir1_i = r1ir1[rindex + i];

	const uint2 u0 = x[j + 0 * m], u2 = x[j + 2 * m], u1 = x[j + 1 * m], u3 = x[j + 3 * m];
	const uint2 v0 = addmod(u0, u2), v2 = submod(u0, u2), v1 = addmod(u1, u3), v3 = mulI(submod(u3, u1));
	x[j + 0 * m] = addmod(v0, v1); x[j + 1 * m] = mulmod(submod(v0, v1), r2_i);
	x[j + 2 * m] = mulmod(addmod(v2, v3), r1ir1_i.s23); x[j + 3 * m] = mulmod(submod(v2, v3), r1ir1_i.s01);
}

__kernel
void intt4(__global uint2 * restrict const x, __global const uint4 * restrict const r1ir1, __global const uint2 * restrict const ir2, const uint m, const uint rindex)
{
	const size_t n_4 = get_global_size(0), k = get_global_id(0);

	const size_t i = k & (m - 1), j = 4 * k - 3 * i;

	const uint2 ir2_i = ir2[rindex + i];
	const uint4 r1ir1_i = r1ir1[rindex + i];

	const uint2 v0 = x[j + 0 * m], v1 = mulmod(x[j + 1 * m], ir2_i), v2 = mulmod(x[j + 2 * m], r1ir1_i.s01), v3 = mulmod(x[j + 3 * m], r1ir1_i.s23);
	const uint2 u0 = addmod(v0, v1), u2 = addmod(v2, v3), u1 = submod(v0, v1), u3 = mulI(submod(v2, v3));
	x[j + 0 * m] = addmod(u0, u2); x[j + 2 * m] = submod(u0, u2);
	x[j + 1 * m] = addmod(u1, u3); x[j + 3 * m] = submod(u1, u3);
}

__kernel
void square2(__global uint2 * restrict const x)
{
	const size_t n_4 = get_global_size(0), k = get_global_id(0);

	const size_t i = 4 * k;

	const uint2 u0 = x[i + 0], u1 = x[i + 1], u2 = x[i + 2], u3 = x[i + 3];
	const uint2 v0 = addmod(u0, u1), v1 = submod(u0, u1), v2 = addmod(u2, u3), v3 = submod(u2, u3);
	const uint2 s0 = sqrmod(v0), s1 = sqrmod(v1), s2 = sqrmod(v2), s3 = sqrmod(v3);
	x[i + 0] = addmod(s0, s1); x[i + 1] = submod(s0, s1); x[i + 2] = addmod(s2, s3); x[i + 3] = submod(s2, s3);
}

__kernel
void square4(__global uint2 * restrict const x)
{
	const size_t n_4 = get_global_size(0), k = get_global_id(0);

	const size_t i = 4 * k;

	const uint2 u0 = x[i + 0], u2 = x[i + 2], u1 = x[i + 1], u3 = x[i + 3];
	const uint2 v0 = addmod(u0, u2), v2 = submod(u0, u2), v1 = addmod(u1, u3), v3 = mulI(submod(u3, u1));
	const uint2 s0 = sqrmod(addmod(v0, v1)), s1 = sqrmod(submod(v0, v1)), s2 = sqrmod(addmod(v2, v3)), s3 = sqrmod(submod(v2, v3));
	const uint2 t0 = addmod(s0, s1), t2 = addmod(s2, s3), t1 = submod(s0, s1), t3 = mulI(submod(s2, s3));
	x[i + 0] = addmod(t0, t2); x[i + 2] = submod(t0, t2); x[i + 1] = addmod(t1, t3); x[i + 3] = submod(t1, t3);
}

__kernel
void poly2int0(__global uint2 * restrict const x, __global long * restrict const cr, const uint2 norm, const uint blk)
{
	const size_t n_blk = get_global_size(0), k = get_global_id(0);

	__global uint2 * restrict const xi = &x[blk * k];

	long l = 0;
	for (size_t j = 0; j < blk; ++j)
	{
		l += getlong(mulmod(xi[j], norm));	// -n/2 . (B-1)^2 <= l <= n/2 . (B-1)^2
		xi[j].s0 = (uint)(l) & digit_mask;
		l >>= digit_bit;
	}
	cr[(k + 1) & (n_blk - 1)] = l;			// |l| < n/2 . (B-1)
}

__kernel
void poly2int1(__global uint2 * restrict const x, __global const long * restrict const cr, const uint blk, __global int * const err)
{
	const size_t k = get_global_id(0);

	__global uint2 * restrict const xi = &x[blk * k];

	long l = cr[k] + xi[0].s0;
	xi[0].s0 = (uint)(l) & digit_mask;
	l >>= digit_bit;						// |l| < n/2

	int f = (int)(l);
	for (size_t j = 1; j < blk; ++j)
	{
		f += xi[j].s0;
		xi[j].s0 = (uint)(f) & digit_mask;
		f >>= digit_bit;					// f = -1, 0 or 1
		if (f == 0) break;
	}

	if (f != 0) atomic_or(err, 1);
}

__kernel
void split0(__global uint2 * restrict const x, const uint e, const int s)
{
	const size_t n = get_global_size(0), k = get_global_id(0);

	const ulong l = ((ulong)(x[e + k + 1].s0) << digit_bit) | x[e + k].s0;
	const uint r = (uint)(l >> s) & digit_mask;
	x[k].s1 = r;
}

__kernel
void split4_i(__global uint2 * restrict const x, __global const uint * restrict const bp, const uint d, const uint d_inv, const int d_shift)
{
	const size_t n = 4 * get_global_size(0), k = get_global_id(0);

	__global uint2 * restrict const xj = &x[n + 4 * k];

	const uint u0 = _rem(x[4 * k + 0].s1 * (ulong)(bp[4 * k + 0]), d, d_inv, d_shift), u1 = _rem(x[4 * k + 1].s1 * (ulong)(bp[4 * k + 1]), d, d_inv, d_shift);
	const uint u2 = _rem(x[4 * k + 2].s1 * (ulong)(bp[4 * k + 2]), d, d_inv, d_shift), u3 = _rem(x[4 * k + 3].s1 * (ulong)(bp[4 * k + 3]), d, d_inv, d_shift);
	const uint u01 = _addmod(u0, u1, d), u23 = _addmod(u2, u3, d);
	xj[0].s1 = _addmod(u01, u23, d); xj[1].s1 = _addmod(u1, u23, d);
	xj[2].s1 = u23; xj[3].s1 = u3;
}

__kernel
void split4_01(__global uint2 * restrict const x, const uint d, const uint m)
{
	const size_t n = 4 * get_global_size(0), k = get_global_id(0);

	const size_t i = k & (m - 1);

	__global uint2 * restrict const xj = &x[n + 4 * (k - i)];

	const uint u0 = xj[i + 0 * m].s0, u1 = xj[i + 1 * m].s0, u2 = xj[i + 2 * m].s0, u3 = xj[i + 3 * m].s0;
	const uint s1 = xj[1 * m].s0, s2 = xj[2 * m].s0, s3 = xj[3 * m].s0;
	const uint u01 = _addmod(u0, s1, d), u23 = _addmod(u2, s3, d), s23 = _addmod(s2, s3, d);
	xj[i + 0 * m].s1 = _addmod(u01, s23, d); xj[i + 1 * m].s1 = _addmod(u1, s23, d);
	xj[i + 2 * m].s1 = u23; xj[i + 3 * m].s1 = u3;
}

__kernel
void split4_10(__global uint2 * restrict const x, const uint d, const uint m)
{
	const size_t n = 4 * get_global_size(0), k = get_global_id(0);

	const size_t i = k & (m - 1);

	__global uint2 * restrict const xj = &x[n + 4 * (k - i)];

	const uint u0 = xj[i + 0 * m].s1, u1 = xj[i + 1 * m].s1, u2 = xj[i + 2 * m].s1, u3 = xj[i + 3 * m].s1;
	const uint s1 = xj[1 * m].s1, s2 = xj[2 * m].s1, s3 = xj[3 * m].s1;
	const uint u01 = _addmod(u0, s1, d), u23 = _addmod(u2, s3, d), s23 = _addmod(s2, s3, d);
	xj[i + 0 * m].s0 = _addmod(u01, s23, d); xj[i + 1 * m].s0 = _addmod(u1, s23, d);
	xj[i + 2 * m].s0 = u23; xj[i + 3 * m].s0 = u3;
}

__kernel
void split2(__global uint2 * restrict const x, const uint d)
{
	const size_t n_2 = get_global_size(0), k = get_global_id(0);

	__global uint2 * restrict const xn = &x[2 * n_2];

	xn[k].s0 = _addmod(xn[k].s0, xn[n_2].s0, d);
}

__kernel
void split2_10(__global uint2 * restrict const x, const uint d)
{
	const size_t n_2 = get_global_size(0), k = get_global_id(0);

	__global uint2 * restrict const xn = &x[2 * n_2];

	xn[k].s0 = _addmod(xn[k].s1, xn[n_2].s1, d);
	xn[k + n_2].s0 = xn[k + n_2].s1;
}

__kernel
void split_o(__global uint2 * restrict const x, __global const uint * restrict const ibp, const uint e, const int s, const uint d, const uint d_inv, const int d_shift)
{
	const size_t n = get_global_size(0), k = get_global_id(0);

	const uint rbk_prev = (k + 1 < n) ? x[n + k + 1].s0 : 0;
	const uint r_prev = _rem(rbk_prev * (ulong)(ibp[k]), d, d_inv, d_shift);

	const uint2 x_k = x[k];

	const ulong q = ((ulong)(r_prev) << digit_bit) | x_k.s1;

	const uint q_d = mul_hi((uint)(q >> d_shift), d_inv);	// d < 2^29
	const uint r = (uint)(q) - q_d * d;
	const uint t = (r >= d) ? 1 : 0;

	//x[k] = (uint2)((k > e + 2) ? 0 : x_k.s0, q_d + t);
	if (k > e + 2) x[k].s0 = 0;
	x[k].s1 = q_d + t;
}

__kernel
void split_o_10(__global uint2 * restrict const x, __global const uint * restrict const ibp, const uint e, const int s, const uint d, const uint d_inv, const int d_shift)
{
	const size_t n = get_global_size(0), k = get_global_id(0);

	const uint rbk_prev = (k + 1 < n) ? x[n + k + 1].s1 : 0;
	const uint r_prev = _rem(rbk_prev * (ulong)(ibp[k]), d, d_inv, d_shift);

	const uint2 x_k = x[k];

	const ulong q = ((ulong)(r_prev) << digit_bit) | x_k.s1;

	const uint q_d = mul_hi((uint)(q >> d_shift), d_inv);	// d < 2^29
	const uint r = (uint)(q) - q_d * d;
	const uint t = (r >= d) ? 1 : 0;

	//x[k] = (uint2)((k > e + 2) ? 0 : x_k.s0, q_d + t);
	if (k > e + 2) x[k].s0 = 0;
	x[k].s1 = q_d + t;

	if (k + 1 == n) { x[n].s0 = x[n].s1; }
}

__kernel
void split_f(__global uint2 * restrict const x, const uint n, const uint e, const int s)
{
	const uint rs = (x[e].s0 & ((1u << s) - 1));
	const ulong rds = ((ulong)(x[n].s0) << s) | rs;		// rds < 2^(29 + digit_bit - 1)

	x[e].s0 = (uint)(rds) & digit_mask;
	if (e + 1 < n) x[e + 1].s0 = (uint)(rds >> digit_bit) & digit_mask;
	if (e + 2 < n) x[e + 2].s0 = (uint)(rds >> (2 * digit_bit)) & digit_mask;
}
