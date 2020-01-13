/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "ocl.h"

#include <cstdint>
#include <cmath>
#include <sstream>
#include <fstream>

#include "proth_ocl.h"

class gpmp
{
private:
	static const int digit_bit = 21;
	static const uint32_t digit_mask = (uint32_t(1) << digit_bit) - 1;

	static const uint32_t P1 = 2130706433u;		// 127 * 2^24 + 1 = 2^31 - 2^24 + 1
	static const uint32_t P2 = 2013265921u;		//  15 * 2^27 + 1 = 2^31 - 2^27 + 1
	static const uint32_t P1_PRIM_ROOT = 3u;
	static const uint32_t P2_PRIM_ROOT = 31u;
	static const uint64_t P1P2 = (P1 * uint64_t(P2));

	// 2^20 / 2 * (2^21 - 1)^2 > P1P2 / 2 > 2^19 / 2 * (2^21 - 1)^2 => max size = 2^19

private:
	static constexpr int log2(const size_t n) { return (n > 1) ? 1 + log2(n >> 1) : 0; }

private:
	static constexpr uint32_t invert(const uint32_t n, const uint32_t m)
	{
		__int64 s0 = 1, s1 = 0, d0 = n % m, d1 = m;
		
		while (d1 != 0)
		{
			const __int64 q = d0 / d1;
			d0 -= q * d1;
			const __int64 t1 = d0; d0 = d1; d1 = t1;
			s0 -= q * s1;
			const __int64 t2 = s0; s0 = s1; s1 = t2;
		}
		
		if (d0 != 1) return 0;

		if (s0 < 0) s0 += m;

		return uint32_t(s0);
	}

private:
	static constexpr size_t transformSize(const uint32_t k, const uint32_t n)
	{
		// P = k.2^n + 1 = k.2^s * (2^digit_bit)^e + 1
		const size_t e = n / digit_bit;
		const int s = int(n % digit_bit);

		// a < P => X = a^2 <= (P - 1)^2 = (k.2^s)^2 * (2^digit_bit)^{2e}
		// X_hi = [X / (2^digit_bit)^e] <= (k.2^s)^2 * (2^digit_bit)^e
		// bit size of X_hi < 2 * (log2(k) + 1 + s) + digit_bit * e
		const size_t x_hi_size = e + (2 * (log2(k) + 1 + s) + digit_bit - 1) / digit_bit;

		// Power of 2 such that size >= 2 * X_hi_size (we have to compute X^2 such that X < P)
		size_t size = 2048;
		while (size < 2 * x_hi_size) size *= 2;
		return size;
	}

private:
	const size_t _size;
	const uint32_t _k, _n;
	bool _sign;
	ocl::Device & _device;
	cl_uint2 * const _x;

private:
	template <uint32_t p> class Zp
	{
	private:
		uint32_t n;

	public:
		explicit Zp(const uint32_t i) : n(i % p) {}

		operator uint32_t() const { return n; }

		Zp & operator*=(const Zp & rhs) { n = uint32_t((uint64_t(n) * rhs.n) % p); return *this; }
		Zp operator*(const Zp & rhs) const { Zp r = *this; r *= rhs; return r; }

		Zp pow(const uint64_t e) const
		{
			Zp r = Zp(1u), y = *this;
			for (uint64_t i = e; i != 1; i /= 2)
			{
				if (i % 2 != 0) r *= y;
				y *= y;
			}
			return r * y;
		}

		Zp invert() const { return Zp(gpmp::invert(n, p)); }
	};

	class RNS
	{
	private:
		Zp<P1> n1;
		Zp<P2> n2;

	public:
		explicit RNS(const uint32_t r1, const uint32_t r2) : n1(r1), n2(r2) {}

		uint32_t get1() const { return n1; }
		uint32_t get2() const { return n2; }

		RNS & operator*=(const RNS & rhs) { n1 *= rhs.n1; n2 *= rhs.n2; return *this; }
		RNS operator*(const RNS & rhs) const { RNS r = *this; r *= rhs; return r; }

		RNS invert() const { return RNS(n1.invert(), n2.invert()); }
		static RNS prRoot(const size_t n) { return RNS(Zp<P1>(P1_PRIM_ROOT).pow((P1 - 1) / n), Zp<P2>(P2_PRIM_ROOT).pow((P2 - 1) / n)); }
	};

public:
	gpmp(const uint32_t k, const uint32_t n, ocl::Device & device) :
		_size(transformSize(k, n)), _k(k), _n(n), _sign(false), _device(device), _x(new cl_uint2[_size])
	{
		const size_t size = _size;
		const size_t constant_max_m = 256;
		const size_t constant_size = 256 + 64 + 16 + 4;	// 340 * 2 * sizeof(cl_uint4) = 10880 bytes

		if (size / 2 * double(digit_mask) * digit_mask >= P1P2 / 2)
		{
			std::stringstream msg; msg << getDigits() << "-digit numbers are not supported.";
			throw std::runtime_error(msg.str());
		}

		std::stringstream src;

		std::ifstream clFile("ocl/proth.cl"); 
		if (clFile.is_open())	// if proth.cl file exists then generate proth_ocl.h
		{
			std::ofstream hFile("src/proth_ocl.h");
			if (!hFile.is_open()) throw std::runtime_error("cannot with 'proth_ocl.h' file.");

			hFile << "/*" << std::endl;
			hFile << "Copyright 2020, Yves Gallot" << std::endl << std::endl;
			hFile << "proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it." << std::endl;
			hFile << "Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful." << std::endl;
			hFile << "*/" << std::endl << std::endl;
			hFile << "static const char * const src_proth_ocl = \\" << std::endl;

			std::string line;
			while (std::getline(clFile, line))
			{
				hFile << "\"" << line << "\\n\" \\" << std::endl;
				src << line << std::endl;
			}
			hFile << ";" << std::endl;

			hFile.close();
			clFile.close();
		}
		else	// otherwise program is proth_ocl.h
		{
			src << src_proth_ocl;
		}

		_device.loadProgram(src.str().c_str());

		_device.allocMemory(size, constant_size);
		const cl_uint2 norm = { cl_uint(P1 - (P1 - 1) / size), cl_uint(P2 - (P2 - 1) / size) };
		const cl_int k_shift = cl_int(log2(k) - 1);
		_device.createKernels(norm, cl_uint(n / digit_bit), cl_int(n % digit_bit), cl_uint(k), cl_uint((uint64_t(1) << (32 + k_shift)) / k), k_shift);

		cl_uint2 * const x = _x;
		x[0] = { 1, 0 };
		for (size_t i = 1; i < size; ++i) x[i] = { 0, 0 };
		_device.writeMemory_x(x);

		// (size + 2) / 3 roots
		cl_uint4 * const r1ir1 = new cl_uint4[size];
		cl_uint2 * const r2 = new cl_uint2[size];
		cl_uint2 * const ir2 = new cl_uint2[size];
		cl_uint4 * const cr1ir1 = new cl_uint4[constant_size];
		cl_uint4 * const cr2ir2 = new cl_uint4[constant_size];
		RNS ps = RNS::prRoot(size), ips = ps.invert();
		size_t j = 0;
		for (size_t m = size / 4; m > 1; m /= 4)
		{
			RNS r1 = RNS(1, 1), ir1 = RNS(1, 1);
			const size_t o = (m < 8) ? 0 : 2 * (m / 2 - 1) / 3;

			for (size_t i = 0; i < m; ++i)
			{
				r1ir1[j] = { r1.get1(), r1.get2(), ir1.get1(), ir1.get2() };
				const RNS r1sq = r1 * r1, ir1sq = ir1 * ir1;
				r2[j] = { r1sq.get1(), r1sq.get2() }; ir2[j] = { ir1sq.get1(), ir1sq.get2() };
				++j;

				if (m <= constant_max_m)
				{
					cr1ir1[o + i] = { r1.get1(), r1.get2(), ir1.get1(), ir1.get2() };
					cr2ir2[o + i] = { r1sq.get1(), r1sq.get2(), ir1sq.get1(), ir1sq.get2() };
				}

				r1 *= ps; ir1 *= ips;
			}
			ps *= ps; ps *= ps; ips *= ips; ips *= ips;
		}
		_device.writeMemory_r(r1ir1, r2, ir2);
		_device.writeMemory_cr(cr1ir1, cr2ir2);
		delete[] r1ir1;
		delete[] r2;
		delete[] ir2;
		delete[] cr1ir1;
		delete[] cr2ir2;

		cl_uint * const bp = new cl_uint[size / 2];
		cl_uint * const ibp = new uint32_t[size / 2];
		const uint32_t ib = invert(uint32_t(1) << digit_bit, k);
		uint32_t bp_i = 1, ibp_i = ib;
		for (size_t i = 0; i < size / 2; ++i)
		{
			bp[i] = cl_uint(bp_i);
			ibp[i] = cl_uint(ibp_i);
			bp_i = uint32_t((uint64_t(bp_i) << digit_bit) % k);
			ibp_i = uint32_t((uint64_t(ibp_i) * ib) % k);
		}
		_device.writeMemory_bp(bp, ibp);
		delete[] bp;
		delete[] ibp;

		cl_int err = 0;
		_device.writeMemory_err(&err);
	}

public:
	virtual ~gpmp()
	{
		_device.releaseKernels();
		_device.releaseMemory();
		_device.clearProgram();

		delete[] _x;
	}

public:
	size_t getSize() const { return _size; }
	size_t getDigits() const { return size_t(ceil(log10(_k) + _n * log10(2))); }

public:
	int getError() const
	{
		cl_int err = 0;
		_device.readMemory_err(&err);
		return int(err);
	}

public:
	// void test()
	// {
	// 	const size_t size = _size;
	// 	cl_uint2 * const x = _x;

	// 	for (size_t i = 0; i < size / 2; ++i) x[i] =  { digit_mask, 0 };
	// 	for (size_t i = size / 2; i < size; ++i) x[i] = { 0, 0 };
	// 	square();

	// 	for (size_t i = 0 * size / 4; i < 1 * size / 4; ++i) x[i] = { digit_mask, 0 };
	// 	for (size_t i = 1 * size / 4; i < 2 * size / 4; ++i) x[i] = { 0, digit_mask };
	// 	for (size_t i = size / 2; i < size; ++i) x[i] = { 0, 0 };
	// 	square();
	// }

public:
	void square()
	{
		const size_t size = _size;

		// x size is size / 2; _x[0] = R, _x[1] = Y; compute (R - Y)^2

		_device.sub_ntt64();

		cl_uint m = cl_uint(size / 4);
		cl_uint rindex = m + m / 4 + m / 16;
		m /= 64;

		while (m > 256)
		{
			_device.ntt64(m, rindex);
			rindex += m + m / 4 + m / 16;
			m /= 64;
		}

		if (m == 256)        _device.square1024();
		else if (m == 128)   _device.square512();
		else if (m == 64)    _device.square256();
		else if (m == 32)    _device.square128();
		else if (m == 16)    _device.square64();
		else /*if (m == 8)*/ _device.square32();

		while (m <= cl_uint(size / 16))
		{
			m *= 64;
			rindex -= m + m / 4 + m / 16;
			_device.intt64(m, rindex);
		}

		_device.poly2int0();
		_device.poly2int1();

		_sign = false;

		// x size is size

		split();

		// Now x size is size / 2, _x[0] = R, _x[1] = Y such that X = R - Y and -k.2^n < R - Y < k.2^n
	}

public:
	void mul(const uint32_t a)
	{
		const size_t size = _size;
		cl_uint2 * const x = _x;

		_device.readMemory_x(x);

		// _x[0] = R, _x[1] = Y; compute R - Y
		const bool sign = _sub(x, size / 2);
		_sign = (_sign != sign);

		// x size is size / 2
		_mul(x, size / 2, a);

		_device.writeMemory_x(x);

		// x size is size / 2 + 1

		split();

		// Now x size is size / 2, _x[0] = R, _x[1] = Y such that X = R - Y and -k.2^n < R - Y < k.2^n
	}

public:
	bool isMinusOne(uint64_t & res64)
	{
		const size_t size = _size;
		cl_uint2 * const x = _x;

		_device.readMemory_x(x);

		// Another method is: compute T = R - Y + 1 or T = (R + k*2^n + 1) - Y + 1  such that 0 <= T < k*2^n + 1
		// T == 0?

		// _x[0] = R, _x[1] = Y; compute R - Y
		bool sign = _sub(x, size / 2);
		_sign = (_sign != sign);

		const bool b = (_sign && _isOne(x, size / 2));

		if (_sign)
		{
			_add(x, size / 2, _k, _n);
			_sign = false;
		}

		res64 = _getRes64(x, size / 2) + 1;
		return b;
	}

private:
	void split()
	{
		// Y = [X / k.2^n]
		// X = Y * k.2^n + R, 0 <= R < k.2^n, 0 <= Y < k.2^n
		// X = Y * (k.2^n + 1) + R - Y
		// Then X mod k.2^n + 1 = R - Y with -k.2^n < R - Y < k.2^n

		// P = k.2^n + 1 = k.2^s * (2^digit_bit)^e + 1 = k.2^s * B^e + 1
		// X_hi = [X / B^e], X_lo = X mod B^e
		// Y = [X_hi / k'], r = X_hi mod k'
		// R = r * B^e + X_lo

		// x size is size, 0 <= e < size / 2

		// Daisuke Takahashi, A parallel algorithm for multiple-precision division by a single-precision integer.

		_device.reduce_i();

		// x size is size / 2, x = X mod B^(size / 2), y = X / (B^e * 2^s)

		const cl_uint n = cl_uint(_size / 2);
		cl_uint j = 4;		// alignment (cl_uint4)
		cl_uint s = n / 64;
		for (; true; s /= 64)
		{
			_device.reduce_upsweep64(s, j);
			j += ((5 * 4 + 5) * 4 + 5) * s;
			if (s <= 256) break;
		}

		if (s == 256)        _device.reduce_topsweep1024(j);
		else if (s == 128)   _device.reduce_topsweep512(j);
		else if (s == 64)    _device.reduce_topsweep256(j);
		else if (s == 32)    _device.reduce_topsweep128(j);
		else if (s == 16)    _device.reduce_topsweep64(j);
		else /*if (s == 8)*/ _device.reduce_topsweep32(j);

		for (; s <= n / 64; s *= 64)
		{
			j -= ((5 * 4 + 5) * 4 + 5) * s;
			_device.reduce_downsweep64(s, j);
		}

		// t[4 + k] remainders y[k + 1] / d, t[0] = remainder y[0] / d

		_device.reduce_o();

		// x size is size / 2, x[0] = X mod B^n, x[1] = Y

		_device.reduce_f();

		// x size is size / 2, _x[0] = R, _x[1] = Y
	}

private:
	static bool _isOne(cl_uint2 * const a, const size_t size)
	{
		if (a[0].s[0] != 1) return false;
		for (size_t i = 1; i < size; ++i) if (a[i].s[0] != 0) return false;
		return true;
	}

private:
	static uint64_t _getRes64(cl_uint2 * const a, const size_t size)
	{
		uint64_t r = 0, b = 1;
		for (size_t i = 0; i < size; ++i)
		{
			r += a[i].s[0] * b;
			b <<= digit_bit;
		}
		return r;
	}

private:
	static void _mul(cl_uint2 * const a, const size_t size, const uint32_t d)
	{
		uint64_t l = 0;
		for (size_t i = 0; i < size; ++i)
		{
			l += a[i].s[0] * uint64_t(d);
			a[i].s[0] = uint32_t(l) & digit_mask;
			a[i + size].s[0] = 0;
			l >>= digit_bit;
		}

		for (size_t i = size; l != 0; ++i)
		{
 			a[i].s[0] = uint32_t(l) & digit_mask;
			l >>= digit_bit;
		}
	}

private:
	static bool _sub(cl_uint2 * const a, const size_t size)
	{
		int32_t carry = 0;
		for (size_t i = 0; i < size; ++i)
		{
			const int32_t s = a[i].s[0] - a[i].s[1] + carry;
			a[i].s[0] = uint32_t(s) & digit_mask;
			carry = s >> digit_bit;
		}
		if (carry == 0) return false;
		carry = 0;
		for (size_t i = 0; i < size; ++i)
		{
			const int32_t s = carry - a[i].s[0];
			a[i].s[0] = uint32_t(s) & digit_mask;
			carry = s >> digit_bit;
		}
		return true;
	}

private:
	static void _add(cl_uint2 * const a, const size_t size, const uint32_t k, const uint32_t n)
	{
		const uint32_t e = n / digit_bit, s = n % digit_bit;
		const uint64_t ks = uint64_t(k) << s;
		const uint32_t ak[3] = { uint32_t(ks) & digit_mask, uint32_t(ks >> digit_bit) & digit_mask, uint32_t(ks >> (2 * digit_bit)) & digit_mask};

		int32_t carry = 1;
		for (size_t i = 0; i < size; ++i)
		{
			if (i == e) carry += ak[0];
			if (i == e + 1) carry += ak[1];
			if (i == e + 2) carry += ak[2];
			const int32_t s = carry - a[i].s[0];
			a[i].s[0] = uint32_t(s) & digit_mask;
			carry = s >> digit_bit;
		}
	}
};
