/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "arith.h"
#include "engine.h"
#include "plan.h"

#include <cstdint>
#include <cmath>
#include <sstream>
#include <fstream>

#include "proth_ocl.h"
#include "proth_1024_ocl.h"

class gpmp
{
private:
	static const uint32_t P1 = 2130706433u;		// 127 * 2^24 + 1 = 2^31 - 2^24 + 1
	static const uint32_t P2 = 2013265921u;		//  15 * 2^27 + 1 = 2^31 - 2^27 + 1
	static const uint32_t P1_PRIM_ROOT = 3u;
	static const uint32_t P2_PRIM_ROOT = 31u;
	static const uint64_t P1P2 = (P1 * uint64_t(P2));

	// 2^20 / 2 * (2^21 - 1)^2 > P1P2 / 2 > 2^19 / 2 * (2^21 - 1)^2 => max size = 2^19

private:
	static constexpr size_t transformSize(const uint32_t k, const uint32_t n, const int digit_bit)
	{
		// P = k.2^n + 1 = k.2^s * (2^digit_bit)^e + 1
		const size_t e = n / digit_bit;
		const int s = int(n % digit_bit);

		// a < P => X = a^2 <= (P - 1)^2 = (k.2^s)^2 * (2^digit_bit)^{2e}
		// X_hi = [X / (2^digit_bit)^e] <= (k.2^s)^2 * (2^digit_bit)^e
		// bit size of X_hi < 2 * (log2(k) + 1 + s) + digit_bit * e
		const size_t x_hi_size = e + (2 * (arith::log2(k) + 1 + s) + digit_bit - 1) / digit_bit;

		// Power of 2 such that size >= 2 * X_hi_size (we have to compute X^2 such that X < P)
		size_t size = 2048;
		while (size < 2 * x_hi_size) size *= 2;
		return size;
	}

private:
	static constexpr int digitBit(const uint32_t k, const uint32_t n)
	{
		for (int digit_bit = 21; digit_bit > 1; --digit_bit)
		{
			const size_t size = transformSize(k, n, digit_bit);
			const double max_digit = double((uint32_t(1) << digit_bit) - 1);
			if ((size / 2) * max_digit * max_digit < P1P2 / 2) return digit_bit;
		}
		return 1;
	}

private:
	const int _digit_bit;
	const size_t _size;
	const uint32_t _k, _n;
	const bool _ext1024;
	engine & _engine;
	plan _plan;
	cl_uint2 * const _mem;

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

		Zp invert() const { return Zp(arith::invert(n, p)); }
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

private:
	static bool readOpenCL(const char * const clFileName, const char * const headerFileName, const char * const varName, std::stringstream & src)
	{
		std::ifstream clFile(clFileName);
		if (!clFile.is_open()) return false;
		
		// if .cl file exists then generate header file
		std::ofstream hFile(headerFileName, std::ios::binary);	// binary: don't convert line endings to `CRLF` 
		if (!hFile.is_open()) throw std::runtime_error("cannot write openCL header file.");

		hFile << "/*" << std::endl;
		hFile << "Copyright 2020, Yves Gallot" << std::endl << std::endl;
		hFile << "proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it." << std::endl;
		hFile << "Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful." << std::endl;
		hFile << "*/" << std::endl << std::endl;

		hFile << "static const char * const " << varName << " = \\" << std::endl;

		std::string line;
		while (std::getline(clFile, line))
		{
			hFile << "\"" << line << "\\n\" \\" << std::endl;
			src << line << std::endl;
		}
		hFile << ";" << std::endl;

		hFile.close();
		clFile.close();
		return true;
	}

private:
	static inline cl_uint2 set2(const cl_uint s0, const cl_uint s1)
	{
		cl_uint2 r; r.s[0] = s0; r.s[1] = s1;
		return r;
	}

	static inline cl_uint4 set4(const cl_uint s0, const cl_uint s1, const cl_uint s2, const cl_uint s3)
	{
		cl_uint4 r; r.s[0] = s0; r.s[1] = s1; r.s[2] = s2; r.s[3] = s3;
		return r;
	}

private:
	void _initEngine()
	{
		const size_t size = _size;
		const size_t constant_max_m = 1024;
		const size_t constant_size = 1024 + 256 + 64 + 16 + 4;	// 1364 * 2 * sizeof(cl_uint4) = 43648 bytes

		std::stringstream src;
		src << "#define\tdigit_bit\t" << _digit_bit << std::endl << std::endl;

		if (!readOpenCL("ocl/proth.cl", "src/proth_ocl.h", "src_proth_ocl", src))
		{
			// if .cl file is not found then program is proth_ocl.h
			src << src_proth_ocl;
		}

		if (_ext1024)
		{
			if (!readOpenCL("ocl/proth_1024.cl", "src/proth_1024_ocl.h", "src_proth_1024_ocl", src))
			{
				src << src_proth_1024_ocl;
			}
		}

		_engine.loadProgram(src.str().c_str());

		_engine.allocMemory(size, constant_size);
		const cl_uint2 norm = set2(cl_uint(P1 - (P1 - 1) / size), cl_uint(P2 - (P2 - 1) / size));
		const cl_int k_shift = cl_int(arith::log2(_k) - 1);
		_engine.createKernels(norm, cl_uint(_n / _digit_bit), cl_int(_n % _digit_bit), cl_uint(_k), cl_uint((uint64_t(1) << (32 + k_shift)) / _k), k_shift, _ext1024);

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
				r1ir1[j] = set4(r1.get1(), r1.get2(), ir1.get1(), ir1.get2());
				const RNS r1sq = r1 * r1, ir1sq = ir1 * ir1;
				r2[j] = set2(r1sq.get1(), r1sq.get2()); ir2[j] = set2(ir1sq.get1(), ir1sq.get2());
				++j;

				if (m <= constant_max_m)
				{
					cr1ir1[o + i] = set4(r1.get1(), r1.get2(), ir1.get1(), ir1.get2());
					cr2ir2[o + i] = set4(r1sq.get1(), r1sq.get2(), ir1sq.get1(), ir1sq.get2());
				}

				r1 *= ps; ir1 *= ips;
			}
			ps *= ps; ps *= ps; ips *= ips; ips *= ips;
		}
		_engine.writeMemory_r(r1ir1, r2, ir2);
		_engine.writeMemory_cr(cr1ir1, cr2ir2);
		delete[] r1ir1;
		delete[] r2;
		delete[] ir2;
		delete[] cr1ir1;
		delete[] cr2ir2;

		cl_uint * const bp = new cl_uint[size / 2];
		cl_uint * const ibp = new uint32_t[size / 2];
		const uint32_t ib = arith::invert(uint32_t(1) << _digit_bit, _k);
		uint32_t bp_i = 1, ibp_i = ib;
		for (size_t i = 0; i < size / 2; ++i)
		{
			bp[i] = cl_uint(bp_i);
			ibp[i] = cl_uint(ibp_i);
			bp_i = uint32_t((uint64_t(bp_i) << _digit_bit) % _k);
			ibp_i = uint32_t((uint64_t(ibp_i) * ib) % _k);
		}
		_engine.writeMemory_bp(bp, ibp);
		delete[] bp;
		delete[] ibp;

		cl_int err = 0;
		_engine.writeMemory_err(&err);
	}

private:
	void _clearEngine()
	{
		_engine.releaseKernels();
		_engine.releaseMemory();
		_engine.clearProgram();
	}

public:
	gpmp(const uint32_t k, const uint32_t n, engine & engine, const bool profile = false) :
		_digit_bit(digitBit(k, n)), _size(transformSize(k, n, _digit_bit)), _k(k), _n(n),
		_ext1024((engine.getMaxWorkGroupSize() >= 1024) && (engine.getLocalMemSize() >= 32768)), _engine(engine), _mem(new cl_uint2[_size])
	{
		const size_t size = _size;

		const double max_digit = double((uint32_t(1) << _digit_bit) - 1);
		if ((size / 2) * max_digit * max_digit >= P1P2 / 2)
		{
			std::stringstream msg; msg << getDigits() << "-digit numbers are not supported.";
			throw std::runtime_error(msg.str());
		}

		engine.setProfiling(true);
		_initEngine();

		const size_t cnt = _plan.getSquareSeqCount(size, _ext1024);
		cl_ulong bestTime = cl_ulong(-1);
		size_t best_i = 0;
		for (size_t i = 0; i < cnt; ++i)
		{
			initProfiling();
			_plan.setSquareSeq(size, i);
			for (size_t j = 0; j < 16; ++j) square();
			const cl_ulong time = engine.getProfileTime();
			if (time < bestTime)
			{
				bestTime = time;
				best_i = i;
			}
			engine.resetProfiles();
		}

		_plan.setSquareSeq(size, best_i);

		_clearEngine();
		engine.setProfiling(profile);
		_initEngine();
	}

public:
	virtual ~gpmp()
	{
		_clearEngine();

		delete[] _mem;
	}

public:
	size_t getSize() const { return _size; }
	size_t getDigitBit() const { return _digit_bit; }
	size_t getDigits() const { return size_t(std::ceil(std::log10(_k) + _n * std::log10(2))); }

public:
	void display() const
	{
		const size_t size = _size / 2;
		cl_uint2 * const x = _mem;
		_engine.readMemory_x(x);
		std::cout << std::endl;
		for (size_t i = 0; i < size; ++i)
		{
			if (x[i].s[0] != 0) std::cout << " " << i << ":0 " << x[i].s[0];
			if (x[i].s[1] != 0) std::cout << " " << i << ":0 " << x[i].s[1];
		}
		std::cout << std::endl;
	}

public:
	static void printRanges(const uint32_t k)
	{
		std::vector<std::pair<uint32_t, uint32_t>> range(32, std::make_pair(0u, 0u));
		uint32_t n_min = 100000, n_max = 2 * n_min;
		while (n_max < 1000000000)
		{
			while (n_max - n_min > 1)
			{
				const size_t s_min = transformSize(k, n_min, digitBit(k, n_min));
				const size_t s_max = transformSize(k, n_max, digitBit(k, n_max));

				const uint32_t m = (n_min + n_max) / 2;
				const size_t s = transformSize(k, m, digitBit(k, m));
				if (s == s_min) n_min = m;
				if (s == s_max) n_max = m;
			}
			const size_t ls_min = arith::log2(transformSize(k, n_min, digitBit(k, n_min)));
			const size_t ls_max = arith::log2(transformSize(k, n_max, digitBit(k, n_max)));
			range[ls_min].second = n_min;
			range[ls_max].first = n_max;
			n_min = n_max; n_max = 2 * n_min + 1000;
		}
		for (size_t i = 17; i <= 24; ++i)
		{
			std::cout << "2^" << i << ": [" << range[i].first << "-" << range[i].second << "]" << std::endl;
		}
	}

public:
	int getError() const
	{
		cl_int err = 0;
		_engine.readMemory_err(&err);
		return int(err);
	}

private:
	static bool _writeContext(std::ofstream & cFile, const char * const ptr, const size_t size)
	{
		cFile.write(ptr, size);
		if (cFile.good()) return true;
		cFile.close();
		return false;
	}

	static bool _readContext(std::ifstream & cFile, char * const ptr, const size_t size)
	{
		cFile.read(ptr, size);
		if (cFile.good()) return true;
		cFile.close();
		return false;
	}

public:
	bool saveContext(const uint32_t i, const double elapsedTime)
	{
		std::ofstream cFile("proth.ctx", std::ios::binary);
		if (!cFile.is_open())
		{
			std::cerr << "cannot write 'proth.ctx' file" << std::endl;
			return false;
		}

		const size_t size = _size;
		cl_uint2 * const mem = _mem;

		const uint32_t version = 0;
		if (!_writeContext(cFile, reinterpret_cast<const char *>(&version), sizeof(version))) return false;
		if (!_writeContext(cFile, reinterpret_cast<const char *>(&elapsedTime), sizeof(elapsedTime))) return false;
		const uint32_t digit_bit = uint32_t(_digit_bit);
		if (!_writeContext(cFile, reinterpret_cast<const char *>(&digit_bit), sizeof(digit_bit))) return false;
		const uint32_t sz = uint32_t(size);
		if (!_writeContext(cFile, reinterpret_cast<const char *>(&sz), sizeof(sz))) return false;
		if (!_writeContext(cFile, reinterpret_cast<const char *>(&_k), sizeof(_k))) return false;
		if (!_writeContext(cFile, reinterpret_cast<const char *>(&_n), sizeof(_n))) return false;

		if (!_writeContext(cFile, reinterpret_cast<const char *>(&i), sizeof(i))) return false;

		_engine.readMemory_x(mem);
		if (!_writeContext(cFile, reinterpret_cast<const char *>(mem), sizeof(cl_uint2) * size / 2)) return false;
		_engine.readMemory_u(mem);
		if (!_writeContext(cFile, reinterpret_cast<const char *>(mem), sizeof(cl_uint2) * size / 2)) return false;
		_engine.readMemory_v(mem);
		if (!_writeContext(cFile, reinterpret_cast<const char *>(mem), sizeof(cl_uint2) * size / 2)) return false;

		cFile.close();
		return true;
	}

public:
	bool restoreContext(uint32_t & i, double & elapsedTime)
	{
		std::ifstream cFile("proth.ctx", std::ios::binary);
		if (!cFile.is_open()) return false;

		const size_t size = _size;
		cl_uint2 * const mem = _mem;
		for (size_t k = 0; k < size; ++k) mem[k] = set2(0, 0);	// read size / 2, the upper part must be zero

		uint32_t version = 0;
		if (!_readContext(cFile, reinterpret_cast<char *>(&version), sizeof(version))) return false;
		if (version != 0) return false;
		if (!_readContext(cFile, reinterpret_cast<char *>(&elapsedTime), sizeof(elapsedTime))) return false;
		uint32_t digit_bit = 0;
		if (!_readContext(cFile, reinterpret_cast<char *>(&digit_bit), sizeof(digit_bit))) return false;
		if (digit_bit != uint32_t(_digit_bit)) return false;
		uint32_t sz = 0;
		if (!_readContext(cFile, reinterpret_cast<char *>(&sz), sizeof(sz))) return false;
		if (sz != uint32_t(size)) return false;
		uint32_t k = 0;
		if (!_readContext(cFile, reinterpret_cast<char *>(&k), sizeof(k))) return false;
		if (k != _k) return false;
		uint32_t n = 0;
		if (!_readContext(cFile, reinterpret_cast<char *>(&n), sizeof(n))) return false;
		if (n != _n) return false;

		if (!_readContext(cFile, reinterpret_cast<char *>(&i), sizeof(i))) return false;

		if (!_readContext(cFile, reinterpret_cast<char *>(mem), sizeof(cl_uint2) * size / 2)) return false;
		_engine.writeMemory_x(mem);
		if (!_readContext(cFile, reinterpret_cast<char *>(mem), sizeof(cl_uint2) * size / 2)) return false;
		_engine.writeMemory_u(mem);
		if (!_readContext(cFile, reinterpret_cast<char *>(mem), sizeof(cl_uint2) * size / 2)) return false;
		_engine.writeMemory_v(mem);

		cFile.close();
		return true;
	}

public:
	void init(const uint32_t a)
	{
		const size_t size = _size;

		cl_uint2 * const x = _mem;
		x[0] = set2(1, 0);
		for (size_t i = 1; i < size; ++i) x[i] = set2(0, 0);
		_engine.writeMemory_x(x);

		cl_uint2 * const u = _mem;
		u[0] = set2(a, 0);
		for (size_t i = 1; i < size; ++i) u[i] = set2(0, 0);
		_engine.writeMemory_u(u);
	}

public:
	void initProfiling()
	{
		const size_t size = _size;

		cl_uint2 * const x = _mem;
		for (size_t i = 0; i < size / 2; ++i) x[i] = set2((uint32_t(1) << _digit_bit) - 1, 0);
		for (size_t i = size / 2; i < size; ++i) x[i] = set2(0, 0);
		_engine.writeMemory_x(x);

		cl_uint2 * const u = _mem;
		for (size_t i = 0 * size / 4; i < 1 * size / 4; ++i) u[i] = set2((uint32_t(1) << _digit_bit) - 1, 0);
		for (size_t i = 1 * size / 4; i < 2 * size / 4; ++i) u[i] = set2(0, (uint32_t(1) << _digit_bit) - 1);
		for (size_t i = size / 2; i < size; ++i) u[i] = set2(0, 0);
		_engine.writeMemory_u(u);
}

public:
	void set_bug()
	{
		cl_uint2 * const x = _mem;
		_engine.readMemory_x(x);
		x[_size / 3].s[0] += 1;
		_engine.writeMemory_x(x);
	}

public:
	void norm()
	{
		// if R < Y then add k.2^n + 1 to R. We have 0 <= R - Y + k.2^n + 1 <= k.2^n
		_engine.set_positive();

		// _x[0] = R, _x[1] = Y; compute R - Y
		_engine.reduce_x();
	}

public:
	void swap_x_u() { _engine.swap_x_u(); }
	void copy_x_u() { _engine.copy_x_u(); }
	void copy_x_v() { _engine.copy_x_v(); }
	void compare_u_v() { _engine.compare_u_v(); }

public:
	void square()
	{
		// x size is size / 2; _x[0] = R, _x[1] = Y; compute (R - Y)^2

		_plan.execSquareSeq(_engine);

		_engine.poly2int();

		// x size is size

		split();

		// Now x size is size / 2, _x[0] = R, _x[1] = Y such that X = R - Y and -k.2^n < R - Y < k.2^n
	}

public:
	void setMultiplicand()
	{
		_engine.copy_u_tu();
		_engine.set_positive_tu();

		_engine.sub_ntt64_u();

		cl_uint m = cl_uint(_size / 4);
		cl_uint rindex = (16 + 4 + 1) * (m / 16);
		m /= 64;

		for (; m > 256; m /= 64)
		{
			_engine.ntt64_u(m / 16, rindex);
			rindex += (16 + 4 + 1) * (m / 16);
		}

		for (; m > 1; m /= 4)
		{
			_engine.ntt4_u(m, rindex);
			rindex += m;
		}
	}

public:
	void mul()
	{
		const size_t size = _size;

		// if R - Y < 0 then the result a * (R - Y) < 0 => error
		// if R < Y then add k.2^n + 1 to R. We have 0 < R - Y + k.2^n + 1 <= k.2^n
		_engine.set_positive();

		_engine.sub_ntt64(0, 0);

		cl_uint m = cl_uint(size / 4);
		cl_uint rindex = (16 + 4 + 1) * (m / 16);
		m /= 64;

		for (; m > 256; m /= 64)
		{
			_engine.ntt64(m / 16, rindex);
			rindex += (16 + 4 + 1) * (m / 16);
		}

		size_t n4 = 0;
		for (; m > 4; m /= 4)
		{
			_engine.ntt4(m, rindex);
			rindex += m;
			++n4;
		}

		_engine.ntt4(m, rindex);
		if (m == 4) _engine.mul4(); else _engine.mul2();
		_engine.intt4(m, rindex);

		for (; n4 != 0; --n4)
		{
			m *= 4;
			rindex -= m;
			_engine.intt4(m, rindex);
		}

		while (m < cl_uint(size / 4))
		{
			m *= 64;
			rindex -= (16 + 4 + 1) * (m / 16);
			_engine.intt64(m / 16, rindex);
		}

		_engine.poly2int();

		split();
	}

public:
	bool isMinusOne(uint64_t & res64)
	{
		norm();

		// res is x + 1 such that 0 <= res < k*2^n + 1
		_engine.copy_x_m1();
		_engine.add1_m1();
		_engine.reduce_z_m1();

		cl_uint2 * const res = _mem;

		_engine.readMemory_m1(res);

		bool isPrime = true;
		for (size_t i = 0, n = _size / 2; i < n; ++i)
		{
			isPrime &= (res[i].s[0] == 0);
		}

		uint64_t r = 0, b = 1;
		for (size_t i = 0; b != 0; ++i)
		{
			r += res[i].s[0] * b;
			b <<= _digit_bit;
		}

		res64 = r;
		return isPrime;
	}

public:
	void Gerbicz_step()
	{
		// u *= x;
		swap_x_u();
		setMultiplicand();
		mul();
		swap_x_u();
	}

public:
	void Gerbicz_check(const size_t L)
	{
		// v * u^(2^L)
		_engine.copy_u_m1();		// m1 = u
		_engine.copy_x_m2();		// m1 = u, m2 = x
		_engine.copy_u_x();
		for (size_t i = 0; i < L; ++i) square();	// x = u^(2^L)
		_engine.copy_v_u();
		setMultiplicand();
		mul();			// x = v * u^(2^L)
		norm();
		_engine.swap_x_m2();		// m1 = u, m2 = v * u^(2^L)
		_engine.copy_m1_u();		// m2 = v * u^(2^L)

		// u * x;
		_engine.copy_x_m1();		// m1 = x
		setMultiplicand();
		mul();
		norm();
		_engine.swap_x_m1();		// m1 = u * x, m2 = v * u^(2^L)

		_engine.compare_m1_m2();
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

		_engine.reduce_i();

		// x size is size / 2, x = X mod B^(size / 2), y = X / (B^e * 2^s)

		const cl_uint n = cl_uint(_size / 2);
		cl_uint j = 4;		// alignment (cl_uint4)
		cl_uint s = n / 4;
		for (; s > 256; s /= 64)
		{
			_engine.reduce_upsweep64(s / 16, j);
			j += 5 * (16 + 4 + 1) * (s / 16);
		}

		if (s == 256)        _engine.reduce_topsweep1024(j);
		else if (s == 128)   _engine.reduce_topsweep512(j);
		else if (s == 64)    _engine.reduce_topsweep256(j);
		else if (s == 32)    _engine.reduce_topsweep128(j);
		else if (s == 16)    _engine.reduce_topsweep64(j);
		else /*if (s == 8)*/ _engine.reduce_topsweep32(j);

		while (s < n / 4)
		{
			s *= 64;
			j -= 5 * (16 + 4 + 1) * (s / 16);
			_engine.reduce_downsweep64(s / 16, j);
		}

		// t[4 + k] remainders y[k + 1] / d, t[0] = remainder y[0] / d

		_engine.reduce_o();

		// x size is size / 2, x[0] = X mod B^n, x[1] = Y

		_engine.reduce_f();

		// x size is size / 2, _x[0] = R, _x[1] = Y
	}
};
