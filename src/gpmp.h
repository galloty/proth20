/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "arith.h"
#include "engine.h"
#include "pio.h"
#include "plan.h"

#include <cstdint>
#include <cmath>
#include <sstream>
#include <vector>

#include "ocl/modarith.h"
#include "ocl/NTT.h"
#include "ocl/square.h"
#include "ocl/poly2int.h"
#include "ocl/reduce.h"
#include "ocl/misc.h"
#include "ocl/squareNTT_512.h"
#include "ocl/squareNTT_1024.h"

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
	const bool _isBoinc;
	bool _ext512, _ext1024;
	engine & _engine;
	plan _plan;
	std::vector<cl_uint2> _mem;

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
		uint32_t get1p() const { return cl_uint((uint64_t(n1) << 32) / P1); }
		uint32_t get2p() const { return cl_uint((uint64_t(n2) << 32) / P2); }

		RNS & operator*=(const RNS & rhs) { n1 *= rhs.n1; n2 *= rhs.n2; return *this; }
		RNS operator*(const RNS & rhs) const { RNS r = *this; r *= rhs; return r; }

		RNS invert() const { return RNS(n1.invert(), n2.invert()); }
		static RNS prRoot(const size_t n) { return RNS(Zp<P1>(P1_PRIM_ROOT).pow((P1 - 1) / n), Zp<P2>(P2_PRIM_ROOT).pow((P2 - 1) / n)); }
	};

private:
	bool readOpenCL(const char * const clFileName, const char * const headerFileName, const char * const varName, std::stringstream & src) const
	{
		if (_isBoinc) return false;

		std::ifstream clFile(clFileName);
		if (!clFile.is_open()) return false;
		
		// if .cl file exists then generate header file
		std::ofstream hFile(headerFileName, std::ios::binary);	// binary: don't convert line endings to `CRLF` 
		if (!hFile.is_open()) throw std::runtime_error("cannot write openCL header file");

		hFile << "/*" << std::endl;
		hFile << "Copyright 2020, Yves Gallot" << std::endl << std::endl;
		hFile << "proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it." << std::endl;
		hFile << "Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful." << std::endl;
		hFile << "*/" << std::endl << std::endl;

		hFile << "static const char * const " << varName << " = \\" << std::endl;

		std::string line;
		while (std::getline(clFile, line))
		{
			hFile << "\"";
			for (char c : line)
			{
				if ((c == '\\') || (c == '\"')) hFile << '\\';
				hFile << c;
			}
			hFile << "\\n\" \\" << std::endl;

			src << line << std::endl;
		}
		hFile << "\"\";" << std::endl;

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
		const size_t constant_size = 1024 + 256 + 64 + 16 + 4;	// 1364 * 4 * sizeof(cl_uint4) = 88576 bytes

		std::stringstream src;
		src << "#define\tdigit_bit\t" << _digit_bit << std::endl << std::endl;

		src << _engine.oclDefines() << std::endl;

		const cl_int k_shift = cl_int(arith::log2(_k) - 1);
		src << "#define\tpconst_size\t" << size << "u" << std::endl;
		src << "#define\tpconst_norm\t(uint2)(" << cl_uint(P1 - (P1 - 1) / size) << "u, " << cl_uint(P2 - (P2 - 1) / size) << "u)" << std::endl;
		src << "#define\tpconst_e\t" << cl_uint(_n / _digit_bit) << "u" << std::endl;
		src << "#define\tpconst_s\t" << cl_int(_n % _digit_bit) << std::endl;
		src << "#define\tpconst_d\t" << cl_uint(_k) << "u" << std::endl;
		src << "#define\tpconst_d_inv\t" << cl_uint((uint64_t(1) << (32 + k_shift)) / _k) << "u" << std::endl;
		src << "#define\tpconst_d_shift\t" << k_shift << std::endl;
		src << std::endl;

		// if xxx.cl file is not found then source is src_ocl_xxx string in src/ocl/xxx.h
		if (!readOpenCL("ocl/modarith.cl", "src/ocl/modarith.h", "src_ocl_modarith", src)) src << src_ocl_modarith;	
		if (!readOpenCL("ocl/NTT.cl", "src/ocl/NTT.h", "src_ocl_NTT", src)) src << src_ocl_NTT;	
		if (!readOpenCL("ocl/square.cl", "src/ocl/square.h", "src_ocl_square", src)) src << src_ocl_square;	
		if (!readOpenCL("ocl/poly2int.cl", "src/ocl/poly2int.h", "src_ocl_poly2int", src)) src << src_ocl_poly2int;	
		if (!readOpenCL("ocl/reduce.cl", "src/ocl/reduce.h", "src_ocl_reduce", src)) src << src_ocl_reduce;	
		if (!readOpenCL("ocl/misc.cl", "src/ocl/misc.h", "src_ocl_misc", src)) src << src_ocl_misc;	

		if (_ext512)
		{
			if (!readOpenCL("ocl/squareNTT_512.cl", "src/ocl/squareNTT_512.h", "src_ocl_squareNTT_512", src)) src << src_ocl_squareNTT_512;	
		}
		if (_ext1024)
		{
			if (!readOpenCL("ocl/squareNTT_1024.cl", "src/ocl/squareNTT_1024.h", "src_ocl_squareNTT_1024", src)) src << src_ocl_squareNTT_1024;	
		}

		_engine.loadProgram(src.str().c_str());

		_engine.allocMemory(size, constant_size);
		_engine.createKernels(_ext512, _ext1024);

		// (size + 2) / 3 roots
		std::vector<cl_uint4> r1ir1(size);
		std::vector<cl_uint2> r2(size), ir2(size);
		std::vector<cl_uint4> cr1(constant_size), cir1(constant_size), cr2(constant_size), cir2(constant_size);
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
					cr1[o + i] = set4(r1.get1(), r1.get2(), r1.get1p(), r1.get2p());
					cir1[o + i] = set4(ir1.get1(), ir1.get2(), ir1.get1p(), ir1.get2p());
					cr2[o + i] = set4(r1sq.get1(), r1sq.get2(), r1sq.get1p(), r1sq.get2p());
					cir2[o + i] = set4(ir1sq.get1(), ir1sq.get2(), ir1sq.get1p(), ir1sq.get2p());
				}

				r1 *= ps; ir1 *= ips;
			}
			ps *= ps; ps *= ps; ips *= ips; ips *= ips;
		}
		_engine.writeMemory_r(r1ir1.data(), r2.data(), ir2.data());
		_engine.writeMemory_cr(cr1.data(), cir1.data(), cr2.data(), cir2.data());

		std::vector<cl_uint> bp(size / 2), ibp(size / 2);
		const uint32_t ib = arith::invert(uint32_t(1) << _digit_bit, _k);
		uint32_t bp_i = 1, ibp_i = ib;
		for (size_t i = 0; i < size / 2; ++i)
		{
			bp[i] = cl_uint(bp_i);
			ibp[i] = cl_uint(ibp_i);
			bp_i = uint32_t((uint64_t(bp_i) << _digit_bit) % _k);
			ibp_i = uint32_t((uint64_t(ibp_i) * ib) % _k);
		}
		_engine.writeMemory_bp(bp.data(), ibp.data());

		_engine.clearMemory_err();
	}

private:
	void _clearEngine()
	{
		_engine.releaseKernels();
		_engine.releaseMemory();
		_engine.clearProgram();
	}

public:
	gpmp(const uint32_t k, const uint32_t n, engine & engine, const bool isBoinc, const bool bestPlan = true, const bool profile = false) :
		_digit_bit(digitBit(k, n)), _size(transformSize(k, n, _digit_bit)), _k(k), _n(n), _isBoinc(isBoinc),
		_ext512(engine.getMaxWorkGroupSize() >= 512), _ext1024((engine.getMaxWorkGroupSize() >= 1024) && (engine.getLocalMemSize() >= 32768)),
		_engine(engine), _mem(_size)
	{
		if (engine.getMaxWorkGroupSize() < 256) throw std::runtime_error("The maximum work-group size must be equal to or greater than 256");

		const size_t size = _size;

		const double max_digit = double((uint32_t(1) << _digit_bit) - 1);
		if ((size / 2) * max_digit * max_digit >= P1P2 / 2)
		{
			std::stringstream ss; ss << getDigits() << "-digit numbers are not supported";
			throw std::runtime_error(ss.str());
		}

reset:
		_plan.init(size, _ext512, _ext1024);

		size_t bestSq_i = 0, bestP2i_i = 0;
		if (bestPlan)
		{
			engine.setProfiling(true);
			_initEngine();

			cl_ulong bestSqTime = cl_ulong(-1);
			for (size_t i = 0, cnt = _plan.getSquareSeqCount(); i < cnt; ++i)
			{
				initProfiling();
				_plan.setSquareSeq(size, i);
				try
				{
					for (size_t j = 0; j < 16; ++j) square();
					const cl_ulong time = engine.getProfileTime();
					if (time < bestSqTime)
					{
						bestSqTime = time;
						bestSq_i = i;
					}
				}
				catch (const std::runtime_error & e)
				{
					if (_ext512 == false) throw e;
					// try to fix runtime error
					std::ostringstream ss; ss << "warning: " << e.what() << ", trying to fix it..." << std::endl;
					pio::error(ss.str(), true);
					_ext512 = _ext1024 = false;
					engine.resetProfiles();
					_clearEngine();
					goto reset;
				}

				engine.resetProfiles();
			}
			_plan.setSquareSeq(size, bestSq_i);

			cl_ulong bestP2iTime = cl_ulong(-1);
			for (size_t i = 0, cnt = _plan.getPoly2intCount(); i < cnt; ++i)
			{
				initProfiling();
				_plan.setPoly2intFn(i);
				for (size_t j = 0; j < 16; ++j) square();
				const cl_ulong time = engine.getProfileTime();
				if (time < bestP2iTime)
				{
					bestP2iTime = time;
					bestP2i_i = i;
				}
				engine.resetProfiles();
			}

			_clearEngine();
		}

		engine.setProfiling(profile);
		_initEngine();
		_plan.setSquareSeq(size, bestSq_i);
		_plan.setPoly2intFn(bestP2i_i);
	}

public:
	virtual ~gpmp()
	{
		_clearEngine();
	}

public:
	size_t getSize() const { return _size; }
	size_t getDigitBit() const { return _digit_bit; }
	size_t getDigits() const { return size_t(std::ceil(std::log10(_k) + _n * std::log10(2))); }

public:
	std::string getPlanString() const { return _plan.getPlanString(_size); }
	size_t getPlanSquareSeqCount() const { return _plan.getSquareSeqCount(); }
	void setPlanSquareSeq(const size_t i) { _plan.setSquareSeq(_size, i); }
	size_t getPlanPoly2intCount() const { return _plan.getPoly2intCount(); }
	void setPlanPoly2intFn(const size_t i) { _plan.setPoly2intFn(i); }

public:
	void display()
	{
		const size_t size = _size / 2;
		cl_uint2 * const x = _mem.data();
		_engine.readMemory_x(x);
		std::stringstream ss; ss << std::endl;
		for (size_t i = 0; i < size; ++i)
		{
			if (x[i].s[0] != 0) ss << " " << i << ":0 " << x[i].s[0];
			if (x[i].s[1] != 0) ss << " " << i << ":0 " << x[i].s[1];
		}
		ss << std::endl;
		pio::display(ss.str());
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
		std::stringstream ss;
		for (size_t i = 17; i <= 24; ++i)
		{
			ss << "2^" << i << ": [" << range[i].first << "-" << range[i].second << "]" << std::endl;
		}
		pio::display(ss.str());
	}

public:
	int getError() const
	{
		cl_int err = 0;
		_engine.readMemory_err(&err);
		return int(err);
	}

public:
	void resetError()
	{
		_engine.clearMemory_err();
	}

private:
	static bool _writeContext(FILE * const cFile, const char * const ptr, const size_t size)
	{
		const size_t ret = std::fwrite(ptr , sizeof(char), size, cFile);
		if (ret == size * sizeof(char)) return true;
		std::fclose(cFile);
		return false;
	}

private:
	static bool _readContext(FILE * const cFile, char * const ptr, const size_t size)
	{
		const size_t ret = std::fread(ptr , sizeof(char), size, cFile);
		if (ret == size * sizeof(char)) return true;
		std::fclose(cFile);
		return false;
	}

private:
	static std::string _filename(const char * const ext)
	{
		return std::string("proth_") + std::string(ext) + std::string(".ctx");
	}

public:
	bool saveContext(const uint32_t i, const double elapsedTime, const char * const ext)
	{
		FILE * const cFile = pio::open(_filename(ext).c_str(), "wb");
		if (cFile == nullptr)
		{
			std::ostringstream ss; ss << "cannot write 'proth.ctx' file " << std::endl;
			pio::error(ss.str());
			return false;
		}

		const size_t size = _size;
		cl_uint2 * const mem = _mem.data();

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

		std::fclose(cFile);
		return true;
	}

public:
	bool restoreContext(uint32_t & i, double & elapsedTime, const char * const ext, const bool restore_uv = true)
	{
		FILE * const cFile = pio::open(_filename(ext).c_str(), "rb");
		if (cFile == nullptr) return false;

		const size_t size = _size;
		cl_uint2 * const mem = _mem.data();
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
		if (restore_uv)
		{
			if (!_readContext(cFile, reinterpret_cast<char *>(mem), sizeof(cl_uint2) * size / 2)) return false;
			_engine.writeMemory_u(mem);
			if (!_readContext(cFile, reinterpret_cast<char *>(mem), sizeof(cl_uint2) * size / 2)) return false;
			_engine.writeMemory_v(mem);
		}

		std::fclose(cFile);
		return true;
	}

public:
	void init(const uint32_t x0, const uint32_t u0)
	{
		const size_t size = _size;

		cl_uint2 * const x = _mem.data();
		x[0] = set2(x0, 0);
		for (size_t i = 1; i < size; ++i) x[i] = set2(0, 0);
		_engine.writeMemory_x(x);

		cl_uint2 * const u = _mem.data();
		u[0] = set2(u0, 0);
		for (size_t i = 1; i < size; ++i) u[i] = set2(0, 0);
		_engine.writeMemory_u(u);
	}

public:
	void initProfiling()
	{
		const size_t size = _size;

		cl_uint2 * const x = _mem.data();
		for (size_t i = 0; i < size / 2; ++i) x[i] = set2((uint32_t(1) << _digit_bit) - 1, 0);
		for (size_t i = size / 2; i < size; ++i) x[i] = set2(0, 0);
		_engine.writeMemory_x(x);

		cl_uint2 * const u = _mem.data();
		for (size_t i = 0 * size / 4; i < 1 * size / 4; ++i) u[i] = set2((uint32_t(1) << _digit_bit) - 1, 0);
		for (size_t i = 1 * size / 4; i < 2 * size / 4; ++i) u[i] = set2(0, (uint32_t(1) << _digit_bit) - 1);
		for (size_t i = size / 2; i < size; ++i) u[i] = set2(0, 0);
		_engine.writeMemory_u(u);
}

public:
	void set_bug()
	{
		cl_uint2 * const x = _mem.data();
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
	void swap_x_v() { _engine.swap_x_v(); }
	void copy_x_v() { _engine.copy_x_v(); }
	void copy_v_x() { _engine.copy_v_x(); }
	void compare_x_v() { _engine.compare_x_v(); }

public:
	void square()
	{
		// x size is size / 2; _x[0] = R, _x[1] = Y; compute (R - Y)^2

		_plan.execSquareSeq(_engine);
		_plan.execPoly2intFn(_engine);
		_engine.poly2int_fix();

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

		_engine.sub_ntt64_16(0, 0);

		cl_uint m = cl_uint(size / 4);
		cl_uint rindex = (16 + 4 + 1) * (m / 16);
		m /= 64;

		size_t n64 = 0;
		for (; m > 256; m /= 64)
		{
			_engine.ntt64_16(m / 16, rindex);
			rindex += (16 + 4 + 1) * (m / 16);
			++n64;
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

		for (; n64 != 0; --n64)
		{
			m *= 64;
			rindex -= (16 + 4 + 1) * (m / 16);
			_engine.intt64_16(m / 16, rindex);
		}

		_engine.lst_intt64_16(0, 0);

		_engine.poly2int_16_16();

		split();
	}

public:
	void pow(const uint32_t e)
	{
		norm();

		bool s = false;
		_engine.copy_x_u();
		setMultiplicand();
		for (int b = 0; b < 32; ++b)
		{
			if (s) square();

			if ((e & (uint32_t(1) << (31 - b))) != 0)
			{
				if (s) mul();
				s = true;
			}
		}
	}

public:
	bool isMinusOne(uint64_t & res64)
	{
		norm();

		// res is x + 1 such that 0 <= res < k*2^n + 1
		_engine.copy_x_m1();
		_engine.add1_m1(1);
		_engine.reduce_z_m1();

		cl_uint2 * const res = _mem.data();

		_engine.readMemory_m1(res);

		bool isPrime = true;
		for (size_t i = 0, n = _size / 2; i < n; ++i) isPrime &= (res[i].s[0] == 0);

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
	bool isOne()
	{
		norm();

		_engine.copy_x_m1();
		_engine.add1_m1(0);
		_engine.reduce_z_m1();

		cl_uint2 * const res = _mem.data();

		_engine.readMemory_m1(res);

		bool isOne = (res[0].s[0] == 1);
		if (isOne)
		{
			for (size_t i = 1, n = _size / 2; i < n; ++i) isOne &= (res[i].s[0] == 0);
		}

		return isOne;
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
