/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "ocl.h"

class engine : public ocl::device
{
private:
	size_t _size = 0, _constant_size = 0;
	cl_mem _x = nullptr, _y = nullptr, _t = nullptr, _cr = nullptr, _u = nullptr, _tu = nullptr, _v = nullptr, _m1 = nullptr, _m2 = nullptr, _err = nullptr;
	cl_mem _r1ir1 = nullptr, _r2 = nullptr, _ir2 = nullptr, _cr1ir1 = nullptr, _cr2ir2 = nullptr, _bp = nullptr, _ibp = nullptr;
	cl_kernel _sub_ntt64 = nullptr, _ntt64 = nullptr, _intt64 = nullptr, _sub_ntt256 = nullptr, _ntt256 = nullptr, _intt256 = nullptr;
	cl_kernel _sub_ntt1024 = nullptr, _ntt1024 = nullptr, _intt1024 = nullptr;
	cl_kernel _square8 = nullptr, _square16 = nullptr, _square32 = nullptr, _square64 = nullptr, _square128 = nullptr, _square256 = nullptr;
	cl_kernel _square512 = nullptr, _square1024 = nullptr, _square2048 = nullptr, _square4096 = nullptr;
	cl_kernel _poly2int0 = nullptr, _poly2int1 = nullptr;
	cl_kernel _reduce_upsweep64 = nullptr, _reduce_downsweep64 = nullptr;
	cl_kernel _reduce_topsweep32 = nullptr, _reduce_topsweep64 = nullptr, _reduce_topsweep128 = nullptr;
	cl_kernel _reduce_topsweep256 = nullptr, _reduce_topsweep512 = nullptr, _reduce_topsweep1024 = nullptr;
	cl_kernel _reduce_i = nullptr, _reduce_o = nullptr, _reduce_f = nullptr, _reduce_x = nullptr, _reduce_z = nullptr;
	cl_kernel _ntt4 = nullptr, _intt4 = nullptr, _mul2 = nullptr, _mul4 = nullptr;
	cl_kernel _set_positive = nullptr, _add1 = nullptr, _swap = nullptr, _copy = nullptr, _compare = nullptr;

	// Must be identical to ocl defines
	static const size_t CHUNK64 = 16, CHUNK256 = 8, CHUNK1024 = 4;
	static const size_t BLK8 = 32, BLK16 = 16, BLK32 = 8, BLK64 = 4, BLK128 = 2, BLK256 = 1;
	static const size_t P2I_WGS = 16, P2I_BLK = 16, RED_BLK = 4;

public:
	engine(const ocl::platform & platform, const size_t d) : ocl::device(platform, d) {}
	virtual ~engine() {}

public:
	void allocMemory(const size_t size, const size_t constant_size)
	{
#if defined (ocl_debug)
		std::cerr << "Alloc gpu memory." << std::endl;
#endif
		_size = size;
		_x = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint2) * size);				// main buffer, square & mul multiplier, NTT => size
		_y = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint) * (size / 2));		// reduce
		_t = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint) * 2 * (size / 2));	// reduce: division algorithm
		_cr = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_long) * size / P2I_BLK);	// carry
		_u = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint2) * size);				// mul multiplicand, NTT => size. d(t) in Gerbicz error checking
		_tu = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint2) * size);			// NTT of mul multiplicand
		_v = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint2) * size / 2);			// u(0) in Gerbicz error checking
		_m1 = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint2) * (size / 2));		// memory register #1
		_m2 = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint2) * (size / 2));		// memory register #2
		_err = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_int));					// error checking

		_r1ir1 = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint4) * size);			// NTT roots
		_r2 = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint2) * size);				// NTT roots (square)
		_ir2 = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint2) * size);			// NTT roots (inverse square)
		_bp = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint) * size / 2);			// b^i mod k (division algorithm)
		_ibp = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint) * size / 2);			// (1/b)^(i+1) mod k (division algorithm)

		_constant_size = constant_size;
		_cr1ir1 = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint4) * constant_size);	// small NTT roots: squaring
		_cr2ir2 = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint4) * constant_size);	// small NTT roots (square): squaring

		// allocated size ~ (1 * 4 + 5 * 2 + 4 * 1 + 3 * 1/2) * sizeof(cl_uint) * size = 78 * size bytes
	}

public:
	void releaseMemory()
	{
#if defined (ocl_debug)
		std::cerr << "Free gpu memory." << std::endl;
#endif
		if (_size != 0)
		{
			_releaseBuffer(_x); _releaseBuffer(_y); _releaseBuffer(_t); _releaseBuffer(_cr); _releaseBuffer(_u); _releaseBuffer(_tu);
			_releaseBuffer(_v); _releaseBuffer(_m1); _releaseBuffer(_m2); _releaseBuffer(_err);
			_releaseBuffer(_r1ir1); _releaseBuffer(_r2); _releaseBuffer(_ir2); _releaseBuffer(_bp); _releaseBuffer(_ibp);
			_size = 0;
		}

		if (_constant_size != 0)
		{
			_releaseBuffer(_cr1ir1); _releaseBuffer(_cr2ir2);
			_constant_size = 0;
		}
	}

private:
	inline cl_kernel _createNttKernel(const char * const kernelName, const bool forward)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_x);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_r1ir1);
		_setKernelArg(kernel, 2, sizeof(cl_mem), forward ? &_r2 : &_ir2);
		return kernel;
	}

private:
	inline cl_kernel _createSquareKernel(const char * const kernelName)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_x);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_cr1ir1);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_cr2ir2);
		return kernel;
	}

private:
	inline cl_kernel _createReduceKernel(const char * const kernelName, const bool forward,
		const cl_uint e, const cl_int s, const cl_uint d, const cl_uint d_inv, const cl_int d_shift)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_x);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_y);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_t);
		_setKernelArg(kernel, 3, sizeof(cl_mem), forward ? &_bp : &_ibp);
		const cl_uint4 e_d_d_inv_d_shift = { e, d, d_inv, cl_uint(d_shift) };
		_setKernelArg(kernel, 4, sizeof(cl_uint4), &e_d_d_inv_d_shift);
		if (forward) _setKernelArg(kernel, 5, sizeof(cl_int), &s);
		return kernel;
	}

private:
	inline cl_kernel _createSweepKernel(const char * const kernelName, const cl_uint d)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_t);
		_setKernelArg(kernel, 1, sizeof(cl_uint), &d);
		return kernel;
	}

public:
	void createKernels(const cl_uint2 norm, const cl_uint e, const cl_int s, const cl_uint d, const cl_uint d_inv, const cl_int d_shift, const bool ext1024)
	{
#if defined (ocl_debug)
		std::cerr << "Create ocl kernels." << std::endl;
#endif
		const cl_uint n = cl_uint(_size / 2);
		const cl_ulong ds = cl_ulong(d) << s;

		_sub_ntt64 = _createNttKernel("sub_ntt64", true);
		_ntt64 = _createNttKernel("ntt64", true);
		_intt64 = _createNttKernel("intt64", false);
		if (ext1024)
		{
			_sub_ntt256 = _createNttKernel("sub_ntt256", true);
			_ntt256 = _createNttKernel("ntt256", true);
			_intt256 = _createNttKernel("intt256", false);
			_sub_ntt1024 = _createNttKernel("sub_ntt1024", true);
			_ntt1024 = _createNttKernel("ntt1024", true);
			_intt1024 = _createNttKernel("intt1024", false);
		}

		_square8 = _createSquareKernel("square8");
		_square16 = _createSquareKernel("square16");
		_square32 = _createSquareKernel("square32");
		_square64 = _createSquareKernel("square64");
		_square128 = _createSquareKernel("square128");
		_square256 = _createSquareKernel("square256");
		_square512 = _createSquareKernel("square512");
		_square1024 = _createSquareKernel("square1024");
		if (ext1024)
		{
			_square2048 = _createSquareKernel("square2048");
			_square4096 = _createSquareKernel("square4096");
		}

		_poly2int0 = _createKernel("poly2int0");
		_setKernelArg(_poly2int0, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_poly2int0, 1, sizeof(cl_mem), &_cr);
		_setKernelArg(_poly2int0, 2, sizeof(cl_uint2), &norm);

		_poly2int1 = _createKernel("poly2int1");
		_setKernelArg(_poly2int1, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_poly2int1, 1, sizeof(cl_mem), &_cr);
		_setKernelArg(_poly2int1, 2, sizeof(cl_mem), &_err);

		_reduce_upsweep64 = _createSweepKernel("reduce_upsweep64", d);
		_reduce_downsweep64 = _createSweepKernel("reduce_downsweep64", d);

		_reduce_topsweep32 = _createSweepKernel("reduce_topsweep32", d);
		_reduce_topsweep64 = _createSweepKernel("reduce_topsweep64", d);
		_reduce_topsweep128 = _createSweepKernel("reduce_topsweep128", d);
		_reduce_topsweep256 = _createSweepKernel("reduce_topsweep256", d);
		_reduce_topsweep512 = _createSweepKernel("reduce_topsweep512", d);
		_reduce_topsweep1024 = _createSweepKernel("reduce_topsweep1024", d);

		_reduce_i = _createReduceKernel("reduce_i", true, e, s, d, d_inv, d_shift);
		_reduce_o = _createReduceKernel("reduce_o", false, e, s, d, d_inv, d_shift);

		_reduce_f = _createKernel("reduce_f");
		_setKernelArg(_reduce_f, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_reduce_f, 1, sizeof(cl_mem), &_t);
		_setKernelArg(_reduce_f, 2, sizeof(cl_uint), &n);
		_setKernelArg(_reduce_f, 3, sizeof(cl_uint), &e);
		_setKernelArg(_reduce_f, 4, sizeof(cl_int), &s);

		_reduce_x = _createKernel("reduce_x");
		_setKernelArg(_reduce_x, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_reduce_x, 1, sizeof(cl_uint), &n);
		_setKernelArg(_reduce_x, 2, sizeof(cl_mem), &_err);

		_reduce_z = _createKernel("reduce_z");
		_setKernelArg(_reduce_z, 0, sizeof(cl_mem), &_m1);
		_setKernelArg(_reduce_z, 1, sizeof(cl_uint), &n);
		_setKernelArg(_reduce_z, 2, sizeof(cl_mem), &_err);

		_ntt4 = _createNttKernel("ntt4", true);
		_intt4 = _createNttKernel("intt4", false);

		_mul2 = _createKernel("mul2");
		_setKernelArg(_mul2, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_mul2, 1, sizeof(cl_mem), &_tu);

		_mul4 = _createKernel("mul4");
		_setKernelArg(_mul4, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_mul4, 1, sizeof(cl_mem), &_tu);

		_set_positive = _createKernel("set_positive");
		_setKernelArg(_set_positive, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_set_positive, 1, sizeof(cl_uint), &n);
		_setKernelArg(_set_positive, 2, sizeof(cl_uint), &e);
		_setKernelArg(_set_positive, 3, sizeof(cl_ulong), &ds);

		_add1 = _createKernel("add1");
		_setKernelArg(_add1, 0, sizeof(cl_mem), &_m1);
		_setKernelArg(_add1, 1, sizeof(cl_uint), &e);
		_setKernelArg(_add1, 2, sizeof(cl_ulong), &ds);

		_swap = _createKernel("swap");
		_copy = _createKernel("copy");
		_compare = _createKernel("compare");
		_setKernelArg(_compare, 2, sizeof(cl_mem), &_err);
	}

public:
	void releaseKernels()
	{
#if defined (ocl_debug)
		std::cerr << "Release ocl kernels." << std::endl;
#endif
		_releaseKernel(_sub_ntt64); _releaseKernel(_ntt64); _releaseKernel(_intt64);
		_releaseKernel(_sub_ntt256); _releaseKernel(_ntt256); _releaseKernel(_intt256);
		_releaseKernel(_sub_ntt1024); _releaseKernel(_ntt1024); _releaseKernel(_intt1024);

		_releaseKernel(_square8); _releaseKernel(_square16); _releaseKernel(_square32); _releaseKernel(_square64); _releaseKernel(_square128);
		_releaseKernel(_square256); _releaseKernel(_square512); _releaseKernel(_square1024); _releaseKernel(_square2048); _releaseKernel(_square4096);

		_releaseKernel(_poly2int0); _releaseKernel(_poly2int1);

		_releaseKernel(_reduce_upsweep64); _releaseKernel(_reduce_downsweep64);

		_releaseKernel(_reduce_topsweep32); _releaseKernel(_reduce_topsweep64); _releaseKernel(_reduce_topsweep128);
		_releaseKernel(_reduce_topsweep256); _releaseKernel(_reduce_topsweep512); _releaseKernel(_reduce_topsweep1024);

		_releaseKernel(_reduce_i); _releaseKernel(_reduce_o); _releaseKernel(_reduce_f); _releaseKernel(_reduce_x); _releaseKernel(_reduce_z);

		_releaseKernel(_ntt4); _releaseKernel(_intt4); _releaseKernel(_mul2); _releaseKernel(_mul4);
		_releaseKernel(_set_positive); _releaseKernel(_add1);

		_releaseKernel(_swap); _releaseKernel(_copy); _releaseKernel(_compare);
	}

public:
	// read half the size
	void readMemory_x(cl_uint2 * const ptr) { _readBuffer(_x, ptr, sizeof(cl_uint2) * _size / 2); }
	void readMemory_u(cl_uint2 * const ptr) { _readBuffer(_u, ptr, sizeof(cl_uint2) * _size / 2); }
	// write full size
	void writeMemory_x(const cl_uint2 * const ptr) { _writeBuffer(_x, ptr, sizeof(cl_uint2) * _size); }
	void writeMemory_u(const cl_uint2 * const ptr) { _writeBuffer(_u, ptr, sizeof(cl_uint2) * _size); }

	void readMemory_v(cl_uint2 * const ptr) { _readBuffer(_v, ptr, sizeof(cl_uint2) * _size / 2); }
	void writeMemory_v(const cl_uint2 * const ptr) { _writeBuffer(_v, ptr, sizeof(cl_uint2) * _size / 2); }

	void readMemory_m1(cl_uint2 * const ptr) { _readBuffer(_m1, ptr, sizeof(cl_uint2) * _size / 2); }

	void readMemory_err(cl_int * const ptr) { _readBuffer(_err, ptr, sizeof(cl_int)); }
	void writeMemory_err(const cl_int * const ptr) { _writeBuffer(_err, ptr, sizeof(cl_int)); }

public:
	void writeMemory_r(const cl_uint4 * const ptr_r1ir1, const cl_uint2 * const ptr_r2, const cl_uint2 * const ptr_ir2)
	{
		_writeBuffer(_r1ir1, ptr_r1ir1, sizeof(cl_uint4) * _size);
		_writeBuffer(_r2, ptr_r2, sizeof(cl_uint2) * _size);
		_writeBuffer(_ir2, ptr_ir2, sizeof(cl_uint2) * _size);
	}

public:
	void writeMemory_cr(const cl_uint4 * const ptr_cr1ir1, const cl_uint4 * const ptr_cr2ir2)
	{
		_writeBuffer(_cr1ir1, ptr_cr1ir1, sizeof(cl_uint4) * _constant_size);
		_writeBuffer(_cr2ir2, ptr_cr2ir2, sizeof(cl_uint4) * _constant_size);
	}

public:
	void writeMemory_bp(const cl_uint * const ptr_bp, const cl_uint * const ptr_ibp)
	{
		_writeBuffer(_bp, ptr_bp, sizeof(cl_uint) * _size / 2);
		_writeBuffer(_ibp, ptr_ibp, sizeof(cl_uint) * _size / 2);
	}

public:
	void sub_ntt64(const cl_uint, const cl_uint) { _executeKernel(_sub_ntt64, _size / 4, CHUNK64 * (64 / 4)); }
	void sub_ntt256(const cl_uint, const cl_uint) { _executeKernel(_sub_ntt256, _size / 4, CHUNK256 * (256 / 4)); }
	void sub_ntt1024(const cl_uint, const cl_uint) { _executeKernel(_sub_ntt1024, _size / 4, CHUNK1024 * (1024 / 4)); }

private:
	inline void _executeNttKernel(cl_kernel kernel, const cl_uint m, const cl_uint rindex, const size_t size)
	{
		_setKernelArg(kernel, 3, sizeof(cl_uint), &m);
		_setKernelArg(kernel, 4, sizeof(cl_uint), &rindex);
		_executeKernel(kernel, _size / 4, size);
	}

public:
	void ntt64(const cl_uint m, const cl_uint rindex) { _executeNttKernel(_ntt64, m, rindex, CHUNK64 * (64 / 4)); }
	void intt64(const cl_uint m, const cl_uint rindex) { _executeNttKernel(_intt64, m, rindex, CHUNK64 * (64 / 4)); }
	void ntt256(const cl_uint m, const cl_uint rindex) { _executeNttKernel(_ntt256, m, rindex, CHUNK256 * (256 / 4)); }
	void intt256(const cl_uint m, const cl_uint rindex) { _executeNttKernel(_intt256, m, rindex, CHUNK256 * (256 / 4)); }
	void ntt1024(const cl_uint m, const cl_uint rindex) { _executeNttKernel(_ntt1024, m, rindex, CHUNK1024 * (1024 / 4)); }
	void intt1024(const cl_uint m, const cl_uint rindex) { _executeNttKernel(_intt1024, m, rindex, CHUNK1024 * (1024 / 4)); }
	void ntt4(const cl_uint m, const cl_uint rindex) { _executeNttKernel(_ntt4, m, rindex, 0); }
	void intt4(const cl_uint m, const cl_uint rindex) { _executeNttKernel(_intt4, m, rindex, 0); }

public:
	void sub_ntt64_u()
	{
		_setKernelArg(_sub_ntt64, 0, sizeof(cl_mem), &_tu);
		_executeKernel(_sub_ntt64, _size / 4, CHUNK64 * (64 / 4));
		_setKernelArg(_sub_ntt64, 0, sizeof(cl_mem), &_x);
	}

public:
	void ntt64_u(const cl_uint m, const cl_uint rindex)
	{
		_setKernelArg(_ntt64, 0, sizeof(cl_mem), &_tu);
		_executeNttKernel(_ntt64, m, rindex, CHUNK64 * (64 / 4));
		_setKernelArg(_ntt64, 0, sizeof(cl_mem), &_x);
	}

public:
	void ntt4_u(const cl_uint m, const cl_uint rindex)
	{
		_setKernelArg(_ntt4, 0, sizeof(cl_mem), &_tu);
		_executeNttKernel(_ntt4, m, rindex, CHUNK64 * (64 / 4));
		_setKernelArg(_ntt4, 0, sizeof(cl_mem), &_x);
	}

public:
	void square8(const cl_uint, const cl_uint) { _executeKernel(_square8, _size / 4, BLK8 * 8 / 4); }
	void square16(const cl_uint, const cl_uint) { _executeKernel(_square16, _size / 4, BLK16 * 16 / 4); }
	void square32(const cl_uint, const cl_uint) { _executeKernel(_square32, _size / 4, BLK32 * 32 / 4); }
	void square64(const cl_uint, const cl_uint) { _executeKernel(_square64, _size / 4, BLK64 * 64 / 4); }
	void square128(const cl_uint, const cl_uint) { _executeKernel(_square128, _size / 4, BLK128 * 128 / 4); }
	void square256(const cl_uint, const cl_uint) { _executeKernel(_square256, _size / 4, BLK256 * 256 / 4); }
	void square512(const cl_uint, const cl_uint) { _executeKernel(_square512, _size / 4, 512 / 4); }
	void square1024(const cl_uint, const cl_uint) { _executeKernel(_square1024, _size / 4, 1024 / 4); }
	void square2048(const cl_uint, const cl_uint) { _executeKernel(_square2048, _size / 4, 2048 / 4); }
	void square4096(const cl_uint, const cl_uint) { _executeKernel(_square4096, _size / 4, 4096 / 4); }

public:
	void mul2() { _executeKernel(_mul2, _size / 4); }
	void mul4() { _executeKernel(_mul4, _size / 4); }

public:
	void poly2int() { _executeKernel(_poly2int0, _size / P2I_BLK, P2I_WGS); _executeKernel(_poly2int1, _size / P2I_BLK); }

private:
	inline void _executeUDsweepKernel(cl_kernel kernel, const cl_uint s, const cl_uint j, const size_t size)
	{
		_setKernelArg(kernel, 2, sizeof(cl_uint), &s);
		_setKernelArg(kernel, 3, sizeof(cl_uint), &j);
		_executeKernel(kernel, (size / 4) * s, RED_BLK * (size / 4));
	}

public:
	void reduce_upsweep64(const cl_uint s, const cl_uint j) { _executeUDsweepKernel(_reduce_upsweep64, s, j, 64); }
	void reduce_downsweep64(const cl_uint s, const cl_uint j) { _executeUDsweepKernel(_reduce_downsweep64, s, j, 64); }

private:
	inline void _executeTopsweepKernel(cl_kernel kernel, const cl_uint j, const size_t size)
	{
		_setKernelArg(kernel, 2, sizeof(cl_uint), &j);
		_executeKernel(kernel, size / 4, size / 4);
	}

public:
	void reduce_topsweep32(const cl_uint j) { _executeTopsweepKernel(_reduce_topsweep32, j, 32); }
	void reduce_topsweep64(const cl_uint j) { _executeTopsweepKernel(_reduce_topsweep64, j, 64); }
	void reduce_topsweep128(const cl_uint j) { _executeTopsweepKernel(_reduce_topsweep128, j, 128); }
	void reduce_topsweep256(const cl_uint j) { _executeTopsweepKernel(_reduce_topsweep256, j, 256); }
	void reduce_topsweep512(const cl_uint j) { _executeTopsweepKernel(_reduce_topsweep512, j, 512); }
	void reduce_topsweep1024(const cl_uint j) { _executeTopsweepKernel(_reduce_topsweep1024, j, 1024); }

public:
	void reduce_i() { _executeKernel(_reduce_i, _size / 2); }
	void reduce_o() { _executeKernel(_reduce_o, _size / 2); }
	void reduce_f() { _executeKernel(_reduce_f, 1); }
	void reduce_x() { _executeKernel(_reduce_x, 1); }
	void reduce_z_m1() { _executeKernel(_reduce_z, 1); }

public:
	void set_positive() { _executeKernel(_set_positive, 1); }
	void add1_m1() { _executeKernel(_add1, 1); }

public:
	void set_positive_tu()
	{
		_setKernelArg(_set_positive, 0, sizeof(cl_mem), &_tu);
		_executeKernel(_set_positive, 1);
		_setKernelArg(_set_positive, 0, sizeof(cl_mem), &_x);
	}

private:
	void _executeSwapKernel(const void * const arg_x, const void * const arg_y)
	{
		_setKernelArg(_swap, 0, sizeof(cl_mem), arg_x);
		_setKernelArg(_swap, 1, sizeof(cl_mem), arg_y);
		_executeKernel(_swap, _size / 2);
	}

public:
	void swap_x_u() { _executeSwapKernel(&_x, &_u); }
	void swap_x_m1() { _executeSwapKernel(&_x, &_m1); }
	void swap_x_m2() { _executeSwapKernel(&_x, &_m2); }

private:
	void _executeCopyKernel(const void * const arg_x, const void * const arg_y)
	{
		_setKernelArg(_copy, 0, sizeof(cl_mem), arg_x);
		_setKernelArg(_copy, 1, sizeof(cl_mem), arg_y);
		_executeKernel(_copy, _size / 2);
	}

public:
	void copy_x_u() { _executeCopyKernel(&_u, &_x); }
	void copy_x_v() { _executeCopyKernel(&_v, &_x); }
	void copy_x_m1() { _executeCopyKernel(&_m1, &_x); }
	void copy_x_m2() { _executeCopyKernel(&_m2, &_x); }
	void copy_u_x() { _executeCopyKernel(&_x, &_u); }
	void copy_u_m1() { _executeCopyKernel(&_m1, &_u); }
	void copy_u_tu() { _executeCopyKernel(&_tu, &_u); }
	void copy_v_u() { _executeCopyKernel(&_u, &_v); }
	void copy_m1_u() { _executeCopyKernel(&_u, &_m1); }

private:
	void _executeCompareKernel(const void * const arg_x, const void * const arg_y)
	{
		_setKernelArg(_compare, 0, sizeof(cl_mem), arg_x);
		_setKernelArg(_compare, 1, sizeof(cl_mem), arg_y);
		_executeKernel(_compare, _size / 2);
	}

public:
	void compare_u_v() { _executeCompareKernel(&_u, &_v); }
	void compare_m1_m2() { _executeCompareKernel(&_m1, &_m2); }
};
