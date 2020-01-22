/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#if defined (__APPLE__)
	#include <OpenCL/cl.h>
	#include <OpenCL/cl_ext.h>
#else
	#include <CL/cl.h>
#endif

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

namespace ocl
{

// #define ocl_debug		1
// #define ocl_profile		1
#define ocl_fast_exec		1

#if defined (ocl_profile)
#define	_executeKernel	_executeKernelP
#else
#define	_executeKernel	_executeKernelN
#endif

class oclObject
{
private:
	static const char * errorString(const cl_int & res)
	{
		switch (res)
		{
	#define oclCheck(err) case err: return #err
			oclCheck(CL_SUCCESS);
			oclCheck(CL_DEVICE_NOT_FOUND);
			oclCheck(CL_DEVICE_NOT_AVAILABLE);
			oclCheck(CL_COMPILER_NOT_AVAILABLE);
			oclCheck(CL_MEM_OBJECT_ALLOCATION_FAILURE);
			oclCheck(CL_OUT_OF_RESOURCES);
			oclCheck(CL_OUT_OF_HOST_MEMORY);
			oclCheck(CL_PROFILING_INFO_NOT_AVAILABLE);
			oclCheck(CL_MEM_COPY_OVERLAP);
			oclCheck(CL_IMAGE_FORMAT_MISMATCH);
			oclCheck(CL_IMAGE_FORMAT_NOT_SUPPORTED);
			oclCheck(CL_BUILD_PROGRAM_FAILURE);
			oclCheck(CL_MAP_FAILURE);
			oclCheck(CL_MISALIGNED_SUB_BUFFER_OFFSET);
			oclCheck(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
			oclCheck(CL_INVALID_VALUE);
			oclCheck(CL_INVALID_DEVICE_TYPE);
			oclCheck(CL_INVALID_PLATFORM);
			oclCheck(CL_INVALID_DEVICE);
			oclCheck(CL_INVALID_CONTEXT);
			oclCheck(CL_INVALID_QUEUE_PROPERTIES);
			oclCheck(CL_INVALID_COMMAND_QUEUE);
			oclCheck(CL_INVALID_HOST_PTR);
			oclCheck(CL_INVALID_MEM_OBJECT);
			oclCheck(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
			oclCheck(CL_INVALID_IMAGE_SIZE);
			oclCheck(CL_INVALID_SAMPLER);
			oclCheck(CL_INVALID_BINARY);
			oclCheck(CL_INVALID_BUILD_OPTIONS);
			oclCheck(CL_INVALID_PROGRAM);
			oclCheck(CL_INVALID_PROGRAM_EXECUTABLE);
			oclCheck(CL_INVALID_KERNEL_NAME);
			oclCheck(CL_INVALID_KERNEL_DEFINITION);
			oclCheck(CL_INVALID_KERNEL);
			oclCheck(CL_INVALID_ARG_INDEX);
			oclCheck(CL_INVALID_ARG_VALUE);
			oclCheck(CL_INVALID_ARG_SIZE);
			oclCheck(CL_INVALID_KERNEL_ARGS);
			oclCheck(CL_INVALID_WORK_DIMENSION);
			oclCheck(CL_INVALID_WORK_GROUP_SIZE);
			oclCheck(CL_INVALID_WORK_ITEM_SIZE);
			oclCheck(CL_INVALID_GLOBAL_OFFSET);
			oclCheck(CL_INVALID_EVENT_WAIT_LIST);
			oclCheck(CL_INVALID_EVENT);
			oclCheck(CL_INVALID_OPERATION);
			oclCheck(CL_INVALID_GL_OBJECT);
			oclCheck(CL_INVALID_BUFFER_SIZE);
			oclCheck(CL_INVALID_MIP_LEVEL);
			oclCheck(CL_INVALID_GLOBAL_WORK_SIZE);
			oclCheck(CL_INVALID_PROPERTY);
	#undef oclCheck
			default: return "CL_UNKNOWN_ERROR";
		}
	}

protected:
	static constexpr bool oclError(const cl_int res)
	{
		return (res == CL_SUCCESS);
	}

protected:
	static void oclFatal(const cl_int res)
	{
		if (!oclError(res))
		{
			std::ostringstream ss; ss << "opencl error: " << errorString(res) << ".";
			throw std::runtime_error(ss.str());
		}
	}
};

class engine : oclObject
{
private:
	struct deviceDesc
	{
		cl_platform_id platform_id;
		cl_device_id device_id;
		std::string name;
	};
	std::vector<deviceDesc> _devices;

public:
	engine()
	{
#if defined (ocl_debug)
		std::cerr << "Create ocl engine." << std::endl;
#endif
		cl_uint num_platforms;
		cl_platform_id platforms[64];
		oclFatal(clGetPlatformIDs(64, platforms, &num_platforms));

		for (cl_uint p = 0; p < num_platforms; ++p)
		{
			char platformName[1024]; oclFatal(clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 1024, platformName, nullptr));

			cl_uint num_devices;
			cl_device_id devices[64];
			oclFatal(clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 64, devices, &num_devices));

			for (cl_uint d = 0; d < num_devices; ++d)
			{
				char deviceName[1024]; oclFatal(clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 1024, deviceName, nullptr));
				char deviceVendor[1024]; oclFatal(clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR, 1024, deviceVendor, nullptr));

				std::stringstream ss; ss << "device '" << deviceName << "', vendor '" << deviceVendor << "', platform '" << platformName << "'";
				deviceDesc device;
				device.platform_id = platforms[p];
				device.device_id = devices[d];
				device.name = ss.str();
				_devices.push_back(device);
			}
		}
	}

public:
	virtual ~engine()
	{
#if defined (ocl_debug)
		std::cerr << "Delete ocl engine." << std::endl;
#endif
	}

public:
	size_t getDeviceCount() const { return _devices.size(); }

public:
	void displayDevices() const
	{
		for (size_t i = 0, n = _devices.size(); i < n; ++i)
		{
			std::cout << i << " - " << _devices[i].name << "." << std::endl;
		}
	}

public:
	cl_platform_id getPlatform(const size_t d) const { return _devices[d].platform_id; }
	cl_device_id getDevice(const size_t d) const { return _devices[d].device_id; }
};

class device : oclObject
{
private:
	const engine & _engine;
	const size_t _d;
	const cl_platform_id _platform;
	const cl_device_id _device;
#if defined (ocl_profile)
	bool _selfTuning = true;
#else
	bool _selfTuning = false;
#endif
	bool _isSync = false;
	size_t _syncCount = 0;
	cl_ulong _localMemSize = 0;
	size_t _maxWorkGroupSize = 0;
	cl_ulong _timerResolution = 0;
	cl_context _context = nullptr;
	cl_command_queue _queue = nullptr;
	cl_program _program = nullptr;
	size_t _size = 0, _constant_size = 0;
	cl_mem _x = nullptr, _y = nullptr, _t = nullptr, _cr = nullptr, _u = nullptr, _tu = nullptr, _v = nullptr, _m1 = nullptr, _m2 = nullptr, _err = nullptr;
	cl_mem _r1ir1 = nullptr, _r2 = nullptr, _ir2 = nullptr, _cr1ir1 = nullptr, _cr2ir2 = nullptr, _bp = nullptr, _ibp = nullptr;
	cl_kernel _sub_ntt64 = nullptr, _ntt64 = nullptr, _intt64 = nullptr;
	cl_kernel _square32 = nullptr, _square64 = nullptr, _square128 = nullptr, _square256 = nullptr,_square512 = nullptr, _square1024 = nullptr;
	cl_kernel _square2048 = nullptr, _square4096 = nullptr;
	cl_kernel _poly2int0 = nullptr, _poly2int1 = nullptr;
	cl_kernel _reduce_upsweep64 = nullptr, _reduce_downsweep64 = nullptr;
	cl_kernel _reduce_topsweep32 = nullptr, _reduce_topsweep64 = nullptr, _reduce_topsweep128 = nullptr;
	cl_kernel _reduce_topsweep256 = nullptr, _reduce_topsweep512 = nullptr, _reduce_topsweep1024 = nullptr;
	cl_kernel _reduce_i = nullptr, _reduce_o = nullptr, _reduce_f = nullptr, _reduce_x = nullptr, _reduce_z = nullptr;
	cl_kernel _ntt4 = nullptr, _intt4 = nullptr, _mul2 = nullptr, _mul4 = nullptr;
	cl_kernel _set_positive = nullptr, _add1 = nullptr, _swap = nullptr, _copy = nullptr, _compare = nullptr;


	enum class EVendor { Unknown, NVIDIA, AMD, INTEL };

	struct profile
	{
		std::string name;
		size_t count;
		cl_ulong time;

		profile() {}
		profile(const std::string & name) : name(name), count(0), time(0.0) {}
	};
	std::map<cl_kernel, profile> _profileMap;

	// Must be identical to ocl defines
	static const size_t CHUNK64 = 16, BLK32 = 8, BLK64 = 4, BLK128 = 2, BLK256 = 1, P2I_WGS = 16, P2I_BLK = 16, RED_BLK = 4;

public:
	device(const engine & parent, const size_t d) : _engine(parent), _d(d), _platform(parent.getPlatform(d)), _device(parent.getDevice(d))
	{
#if defined (ocl_debug)
		std::cerr << "Create ocl device " << d << "." << std::endl;
#endif

		char deviceName[1024]; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_NAME, 1024, deviceName, nullptr));
		char deviceVendor[1024]; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_VENDOR, 1024, deviceVendor, nullptr));
		char deviceVersion[1024]; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_VERSION, 1024, deviceVersion, nullptr));
		char driverVersion[1024]; oclFatal(clGetDeviceInfo(_device, CL_DRIVER_VERSION, 1024, driverVersion, nullptr));

		cl_uint computeUnits; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr));
		cl_uint maxClockFrequency; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockFrequency), &maxClockFrequency, nullptr));
		cl_ulong memSize; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize), &memSize, nullptr));
		cl_ulong memCacheSize; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(memCacheSize), &memCacheSize, nullptr));
		cl_uint memCacheLineSize; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(memCacheLineSize), &memCacheLineSize, nullptr));
		oclFatal(clGetDeviceInfo(_device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(_localMemSize), &_localMemSize, nullptr));
		cl_ulong memConstSize; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(memConstSize), &memConstSize, nullptr));
		oclFatal(clGetDeviceInfo(_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(_maxWorkGroupSize), &_maxWorkGroupSize, nullptr));
		oclFatal(clGetDeviceInfo(_device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(_timerResolution), &_timerResolution, nullptr));

		std::cout << "Running on device '" << deviceName<< "', vendor '" << deviceVendor
			<< "', version '" << deviceVersion << "' and driver '" << driverVersion << "'." << std::endl;

		std::cout << computeUnits << " computeUnits @ " << maxClockFrequency << " MHz, memSize = " << (memSize >> 20) << " MB, cacheSize = "
			<< (memCacheSize >> 10) << " kB, cacheLineSize = " << memCacheLineSize << " B, localMemSize = " << (_localMemSize >> 10)
			<< " kB, constMemSize = " << (memConstSize >> 10) << " kB, maxWorkGroupSize = " << _maxWorkGroupSize << "." << std::endl << std::endl;
		
		const cl_context_properties contextProperties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)_platform, 0 };
		cl_int err_cc;
		_context = clCreateContext(contextProperties, 1, &_device, nullptr, nullptr, &err_cc);
		oclFatal(err_cc);
		cl_int err_ccq;
		_queue = clCreateCommandQueue(_context, _device, _selfTuning ? CL_QUEUE_PROFILING_ENABLE : 0, &err_ccq);
		oclFatal(err_ccq);

		if (getVendor(deviceVendor) != EVendor::NVIDIA) _isSync = true;
	}

public:
	virtual ~device()
	{
#if defined (ocl_debug)
		std::cerr << "Delete ocl device " << _d << "." << std::endl;
#endif
		oclFatal(clReleaseCommandQueue(_queue));
		oclFatal(clReleaseContext(_context));
	}

public:
	size_t getMaxWorkGroupSize() const { return _maxWorkGroupSize; }
	size_t getLocalMemSize() const { return _localMemSize; }

private:
	static EVendor getVendor(const std::string & vendorString)
	{
		std::string lVendorString; lVendorString.resize(vendorString.size());
		std::transform(vendorString.begin(), vendorString.end(), lVendorString.begin(), [](char c){ return std::tolower(c); });

		if (strstr(lVendorString.c_str(), "nvidia") != nullptr) return EVendor::NVIDIA;
		if (strstr(lVendorString.c_str(), "amd") != nullptr) return EVendor::AMD;
		if (strstr(lVendorString.c_str(), "advanced micro devices") != nullptr) return EVendor::AMD;
		if (strstr(lVendorString.c_str(), "intel") != nullptr) return EVendor::INTEL;
		// must be tested after 'Intel' because 'ati' is in 'Intel(R) Corporation' string
		if (strstr(lVendorString.c_str(), "ati") != nullptr) return EVendor::AMD;
		return EVendor::Unknown;
	}

public:
	void displayProfiles(const size_t count) const
	{
		cl_ulong ptime = 0;
		for (auto it : _profileMap) ptime += it.second.time;
		ptime /= count;

		for (auto it : _profileMap)
		{
			const profile & prof = it.second;
			if (prof.count != 0)
			{
				const size_t ncount = prof.count / count;
				const cl_ulong ntime = prof.time / count;
				std::cout << "- " << prof.name << ": " << ncount << ", " << std::setprecision(3)
					<< ntime * 100.0 / ptime << " %, " << ntime << " (" << (ntime / ncount) << ")" << std::endl;
			}
		}
	}

public:
	void loadProgram(const std::string & programSrc)
	{
#if defined (ocl_debug)
		std::cerr << "Load ocl program." << std::endl;
#endif
		const char * src[1]; src[0] = programSrc.c_str();
		cl_int err_cpws;
		_program = clCreateProgramWithSource(_context, 1, src, nullptr, &err_cpws);
		oclFatal(err_cpws);

		char pgmOptions[1024];
		strcpy(pgmOptions, "");
#if defined (ocl_debug)
		strcat(pgmOptions, " -cl-nv-verbose");
#endif
		const cl_int err = clBuildProgram(_program, 1, &_device, pgmOptions, nullptr, nullptr);

#if !defined (ocl_debug)
		if (err != CL_SUCCESS)
#endif		
		{
			size_t logSize; clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
			if (logSize > 2)
			{
				char * buildLog = new char[logSize + 1];
				clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_LOG, logSize, buildLog, nullptr);
				buildLog[logSize] = '\0';
				std::cerr << buildLog << std::endl;
#if defined (ocl_debug)
				std::ofstream fileOut("pgm.log"); 
				fileOut << buildLog << std::endl;
				fileOut.close();
#endif
				delete[] buildLog;
			}
		}

		oclFatal(err);

#if defined (ocl_debug)
		size_t binSize; clGetProgramInfo(_program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binSize, nullptr);
		char * binary = new char[binSize];
		clGetProgramInfo(_program, CL_PROGRAM_BINARIES, sizeof(char *), &binary, nullptr);
		std::ofstream fileOut("pgm.txt", std::ios::binary);
		fileOut.write(binary, binSize);
		fileOut.close();
		delete[] binary;
#endif	
	}

public:
	void clearProgram()
	{
#if defined (ocl_debug)
		std::cerr << "Clear ocl program." << std::endl;
#endif
		oclFatal(clReleaseProgram(_program));
		_program = nullptr;
	}

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
			_releaseBuffer(_x);
			_releaseBuffer(_y);
			_releaseBuffer(_t);
			_releaseBuffer(_cr);
			_releaseBuffer(_u);
			_releaseBuffer(_tu);
			_releaseBuffer(_v);
			_releaseBuffer(_m1);
			_releaseBuffer(_m2);
			_releaseBuffer(_err);

			_releaseBuffer(_r1ir1);
			_releaseBuffer(_r2);
			_releaseBuffer(_ir2);
			_releaseBuffer(_bp);
			_releaseBuffer(_ibp);
			_size = 0;
		}

		if (_constant_size != 0)
		{
			_releaseBuffer(_cr1ir1);
			_releaseBuffer(_cr2ir2);
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
		_releaseKernel(_sub_ntt64);
		_releaseKernel(_ntt64);
		_releaseKernel(_intt64);

		_releaseKernel(_square32);
		_releaseKernel(_square64);
		_releaseKernel(_square128);
		_releaseKernel(_square256);
		_releaseKernel(_square512);
		_releaseKernel(_square1024);
		_releaseKernel(_square2048);
		_releaseKernel(_square4096);

		_releaseKernel(_poly2int0);
		_releaseKernel(_poly2int1);

		_releaseKernel(_reduce_upsweep64);
		_releaseKernel(_reduce_downsweep64);

		_releaseKernel(_reduce_topsweep32);
		_releaseKernel(_reduce_topsweep64);
		_releaseKernel(_reduce_topsweep128);
		_releaseKernel(_reduce_topsweep256);
		_releaseKernel(_reduce_topsweep512);
		_releaseKernel(_reduce_topsweep1024);

		_releaseKernel(_reduce_i);
		_releaseKernel(_reduce_o);
		_releaseKernel(_reduce_f);
		_releaseKernel(_reduce_x);
		_releaseKernel(_reduce_z);

		_releaseKernel(_ntt4);
		_releaseKernel(_intt4);
		_releaseKernel(_mul2);
		_releaseKernel(_mul4);
		_releaseKernel(_set_positive);
		_releaseKernel(_add1);

		_releaseKernel(_swap);
		_releaseKernel(_copy);
		_releaseKernel(_compare);
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
	void sub_ntt64() { _executeKernel(_sub_ntt64, _size / 4, CHUNK64 * (64 / 4)); }

private:
	inline void _executeNttKernel(cl_kernel kernel, const cl_uint m, const cl_uint rindex)
	{
		_setKernelArg(kernel, 3, sizeof(cl_uint), &m);
		_setKernelArg(kernel, 4, sizeof(cl_uint), &rindex);
		_executeKernel(kernel, _size / 4, CHUNK64 * (64 / 4));
	}

public:
	void ntt64(const cl_uint m, const cl_uint rindex) { _executeNttKernel(_ntt64, m, rindex); }
	void intt64(const cl_uint m, const cl_uint rindex) { _executeNttKernel(_intt64, m, rindex); }
	void ntt4(const cl_uint m, const cl_uint rindex) { _executeNttKernel(_ntt4, m, rindex); }
	void intt4(const cl_uint m, const cl_uint rindex) { _executeNttKernel(_intt4, m, rindex); }

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
		_executeNttKernel(_ntt64, m, rindex);
		_setKernelArg(_ntt64, 0, sizeof(cl_mem), &_x);
	}

public:
	void ntt4_u(const cl_uint m, const cl_uint rindex)
	{
		_setKernelArg(_ntt4, 0, sizeof(cl_mem), &_tu);
		_executeNttKernel(_ntt4, m, rindex);
		_setKernelArg(_ntt4, 0, sizeof(cl_mem), &_x);
	}

public:
	void square32() { _executeKernel(_square32, _size / 4, BLK32 * 32 / 4); }
	void square64() { _executeKernel(_square64, _size / 4, BLK64 * 64 / 4); }
	void square128() { _executeKernel(_square128, _size / 4, BLK128 * 128 / 4); }
	void square256() { _executeKernel(_square256, _size / 4, BLK256 * 256 / 4); }
	void square512() { _executeKernel(_square512, _size / 4, 512 / 4); }
	void square1024() { _executeKernel(_square1024, _size / 4, 1024 / 4); }
	void square2048() { _executeKernel(_square2048, _size / 4, 2048 / 4); }
	void square4096() { _executeKernel(_square4096, _size / 4, 4096 / 4); }

public:
	void mul2() { _executeKernel(_mul2, _size / 4); }
	void mul4() { _executeKernel(_mul4, _size / 4); }

public:
	void poly2int0() { _executeKernel(_poly2int0, _size / P2I_BLK, P2I_WGS); }
	void poly2int1() { _executeKernel(_poly2int1, _size / P2I_BLK); }

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

private:
	void _sync()
	{
		_syncCount = 0;
		oclFatal(clFinish(_queue));
	}

private:
	cl_mem _createBuffer(const cl_mem_flags flags, const size_t size, const bool debug = false) const
	{
		cl_int err;
		cl_mem mem = clCreateBuffer(_context, flags, size, nullptr, &err);
		oclFatal(err);
		if (debug)
		{
			uint8_t * const ptr = new uint8_t[size];
			for (size_t i = 0; i < size; ++i) ptr[i] = 0xff;
			oclFatal(clEnqueueWriteBuffer(_queue, mem, CL_TRUE, 0, size, ptr, 0, nullptr, nullptr));
			delete[] ptr;
		}
		return mem;
	}

private:
	static void _releaseBuffer(cl_mem & mem)
	{
		if (mem != nullptr)
		{
			oclFatal(clReleaseMemObject(mem));
			mem = nullptr;
		}
	}

private:
	void _readBuffer(cl_mem & mem, void * const ptr, const size_t size)
	{
		_sync();
		oclFatal(clEnqueueReadBuffer(_queue, mem, CL_TRUE, 0, size, ptr, 0, nullptr, nullptr));
	}

private:
	void _writeBuffer(cl_mem & mem, const void * const ptr, const size_t size)
	{
		_sync();
		oclFatal(clEnqueueWriteBuffer(_queue, mem, CL_TRUE, 0, size, ptr, 0, nullptr, nullptr));
	}

private:
	cl_kernel _createKernel(const char * const kernelName)
	{
		cl_int err;
		cl_kernel kernel = clCreateKernel(_program, kernelName, &err);
		oclFatal(err);
		_profileMap[kernel] = profile(kernelName);
		return kernel;
	}

private:
	static void _releaseKernel(cl_kernel & kernel)
	{
		if (kernel != nullptr)
		{
			oclFatal(clReleaseKernel(kernel));
			kernel = nullptr;
		}		
	}

private:
	static void _setKernelArg(cl_kernel kernel, const cl_uint arg_index, const size_t arg_size, const void * const arg_value)
	{
#if !defined (ocl_fast_exec) || defined (ocl_debug)
		cl_int err =
#endif
		clSetKernelArg(kernel, arg_index, arg_size, arg_value);
#if !defined (ocl_fast_exec) || defined (ocl_debug)
		oclFatal(err);
#endif
	}

private:
	void _executeKernelN(cl_kernel kernel, const size_t globalWorkSize, const size_t localWorkSize = 0)
	{
#if !defined (ocl_fast_exec) || defined (ocl_debug)
		cl_int err =
#endif
		clEnqueueNDRangeKernel(_queue, kernel, 1, nullptr, &globalWorkSize, (localWorkSize == 0) ? nullptr : &localWorkSize, 0, nullptr, nullptr);
#if !defined (ocl_fast_exec) || defined (ocl_debug)
		oclFatal(err);
#endif
		if (_isSync)
		{
			++_syncCount;
			if (_syncCount == 1024) _sync();
		}
	}

private:
	void _executeKernelP(cl_kernel kernel, const size_t globalWorkSize, const size_t localWorkSize = 0)
	{
		_sync();
		cl_event evt;
		oclFatal(clEnqueueNDRangeKernel(_queue, kernel, 1, nullptr, &globalWorkSize, (localWorkSize == 0) ? nullptr : &localWorkSize, 0, nullptr, &evt));
		cl_ulong dt = 0;
		if (clWaitForEvents(1, &evt) == CL_SUCCESS)
		{
			cl_ulong start, end;
			cl_int err_s = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
			cl_int err_e = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
			if ((err_s == CL_SUCCESS) && (err_e == CL_SUCCESS)) dt = end - start;
		}
		clReleaseEvent(evt);

		profile & prof = _profileMap[kernel];
		prof.count++;
		prof.time += dt;
	}
};

}