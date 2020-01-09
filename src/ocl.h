/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <CL/cl.h>

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

namespace ocl
{

//#define ocl_debug	1
//#define ocl_profile	1

#ifdef ocl_profile
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
	static bool oclError(const cl_int res)
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

class Engine : oclObject
{
private:
	struct SDevice
	{
		cl_platform_id platform_id;
		cl_device_id device_id;
		std::string name;
	};
	std::vector<SDevice> _devices;

public:
	Engine()
	{
#if ocl_debug
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
				SDevice device;
				device.platform_id = platforms[p];
				device.device_id = devices[d];
				device.name = ss.str();
				_devices.push_back(device);
			}
		}
	}

public:
	virtual ~Engine()
	{
#if ocl_debug
		std::cerr << "Delete ocl engine." << std::endl;
#endif
	}

public:
	void displayDevices() const
	{
		for (size_t i = 0, n = _devices.size(); i < n; ++i)
		{
			std::cout << i << " - " << _devices[i].name << "." << std::endl;
		}
		std::cout << std::endl;
	}

public:
	cl_platform_id getPlatform(const size_t d) const { return _devices[d].platform_id; }
	cl_device_id getDevice(const size_t d) const { return _devices[d].device_id; }
};

class Device : oclObject
{
private:
	const Engine & _engine;
	const size_t _d;
	const cl_platform_id _platform;
	const cl_device_id _device;
#ifdef ocl_profile
	bool _selfTuning = true;
#else
	bool _selfTuning = false;
#endif
	size_t _syncCount = 0;
	cl_ulong _localMemSize = 0;
	size_t _maxWorkGroupSize = 0;
	cl_ulong _timerResolution = 0;
	cl_context _context = nullptr;
	cl_command_queue _queue = nullptr;
	cl_program _program = nullptr;
	size_t _size = 0, _constant_size = 0;
	cl_mem _x = nullptr, _r1ir1 = nullptr, _r2 = nullptr, _ir2 = nullptr, _cr = nullptr, _bp = nullptr, _ibp = nullptr, _t = nullptr, _err = nullptr;
	cl_mem _cr1ir1 = nullptr, _cr2ir2 = nullptr;
	cl_kernel _sub_ntt64 = nullptr, _ntt64 = nullptr, _intt64 = nullptr;
	cl_kernel _square32 = nullptr, _square64 = nullptr, _square128 = nullptr, _square256 = nullptr,_square512 = nullptr, _square1024 = nullptr;
	cl_kernel _poly2int0 = nullptr, _poly2int1 = nullptr;
	cl_kernel _reduce_i = nullptr, _reduce_o = nullptr, _reduce_f = nullptr;
	cl_kernel _reduce_upsweep4 = nullptr, _reduce_downsweep4 = nullptr, _reduce_topsweep2 = nullptr, _reduce_topsweep4 = nullptr;
	;
	struct Profile
	{
		std::string name;
		size_t count;
		cl_ulong time;

		Profile() {}
		Profile(const std::string & name) : name(name), count(0), time(0.0) {}
	};
	std::map<cl_kernel, Profile> _profileMap;

	// Must be identical to ocl defines
	static const size_t CHUNK64 = 16, BLK32 = 8, BLK64 = 4, BLK128 = 2, BLK256 = 1, P2I_WGS = 16, P2I_BLK = 16;

public:
	Device(const Engine & engine, const size_t d) : _engine(engine), _d(d), _platform(engine.getPlatform(d)), _device(engine.getDevice(d))
	{
#if ocl_debug
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
	}

public:
	virtual ~Device()
	{
#if ocl_debug
		std::cerr << "Delete ocl device " << _d << "." << std::endl;
#endif
		oclFatal(clReleaseCommandQueue(_queue));
		oclFatal(clReleaseContext(_context));
	}

public:
	void displayProfiles(const size_t count) const
	{
		cl_ulong ptime = 0;
		for (auto it : _profileMap) ptime += it.second.time;
		ptime /= count;

		for (auto it : _profileMap)
		{
			const Profile & profile = it.second;
			if (profile.count != 0)
			{
				const size_t ncount = profile.count / count;
				const cl_ulong ntime = profile.time / count;
				std::cout << "- " << profile.name << ": " << ncount << ", " << std::setprecision(3)
					<< ntime * 100.0 / ptime << " %, " << ntime << " (" << (ntime / ncount) << ")" << std::endl;
			}
		}
	}

public:
	void loadProgram(const std::string & programSrc)
	{
#if ocl_debug
		std::cerr << "Load ocl program." << std::endl;
#endif
		const char * src[1]; src[0] = programSrc.c_str();
		cl_int err_cpws;
		_program = clCreateProgramWithSource(_context, 1, src, nullptr, &err_cpws);
		oclFatal(err_cpws);

		char pgmOptions[1024];
		strcpy(pgmOptions, "");
#if ocl_debug
		strcat(pgmOptions, " -cl-nv-verbose");
#endif
		const cl_int err = clBuildProgram(_program, 1, &_device, pgmOptions, nullptr, nullptr);

#if !ocl_debug
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
#if ocl_debug
				std::ofstream fileOut("pgm.log"); 
				fileOut << buildLog << std::endl;
				fileOut.close();
#endif
				delete[] buildLog;
			}
		}

		oclFatal(err);

#if ocl_debug
		size_t binSize; clGetProgramInfo(_program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binSize, nullptr);
		char * binary = new char[binSize];
		clGetProgramInfo(_program, CL_PROGRAM_BINARIES, sizeof(char *), &binary, nullptr);
		std::ofstream fileOut("pgm.txt", std::ios_base::binary);
		fileOut.write(binary, binSize);
		fileOut.close();
		delete[] binary;
#endif	
	}

public:
	void clearProgram()
	{
#if ocl_debug
		std::cerr << "Clear ocl program." << std::endl;
#endif
		oclFatal(clReleaseProgram(_program));
		_program = nullptr;
	}

public:
	void allocMemory(const size_t size, const size_t constant_size)
	{
#if ocl_debug
		std::cerr << "Alloc gpu memory." << std::endl;
#endif
		_size = size;
		_x = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint2) * size);
		_r1ir1 = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint4) * size);
		_r2 = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint2) * size);
		_ir2 = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint2) * size);
		_cr = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_long) * size / P2I_BLK);
		_bp = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint4) * size / 2 / 4);
		_ibp = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint) * size / 2);
		_t = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint) * (size / 2 + 1));
		_err = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_int));

		_constant_size = constant_size;
		_cr1ir1 = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint4) * constant_size);
		_cr2ir2 = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint4) * constant_size);
	}

public:
	void releaseMemory()
	{
#if ocl_debug
		std::cerr << "Free gpu memory." << std::endl;
#endif
		if (_size != 0)
		{
			_releaseBuffer(_x);
			_releaseBuffer(_r1ir1);
			_releaseBuffer(_r2);
			_releaseBuffer(_ir2);
			_releaseBuffer(_cr);
			_releaseBuffer(_bp);
			_releaseBuffer(_ibp);
			_releaseBuffer(_t);
			_releaseBuffer(_err);
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
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_t);
		_setKernelArg(kernel, 2, sizeof(cl_mem), forward ? &_bp : &_ibp);
		_setKernelArg(kernel, 3, sizeof(cl_uint), &e);
		_setKernelArg(kernel, 4, sizeof(cl_int), &s);
		_setKernelArg(kernel, 5, sizeof(cl_uint), &d);
		_setKernelArg(kernel, 6, sizeof(cl_uint), &d_inv);
		_setKernelArg(kernel, 7, sizeof(cl_int), &d_shift);
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
	void createKernels(const cl_uint2 norm, const cl_uint e, const cl_int s, const cl_uint d, const cl_uint d_inv, const cl_int d_shift)
	{
#if ocl_debug
		std::cerr << "Create ocl kernels." << std::endl;
#endif
		_sub_ntt64 = _createNttKernel("sub_ntt64", true);
		_ntt64 = _createNttKernel("ntt64", true);
		_intt64 = _createNttKernel("intt64", false);

		_square32 = _createSquareKernel("square32");
		_square64 = _createSquareKernel("square64");
		_square128 = _createSquareKernel("square128");
		_square256 = _createSquareKernel("square256");
		_square512 = _createSquareKernel("square512");
		_square1024 = _createSquareKernel("square1024");

		_poly2int0 = _createKernel("poly2int0");
		_setKernelArg(_poly2int0, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_poly2int0, 1, sizeof(cl_mem), &_cr);
		_setKernelArg(_poly2int0, 2, sizeof(cl_uint2), &norm);

		_poly2int1 = _createKernel("poly2int1");
		_setKernelArg(_poly2int1, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_poly2int1, 1, sizeof(cl_mem), &_cr);
		_setKernelArg(_poly2int1, 2, sizeof(cl_mem), &_err);

		_reduce_i = _createReduceKernel("reduce_i", true, e, s, d, d_inv, d_shift);
		_reduce_o = _createReduceKernel("reduce_o", false, e, s, d, d_inv, d_shift);

		_reduce_f = _createKernel("reduce_f");
		_setKernelArg(_reduce_f, 0, sizeof(cl_mem), &_x);
		const cl_uint n = cl_uint(_size / 2);
		_setKernelArg(_reduce_f, 1, sizeof(cl_uint), &n);
		_setKernelArg(_reduce_f, 2, sizeof(cl_uint), &e);
		_setKernelArg(_reduce_f, 3, sizeof(cl_int), &s);

		_reduce_upsweep4 = _createSweepKernel("reduce_upsweep4", d);
		_reduce_downsweep4 = _createSweepKernel("reduce_downsweep4", d);

		_reduce_topsweep2 = _createSweepKernel("reduce_topsweep2", d);
		const cl_uint n_2 = cl_uint(_size / 4);
		_setKernelArg(_reduce_topsweep2, 2, sizeof(cl_uint), &n_2);

		_reduce_topsweep4 = _createSweepKernel("reduce_topsweep4", d);
		const cl_uint n_4 = cl_uint(_size / 8);
		_setKernelArg(_reduce_topsweep4, 2, sizeof(cl_uint), &n_4);
	}

public:
	void releaseKernels()
	{
#if ocl_debug
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

		_releaseKernel(_poly2int0);
		_releaseKernel(_poly2int1);

		_releaseKernel(_reduce_i);
		_releaseKernel(_reduce_o);
		_releaseKernel(_reduce_f);
		_releaseKernel(_reduce_upsweep4);
		_releaseKernel(_reduce_downsweep4);
		_releaseKernel(_reduce_topsweep2);
		_releaseKernel(_reduce_topsweep4);
	}

public:
	void readMemory_x(cl_uint2 * const ptr)
	{
		_sync();
		oclFatal(clEnqueueReadBuffer(_queue, _x, CL_TRUE, 0, sizeof(cl_uint2) * _size, ptr, 0, nullptr, nullptr));
	}

public:
	void writeMemory_x(const cl_uint2 * const ptr)
	{
		_sync();
		oclFatal(clEnqueueWriteBuffer(_queue, _x, CL_TRUE, 0, sizeof(cl_uint2) * _size, ptr, 0, nullptr, nullptr));
	}

public:
	void writeMemory_r(const cl_uint4 * const ptr_r1ir1, const cl_uint2 * const ptr_r2, const cl_uint2 * const ptr_ir2)
	{
		_sync();
		oclFatal(clEnqueueWriteBuffer(_queue, _r1ir1, CL_TRUE, 0, sizeof(cl_uint4) * _size, ptr_r1ir1, 0, nullptr, nullptr));
		oclFatal(clEnqueueWriteBuffer(_queue, _r2, CL_TRUE, 0, sizeof(cl_uint2) * _size, ptr_r2, 0, nullptr, nullptr));
		oclFatal(clEnqueueWriteBuffer(_queue, _ir2, CL_TRUE, 0, sizeof(cl_uint2) * _size, ptr_ir2, 0, nullptr, nullptr));
	}

public:
	void writeMemory_cr(const cl_uint4 * const ptr_cr1ir1, const cl_uint4 * const ptr_cr2ir2)
	{
		_sync();
		oclFatal(clEnqueueWriteBuffer(_queue, _cr1ir1, CL_TRUE, 0, sizeof(cl_uint4) * _constant_size, ptr_cr1ir1, 0, nullptr, nullptr));
		oclFatal(clEnqueueWriteBuffer(_queue, _cr2ir2, CL_TRUE, 0, sizeof(cl_uint4) * _constant_size, ptr_cr2ir2, 0, nullptr, nullptr));
	}

public:
	void writeMemory_bp(const cl_uint4 * const ptr_bp, const cl_uint * const ptr_ibp)
	{
		_sync();
		oclFatal(clEnqueueWriteBuffer(_queue, _bp, CL_TRUE, 0, sizeof(cl_uint4) * _size / 2 / 4, ptr_bp, 0, nullptr, nullptr));
		oclFatal(clEnqueueWriteBuffer(_queue, _ibp, CL_TRUE, 0, sizeof(cl_uint) * _size / 2, ptr_ibp, 0, nullptr, nullptr));
	}

public:
	void readMemory_err(cl_int * const ptr)
	{
		_sync();
		oclFatal(clEnqueueReadBuffer(_queue, _err, CL_TRUE, 0, sizeof(cl_int), ptr, 0, nullptr, nullptr));
	}

public:
	void writeMemory_err(const cl_int * const ptr)
	{
		_sync();
		oclFatal(clEnqueueWriteBuffer(_queue, _err, CL_TRUE, 0, sizeof(cl_int), ptr, 0, nullptr, nullptr));
	}

public:
	void sub_ntt64() { _executeKernel(_sub_ntt64, _size / 4, CHUNK64 * (64 / 4)); }

public:
	void ntt64(const cl_uint m, const cl_uint rindex)
	{
		const cl_uint m_16 = (m / 16);
		_setKernelArg(_ntt64, 3, sizeof(cl_uint), &m_16);
		_setKernelArg(_ntt64, 4, sizeof(cl_uint), &rindex);
		_executeKernel(_ntt64, _size / 4, CHUNK64 * (64 / 4));
	}

public:
	void intt64(const cl_uint m, const cl_uint rindex)
	{
		const cl_uint m_16 = (m / 16);
		_setKernelArg(_intt64, 3, sizeof(cl_uint), &m_16);
		_setKernelArg(_intt64, 4, sizeof(cl_uint), &rindex);
		_executeKernel(_intt64, _size / 4, CHUNK64 * (64 / 4));
	}

public:
	void square32() { _executeKernel(_square32, _size / 4, BLK32 * 32 / 4); }
	void square64() { _executeKernel(_square64, _size / 4, BLK64 * 64 / 4); }
	void square128() { _executeKernel(_square128, _size / 4, BLK128 * 128 / 4); }
	void square256() { _executeKernel(_square256, _size / 4, BLK256 * 256 / 4); }
	void square512() { _executeKernel(_square512, _size / 4, 512 / 4); }
	void square1024() { _executeKernel(_square1024, _size / 4, 1024 / 4); }

public:
	void poly2int0() { _executeKernel(_poly2int0, _size / P2I_BLK, P2I_WGS); }
	void poly2int1() { _executeKernel(_poly2int1, _size / P2I_BLK); }

public:
	void reduce_i() { _executeKernel(_reduce_i, _size / 8); }

public:
	void reduce_upsweep4(const cl_uint m, const cl_uint s)
	{
		_setKernelArg(_reduce_upsweep4, 2, sizeof(cl_uint), &m);
		_executeKernel(_reduce_upsweep4, s);
	}

public:
	void reduce_downsweep4(const cl_uint m, const cl_uint s)
	{
		_setKernelArg(_reduce_downsweep4, 2, sizeof(cl_uint), &m);
		_executeKernel(_reduce_downsweep4, s);
	}

public:
	void reduce_topsweep2() { _executeKernel(_reduce_topsweep2, 1); }
	void reduce_topsweep4() { _executeKernel(_reduce_topsweep4, 1); }

public:
	void reduce_o() { _executeKernel(_reduce_o, _size / 2); }
	void reduce_f() { _executeKernel(_reduce_f, 1); }

private:
	void _sync()
	{
		_syncCount = 0;
		oclFatal(clFinish(_queue));
	}

private:
	cl_mem _createBuffer(const cl_mem_flags flags, const size_t size) const
	{
		cl_int err;
		cl_mem mem = clCreateBuffer(_context, flags, size, nullptr, &err);
		oclFatal(err);
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
	cl_kernel _createKernel(const char * const kernelName)
	{
		cl_int err;
		cl_kernel kernel = clCreateKernel(_program, kernelName, &err);
		oclFatal(err);
		_profileMap[kernel] = Profile(kernelName);
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
		oclFatal(clSetKernelArg(kernel, arg_index, arg_size, arg_value));
	}

private:
	void _executeKernelN(cl_kernel kernel, const size_t globalWorkSize, const size_t localWorkSize = 0)
	{
		oclFatal(clEnqueueNDRangeKernel(_queue, kernel, 1, nullptr, &globalWorkSize, (localWorkSize == 0) ? nullptr : &localWorkSize, 0, nullptr, nullptr));
		++_syncCount;
		//if (_syncCount >= 1024) _sync();
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

		Profile & profile = _profileMap[kernel];
		profile.count++;
		profile.time += dt;
	}
};

}