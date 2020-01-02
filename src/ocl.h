/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <CL/cl.h>

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

namespace ocl
{

//#define ocl_debug	1

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

	virtual ~Engine()
	{
#if ocl_debug
		std::cerr << "Delete ocl engine." << std::endl;
#endif
	}

	void displayDevices() const
	{
		for (size_t i = 0, n = _devices.size(); i < n; ++i)
		{
			std::cout << i << " - " << _devices[i].name << "." << std::endl;
		}
		std::cout << std::endl;
	}

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
	bool _selfTuning = false;
	size_t _syncCount = 0;
	cl_ulong _localMemSize = 0;
	size_t _maxWorkGroupSize = 0;
	cl_ulong _timerResolution = 0;
	cl_context _context = nullptr;
	cl_command_queue _queue = nullptr;
	cl_program _program = nullptr;
	size_t _size = 0, _size_blk = 0;
	cl_mem _x = nullptr, _r1ir1 = nullptr, _r2 = nullptr, _ir2 = nullptr, _cr = nullptr, _bp = nullptr, _ibp = nullptr, _err = nullptr;
	cl_kernel _sub_ntt4 = nullptr, _ntt4 = nullptr, _intt4 = nullptr, _square2 = nullptr, _square4 = nullptr;
	cl_kernel _poly2int0 = nullptr, _poly2int1 = nullptr, _split0 = nullptr, _split4_i = nullptr, _split4_01 = nullptr, _split4_10 = nullptr;
	cl_kernel _split2 = nullptr, _split2_10 = nullptr, _split_o = nullptr, _split_o_10 = nullptr, _split_f = nullptr;

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
		oclFatal(clGetDeviceInfo(_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(_maxWorkGroupSize), &_maxWorkGroupSize, nullptr));
		oclFatal(clGetDeviceInfo(_device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(_timerResolution), &_timerResolution, nullptr));

		std::cout << "Running on device '" << deviceName<< "', vendor '" << deviceVendor
			<< "', version '" << deviceVersion << "' and driver '" << driverVersion << "'." << std::endl;

		std::cout << computeUnits << " computeUnits @ " << maxClockFrequency << " MHz, memSize = " << (memSize >> 20) << " MB, cacheSize = "
			<< (memCacheSize >> 10) << " kB, cacheLineSize = " << memCacheLineSize << " B, localMemSize = " << (_localMemSize >> 10)
			<< " kB, maxWorkGroupSize = " << _maxWorkGroupSize << "." << std::endl << std::endl;
		
		const cl_context_properties contextProperties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)_platform, 0 };
		cl_int err_cc;
		_context = clCreateContext(contextProperties, 1, &_device, nullptr, nullptr, &err_cc);
		oclFatal(err_cc);
		cl_int err_ccq;
		_queue = clCreateCommandQueue(_context, _device, _selfTuning ? CL_QUEUE_PROFILING_ENABLE : 0, &err_ccq);
		oclFatal(err_ccq);
	}

	virtual ~Device()
	{
#if ocl_debug
		std::cerr << "Delete ocl device " << _d << "." << std::endl;
#endif
		oclFatal(clReleaseCommandQueue(_queue));
		oclFatal(clReleaseContext(_context));
	}

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

	void clearProgram()
	{
#if ocl_debug
		std::cerr << "Clear ocl program." << std::endl;
#endif
		oclFatal(clReleaseProgram(_program));
		_program = nullptr;
	}

	void allocMemory(const size_t size)
	{
#if ocl_debug
		std::cerr << "Alloc gpu memory." << std::endl;
#endif
		_size = size;
		_x = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint2) * size);
		_r1ir1 = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint4) * size);
		_r2 = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint2) * size);
		_ir2 = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint2) * size);
		_cr = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_long) * size);
		_bp = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint) * size / 2);
		_ibp = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint) * size / 2);
		_err = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_int));
	}

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
			_releaseBuffer(_err);
			_size = 0;
		}
	}

	void createKernels(const cl_uint2 norm, const cl_uint blk, const cl_uint e, const cl_int s, const cl_uint d, const cl_uint d_inv, const cl_int d_shift)
	{
#if ocl_debug
		std::cerr << "Create ocl kernels." << std::endl;
#endif
		_size_blk = _size / blk;

		_sub_ntt4 = _createKernel("sub_ntt4");
		_setKernelArg(_sub_ntt4, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_sub_ntt4, 1, sizeof(cl_mem), &_r1ir1);
		_setKernelArg(_sub_ntt4, 2, sizeof(cl_mem), &_r2);

		_ntt4 = _createKernel("ntt4");
		_setKernelArg(_ntt4, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_ntt4, 1, sizeof(cl_mem), &_r1ir1);
		_setKernelArg(_ntt4, 2, sizeof(cl_mem), &_r2);

		_intt4 = _createKernel("intt4");
		_setKernelArg(_intt4, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_intt4, 1, sizeof(cl_mem), &_r1ir1);
		_setKernelArg(_intt4, 2, sizeof(cl_mem), &_ir2);

		_square2 = _createKernel("square2");
		_setKernelArg(_square2, 0, sizeof(cl_mem), &_x);

		_square4 = _createKernel("square4");
		_setKernelArg(_square4, 0, sizeof(cl_mem), &_x);

		_poly2int0 = _createKernel("poly2int0");
		_setKernelArg(_poly2int0, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_poly2int0, 1, sizeof(cl_mem), &_cr);
		_setKernelArg(_poly2int0, 2, sizeof(cl_uint2), &norm);
		_setKernelArg(_poly2int0, 3, sizeof(cl_uint), &blk);

		_poly2int1 = _createKernel("poly2int1");
		_setKernelArg(_poly2int1, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_poly2int1, 1, sizeof(cl_mem), &_cr);
		_setKernelArg(_poly2int1, 2, sizeof(cl_uint), &blk);
		_setKernelArg(_poly2int1, 3, sizeof(cl_mem), &_err);

		_split0 = _createKernel("split0");
		_setKernelArg(_split0, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_split0, 1, sizeof(cl_uint), &e);
		_setKernelArg(_split0, 2, sizeof(cl_int), &s);

		_split4_i = _createKernel("split4_i");
		_setKernelArg(_split4_i, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_split4_i, 1, sizeof(cl_mem), &_bp);
		_setKernelArg(_split4_i, 2, sizeof(cl_uint), &d);
		_setKernelArg(_split4_i, 3, sizeof(cl_uint), &d_inv);
		_setKernelArg(_split4_i, 4, sizeof(cl_int), &d_shift);

		_split4_01 = _createKernel("split4_01");
		_setKernelArg(_split4_01, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_split4_01, 1, sizeof(cl_uint), &d);

		_split4_10 = _createKernel("split4_10");
		_setKernelArg(_split4_10, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_split4_10, 1, sizeof(cl_uint), &d);

		_split2 = _createKernel("split2");
		_setKernelArg(_split2, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_split2, 1, sizeof(cl_uint), &d);

		_split2_10 = _createKernel("split2_10");
		_setKernelArg(_split2_10, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_split2_10, 1, sizeof(cl_uint), &d);

		_split_o = _createKernel("split_o");
		_setKernelArg(_split_o, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_split_o, 1, sizeof(cl_mem), &_ibp);
		_setKernelArg(_split_o, 2, sizeof(cl_uint), &e);
		_setKernelArg(_split_o, 3, sizeof(cl_int), &s);
		_setKernelArg(_split_o, 4, sizeof(cl_uint), &d);
		_setKernelArg(_split_o, 5, sizeof(cl_uint), &d_inv);
		_setKernelArg(_split_o, 6, sizeof(cl_int), &d_shift);

		_split_o_10 = _createKernel("split_o_10");
		_setKernelArg(_split_o_10, 0, sizeof(cl_mem), &_x);
		_setKernelArg(_split_o_10, 1, sizeof(cl_mem), &_ibp);
		_setKernelArg(_split_o_10, 2, sizeof(cl_uint), &e);
		_setKernelArg(_split_o_10, 3, sizeof(cl_int), &s);
		_setKernelArg(_split_o_10, 4, sizeof(cl_uint), &d);
		_setKernelArg(_split_o_10, 5, sizeof(cl_uint), &d_inv);
		_setKernelArg(_split_o_10, 6, sizeof(cl_int), &d_shift);

		_split_f = _createKernel("split_f");
		_setKernelArg(_split_f, 0, sizeof(cl_mem), &_x);
		const cl_uint n = cl_uint(_size / 2);
		_setKernelArg(_split_f, 1, sizeof(cl_uint), &n);
		_setKernelArg(_split_f, 2, sizeof(cl_uint), &e);
		_setKernelArg(_split_f, 3, sizeof(cl_int), &s);
	}

	void releaseKernels()
	{
#if ocl_debug
		std::cerr << "Release ocl kernels." << std::endl;
#endif
		_size_blk = 0;

		_releaseKernel(_sub_ntt4);
		_releaseKernel(_ntt4);
		_releaseKernel(_intt4);
		_releaseKernel(_square2);
		_releaseKernel(_square4);
		_releaseKernel(_poly2int0);
		_releaseKernel(_poly2int1);
		_releaseKernel(_split0);
		_releaseKernel(_split4_i);
		_releaseKernel(_split4_01);
		_releaseKernel(_split4_10);
		_releaseKernel(_split2);
		_releaseKernel(_split2_10);
		_releaseKernel(_split_o);
		_releaseKernel(_split_o_10);
		_releaseKernel(_split_f);

	}

	void readMemory_x(cl_uint2 * const ptr)
	{
		_sync();
		oclFatal(clEnqueueReadBuffer(_queue, _x, CL_TRUE, 0, sizeof(cl_uint2) * _size, ptr, 0, nullptr, nullptr));
	}

	void writeMemory_x(const cl_uint2 * const ptr)
	{
		_sync();
		oclFatal(clEnqueueWriteBuffer(_queue, _x, CL_TRUE, 0, sizeof(cl_uint2) * _size, ptr, 0, nullptr, nullptr));
	}

	void writeMemory_r(const cl_uint4 * const ptr_r1ir1, const cl_uint2 * const ptr_r2, const cl_uint2 * const ptr_ir2)
	{
		_sync();
		oclFatal(clEnqueueWriteBuffer(_queue, _r1ir1, CL_TRUE, 0, sizeof(cl_uint4) * _size, ptr_r1ir1, 0, nullptr, nullptr));
		oclFatal(clEnqueueWriteBuffer(_queue, _r2, CL_TRUE, 0, sizeof(cl_uint2) * _size, ptr_r2, 0, nullptr, nullptr));
		oclFatal(clEnqueueWriteBuffer(_queue, _ir2, CL_TRUE, 0, sizeof(cl_uint2) * _size, ptr_ir2, 0, nullptr, nullptr));
	}

	void writeMemory_bp(const cl_uint * const ptr_bp, const cl_uint * const ptr_ibp)
	{
		_sync();
		oclFatal(clEnqueueWriteBuffer(_queue, _bp, CL_TRUE, 0, sizeof(cl_uint) * _size / 2, ptr_bp, 0, nullptr, nullptr));
		oclFatal(clEnqueueWriteBuffer(_queue, _ibp, CL_TRUE, 0, sizeof(cl_uint) * _size / 2, ptr_ibp, 0, nullptr, nullptr));
	}

	void readMemory_err(cl_int * const ptr)
	{
		_sync();
		oclFatal(clEnqueueReadBuffer(_queue, _err, CL_TRUE, 0, sizeof(cl_int), ptr, 0, nullptr, nullptr));
	}

	void writeMemory_err(const cl_int * const ptr)
	{
		_sync();
		oclFatal(clEnqueueWriteBuffer(_queue, _err, CL_TRUE, 0, sizeof(cl_int), ptr, 0, nullptr, nullptr));
	}

	void sub_ntt4(const cl_uint rindex)
	{
		_setKernelArg(_sub_ntt4, 3, sizeof(cl_uint), &rindex);
		_executeKernel(_sub_ntt4, _size / 4);
	}

	void ntt4(const cl_uint m, const cl_uint rindex)
	{
		_setKernelArg(_ntt4, 3, sizeof(cl_uint), &m);
		_setKernelArg(_ntt4, 4, sizeof(cl_uint), &rindex);
		_executeKernel(_ntt4, _size / 4);
	}

	void intt4(const cl_uint m, const cl_uint rindex)
	{
		_setKernelArg(_intt4, 3, sizeof(cl_uint), &m);
		_setKernelArg(_intt4, 4, sizeof(cl_uint), &rindex);
		_executeKernel(_intt4, _size / 4);
	}

	void square2() { _executeKernel(_square2, _size / 4); }
	void square4() { _executeKernel(_square4, _size / 4); }

	void poly2int0() { _executeKernel(_poly2int0, _size_blk); }
	void poly2int1() { _executeKernel(_poly2int1, _size_blk); }

	void split0() { _executeKernel(_split0, _size / 2); }
	void split4_i() { _executeKernel(_split4_i, _size / 8); }

	void split4_01(const cl_uint m)
	{
		_setKernelArg(_split4_01, 2, sizeof(cl_uint), &m);
		_executeKernel(_split4_01, _size / 8);
	}

	void split4_10(const cl_uint m)
	{
		_setKernelArg(_split4_10, 2, sizeof(cl_uint), &m);
		_executeKernel(_split4_10, _size / 8);
	}

	void split2() { _executeKernel(_split2, _size / 4); }
	void split2_10() { _executeKernel(_split2_10, _size / 4); }

	void split_o() { _executeKernel(_split_o, _size / 2); }
	void split_o_10() { _executeKernel(_split_o_10, _size / 2); }

	void split_f() { _executeKernel(_split_f, 1); }

private:
	void _sync()
	{
		_syncCount = 0;
		oclFatal(clFinish(_queue));
	}

	cl_mem _createBuffer(const cl_mem_flags flags, const size_t size) const
	{
		cl_int err;
		cl_mem mem = clCreateBuffer(_context, flags, size, nullptr, &err);
		oclFatal(err);
		return mem;
	}

	static void _releaseBuffer(cl_mem & mem)
	{
		if (mem != nullptr)
		{
			oclFatal(clReleaseMemObject(mem));
			mem = nullptr;
		}
	}

	cl_kernel _createKernel(const char * const kernelName) const
	{
		cl_int err;
		cl_kernel kern = clCreateKernel(_program, kernelName, &err);
		oclFatal(err);
		return kern;
	}

	static void _releaseKernel(cl_kernel & kernel)
	{
		if (kernel != nullptr)
		{
			oclFatal(clReleaseKernel(kernel));
			kernel = nullptr;
		}		
	}

	static void _setKernelArg(cl_kernel kernel, const cl_uint arg_index, const size_t arg_size, const void * const arg_value)
	{
		oclFatal(clSetKernelArg(kernel, arg_index, arg_size, arg_value));
	}

	void _executeKernel(cl_kernel kernel, const size_t globalWorkSize, const size_t localWorkSize = 0)
	{
		//size_t localWS = std::min(globalWorkSize, size_t(16));
		//oclFatal(clEnqueueNDRangeKernel(_queue, kernel, 1, nullptr, &globalWorkSize, &localWS, 0, nullptr, nullptr));
		oclFatal(clEnqueueNDRangeKernel(_queue, kernel, 1, nullptr, &globalWorkSize, (localWorkSize == 0) ? nullptr : &localWorkSize, 0, nullptr, nullptr));
		++_syncCount;
		//if (_syncCount >= 1024) _sync();
	}

	cl_ulong _profileKernel(cl_kernel kernel, const size_t globalWorkSize, const size_t localWorkSize = 0)
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
		return dt;
	}
};

}