/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <string>
#include <sstream>
#include <iomanip>

#if defined (_WIN32) // use Performance Counter

#include <Windows.h>

class Timer
{
public:
	typedef LARGE_INTEGER Time;

	static Time currentTime()
	{
		LARGE_INTEGER time; QueryPerformanceCounter(&time);
		return time;
	}

	static double diffTime(const Time & end, const Time & start)
	{
		LARGE_INTEGER freq; QueryPerformanceFrequency(&freq);
		return double(end.QuadPart - start.QuadPart) / double(freq.QuadPart);
	}

#else // Otherwise use gettimeofday() instead

#include <sys/time.h>

class Timer
{
public:
	typedef timeval Time;

	static Time currentTime()
	{
		timeval time; gettimeofday(&time, nullptr);
		return time;
	}

	static double diffTime(const Time & end, const Time & start)
	{
		return double(end.tv_sec - start.tv_sec) + double(end.tv_usec - start.tv_usec) * 1e-6;
	}

#endif

	static std::string formatTime(const double time)
	{
		uint64_t seconds = uint64_t(time), minutes = seconds / 60, hours = minutes / 60;
		seconds -= minutes * 60; minutes -= hours * 60;

		std::stringstream ss;
		ss << std::setfill('0') << std::setw(2) <<  hours << ':' << std::setw(2) << minutes << ':' << std::setw(2) << seconds;
		return ss.str();
	}
};