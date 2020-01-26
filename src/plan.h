/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "engine.h"

class plan
{
private:
	struct slice
	{
		uint32_t m;
		uint32_t chunk;

		slice(const uint32_t m, const uint32_t chunk) : m(m), chunk(chunk) {}
	};
	typedef std::vector<slice> solution;

private:
	class squareSplitter
	{
	private:
		bool _b512 = false, _b1024 = false;
		std::vector<solution> _squareSet;

	private:
		void check(const uint32_t m, const uint32_t ms, const uint32_t chunk, const size_t i, solution & sol)
		{
			if (m >= ms / 4 * chunk)
			{
				sol.push_back(slice(ms, chunk));
				split(m / ms, i + 1, sol);
				sol.pop_back();
			}
		}

	private:
		void split(const uint32_t m, const size_t i, solution & sol)
		{
			if (_b1024) check(m, 1024, 4, i, sol);
			if (_b512) check(m, 1024, 2, i, sol);
			check(m, 1024, 1, i, sol);

			if (_b1024) check(m, 256, 16, i, sol);
			if (_b512) check(m, 256, 8, i, sol);
			check(m, 256, 4, i, sol);

			check(m, 64, 16, i, sol);

			if ((i != 0) && (m >= 2) && ((m <= 256) || (_b512 && (m <= 512)) || (_b1024 && (m <= 1024))))
			{
				_squareSet.push_back(sol);
			}
		}

	public:
		squareSplitter() {}
		virtual ~squareSplitter() {}

	public:
		void init(const uint32_t n, const bool b512, const bool b1024)
		{
			_b512 = b512; _b1024 = b1024;
			_squareSet.clear();
			solution sol; split(n, 0, sol);
		}

		size_t getSquareSize() const { return _squareSet.size(); }
		const solution & getSquareSeq(const size_t i) const { return _squareSet.at(i); }
		std::string getString(const size_t size, const size_t i) const
		{
			std::ostringstream ss;
			size_t m = size;
			for (const slice & s : _squareSet.at(i)) { ss << s.m << "_" << s.chunk << " "; m /= s.m; }
			ss << "sq_" << m;
 			return ss.str();
		}
	};

private:
	class squareSeq
	{
	private:
		struct func
		{
			void(engine::*_fn)(cl_uint, cl_uint);
			cl_uint _m;
			cl_uint _rindex;

			func() : _fn(nullptr), _m(0), _rindex(0) {}
			func(void(engine::*fn)(cl_uint, cl_uint), const cl_uint m = 0, const cl_uint rindex = 0) : _fn(fn), _m(m), _rindex(rindex) {}
		};
		size_t _n;
		func f[16];

	public:
		squareSeq() : _n(0) {}
		virtual ~squareSeq() {}

	public:
		void init(const size_t size, const solution & sol)
		{
			size_t n = 0;

			cl_uint m = cl_uint(size / 4);
			cl_uint rindex = 0;

			const slice & s = sol[0];
			if (s.m == 1024)
			{
				if (s.chunk == 1) f[n] = func(&engine::sub_ntt1024_1);
				if (s.chunk == 2) f[n] = func(&engine::sub_ntt1024_2);
				if (s.chunk == 4) f[n] = func(&engine::sub_ntt1024_4);
				rindex += (256 + 64 + 16 + 4 + 1) * (m / 256);
				m /= 1024;
			}
			else if (s.m == 256)
			{
				if (s.chunk == 4) f[n] = func(&engine::sub_ntt256_4);
				if (s.chunk == 8) f[n] = func(&engine::sub_ntt256_8);
				if (s.chunk == 16) f[n] = func(&engine::sub_ntt256_16);
				rindex += (64 + 16 + 4 + 1) * (m / 64);
				m /= 256;
			}
			else if (s.m == 64)
			{
				if (s.chunk == 16) f[n] = func(&engine::sub_ntt64_16);
				rindex += (16 + 4 + 1) * (m / 16);
				m /= 64;
			}
			++n;

			for (size_t i = 1; i < sol.size(); ++i)
			{
				const slice & s = sol[i];
				if (s.m == 1024)
				{
					if (s.chunk == 1) f[n] = func(&engine::ntt1024_1, m / 256, rindex);
					if (s.chunk == 2) f[n] = func(&engine::ntt1024_2, m / 256, rindex);
					if (s.chunk == 4) f[n] = func(&engine::ntt1024_4, m / 256, rindex);
					rindex += (256 + 64 + 16 + 4 + 1) * (m / 256);
					m /= 1024;
				} 
				else if (s.m == 256)
				{
					if (s.chunk == 4) f[n] = func(&engine::ntt256_4, m / 64, rindex);
					if (s.chunk == 8) f[n] = func(&engine::ntt256_8, m / 64, rindex);
					if (s.chunk == 16) f[n] = func(&engine::ntt256_16, m / 64, rindex);
					rindex += (64 + 16 + 4 + 1) * (m / 64);
					m /= 256;
				}
				else if (s.m == 64)
				{
					if (s.chunk == 16) f[n] = func(&engine::ntt64_16, m / 16, rindex);
					rindex += (16 + 4 + 1) * (m / 16);
					m /= 64;
				}
				++n;
			}

			if (m == 1024)       f[n] = func(&engine::square4096);
			else if (m == 512)   f[n] = func(&engine::square2048);
			else if (m == 256)   f[n] = func(&engine::square1024);
			else if (m == 128)   f[n] = func(&engine::square512);
			else if (m == 64)    f[n] = func(&engine::square256);
			else if (m == 32)    f[n] = func(&engine::square128);
			else if (m == 16)    f[n] = func(&engine::square64);
			else if (m == 8)     f[n] = func(&engine::square32);
			else if (m == 4)     f[n] = func(&engine::square16);
			else /*if (m == 2)*/ f[n] = func(&engine::square8);
			++n;

			for (size_t i = 0; i < sol.size(); ++i)
			{
				const size_t ri = sol.size() - 1 - i;

				const slice & s = sol[ri];
				if (s.m == 1024)
				{
					m *= 1024;
					rindex -= (256 + 64 + 16 + 4 + 1) * (m / 256);
					if (s.chunk == 1) f[n] = func(&engine::intt1024_1, m / 256, rindex);
					if (s.chunk == 2) f[n] = func(&engine::intt1024_2, m / 256, rindex);
					if (s.chunk == 4) f[n] = func(&engine::intt1024_4, m / 256, rindex);
				} 
				else if (s.m == 256)
				{
					m *= 256;
					rindex -= (64 + 16 + 4 + 1) * (m / 64);
					if (s.chunk == 4) f[n] = func(&engine::intt256_4, m / 64, rindex);
					if (s.chunk == 8) f[n] = func(&engine::intt256_8, m / 64, rindex);
					if (s.chunk == 16) f[n] = func(&engine::intt256_16, m / 64, rindex);
				}
				else if (s.m == 64)
				{
					m *= 64;
					rindex -= (16 + 4 + 1) * (m / 16);
					if (s.chunk == 16) f[n] = func(&engine::intt64_16, m / 16, rindex);
				}
				++n;
			}

			_n = n;
		}

	public:
		void exec(engine & engine) const
		{
			for (size_t i = 0, n = _n; i < n; ++i)
			{
				const func & fi = f[i];
				(engine.*fi._fn)(fi._m, fi._rindex);
			}
		}
	};

private:
	squareSplitter _squareSplitter;
	squareSeq _squareSeq;
	size_t _seq_i = 0;

public:
	plan() {}
	virtual ~plan() {}

public:
	void init(const size_t size, const bool b512, const bool b1024) { _squareSplitter.init(size / 4, b512, b1024); }
	size_t getSquareSeqCount() const { return _squareSplitter.getSquareSize(); }
	void setSquareSeq(const size_t size, const size_t i) { _seq_i = i; _squareSeq.init(size, _squareSplitter.getSquareSeq(i)); }
	std::string getSquareSeqString(const size_t size) const { return _squareSplitter.getString(size, _seq_i); }
	void execSquareSeq(engine & engine) { _squareSeq.exec(engine); }
};
