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
		void split(const uint32_t m, const size_t i, solution & sol)
		{
			static const size_t CHUNK64 = 16, CHUNK256 = 8, CHUNK1024 = 4;	// TODO

			if (_b1024 && (m >= 1024 / 4 * CHUNK1024))
			{
				sol.push_back(slice(1024, CHUNK1024));
				split(m / 1024, i + 1, sol);
				sol.pop_back();
			}
			if (_b512 && (m >= 256 / 4 * CHUNK256))
			{
				sol.push_back(slice(256, CHUNK256));
				split(m / 256, i + 1, sol);
				sol.pop_back();
			}
			if (m >= 64 / 4 * CHUNK64)
			{
				sol.push_back(slice(64, CHUNK64));
				split(m / 64, i + 1, sol);
				sol.pop_back();
			}

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
			for (const slice & s : _squareSet.at(i)) { ss << s.m << " "; m /= s.m; }
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

			if (sol[0].m == 1024)
			{
				f[n] = func(&engine::sub_ntt1024);
				rindex += (256 + 64 + 16 + 4 + 1) * (m / 256);
				m /= 1024;
			}
			else if (sol[0].m == 256)
			{
				f[n] = func(&engine::sub_ntt256);
				rindex += (64 + 16 + 4 + 1) * (m / 64);
				m /= 256;
			}
			else /*if (sol[0].m == 64)*/
			{
				f[n] = func(&engine::sub_ntt64);
				rindex += (16 + 4 + 1) * (m / 16);
				m /= 64;
			}
			++n;

			for (size_t i = 1; i < sol.size(); ++i)
			{
				if (sol[i].m == 1024)
				{
					f[n] = func(&engine::ntt1024, m / 256, rindex);
					rindex += (256 + 64 + 16 + 4 + 1) * (m / 256);
					m /= 1024;
				} 
				else if (sol[i].m == 256)
				{
					f[n] = func(&engine::ntt256, m / 64, rindex);
					rindex += (64 + 16 + 4 + 1) * (m / 64);
					m /= 256;
				}
				else /*if (sol[i].m == 64)*/
				{
					f[n] = func(&engine::ntt64, m / 16, rindex);
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

				if (sol[ri].m == 1024)
				{
					m *= 1024;
					rindex -= (256 + 64 + 16 + 4 + 1) * (m / 256);
					f[n] = func(&engine::intt1024, m / 256, rindex);
				} 
				else if (sol[ri].m == 256)
				{
					m *= 256;
					rindex -= (64 + 16 + 4 + 1) * (m / 64);
					f[n] = func(&engine::intt256, m / 64, rindex);
				}
				else /*if (sol[ri].m == 64)*/
				{
					m *= 64;
					rindex -= (16 + 4 + 1) * (m / 16);
					f[n] = func(&engine::intt64, m / 16, rindex);
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

public:
	plan() {}
	virtual ~plan() {}

public:
	void init(const size_t size, const bool b512, const bool b1024) { _squareSplitter.init(size / 4, b512, b1024); }
	size_t getSquareSeqCount() const { return _squareSplitter.getSquareSize(); }
	void setSquareSeq(const size_t size, const size_t i) { _squareSeq.init(size, _squareSplitter.getSquareSeq(i)); }
	std::string getSquareSeqString(const size_t size, const size_t i) const { return _squareSplitter.getString(size, i); }
	void execSquareSeq(engine & engine) { _squareSeq.exec(engine); }
};
