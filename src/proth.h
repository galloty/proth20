/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "ocl.h"
#include "arith.h"
#include "gpmp.h"
#include "pio.h"
#include "timer.h"

#include <thread>
#include <chrono>

class proth
{
private:
	struct deleter { void operator()(const proth * const p) { delete p; } };

public:
	proth() {}
	virtual ~proth() {}

	static proth & getInstance()
	{
		static std::unique_ptr<proth, deleter> pInstance(new proth());
		return *pInstance;
	}

public:
	void quit() { _quit = true; }
	void setBoinc(const bool isBoinc) { _isBoinc = isBoinc; }

protected:
	volatile bool _quit = false;
private:
	bool _isBoinc = false;

private:
	static constexpr uint32_t benchCount(const uint32_t n) { return (n < 100000) ? 50000 : 50000000 / (n / 1000); }

private:
	static std::string res64String(const uint64_t res64)
	{
		std::stringstream ss; ss << std::uppercase << std::hex << std::setfill('0') << std::setw(16) << res64;
		return ss.str();
	}

private:
	static bool boincQuitRequest(const BOINC_STATUS & status)
	{
		if ((status.quit_request | status.abort_request | status.no_heartbeat) == 0) return false;

		std::ostringstream ss; ss << std::endl << "Terminating because BOINC ";
		if (status.quit_request != 0) ss << "requested that we should quit.";
		else if (status.abort_request != 0) ss << "requested that we should abort.";
		else if (status.no_heartbeat != 0) ss << "heartbeat was lost.";
		ss << std::endl;
		pio::print(ss.str());
		return true;
	}

protected:
	static void checkError(gpmp & X)
	{
		int err = X.getError();
		if (err != 0)
		{
			throw std::runtime_error("GPU error detected!");
		}
	}

public:
	bool apowk(gpmp & X, const uint32_t a, const uint32_t k) const
	{
		// X = a^k, left-to-right algorithm
		bool s = false;
		X.init(1, a);					// x = 1, u = a
		X.setMultiplicand();			// x = 1, tu = NTT(u)
		for (int b = 0; b < 32; ++b)
		{
			if (s) X.square();			// x = iNTT(NTT(x)^2)

			if ((k & (uint32_t(1) << (31 - b))) != 0)
			{
				s = true;
				X.mul();				// x = iNTT(NTT(x).tu)
			}

			if (_quit) return false;
		}
		X.norm();
		X.copy_x_v();

		// X = a^k, right-to-left algorithm
		X.init(1, a);					// x = 1, u = a
		for (uint32_t b = 1; b <= uint32_t(k); b *= 2)
		{
			if ((k & b) != 0)
			{
				X.setMultiplicand();	// tu = NTT(u)
				X.mul();				// x = iNTT(NTT(x).tu)
			}

			X.swap_x_u();				// x <-> u
			X.square();
			X.swap_x_u();				// u = u^2

			if (_quit) return false;
		}
		X.norm();
		X.copy_x_u();
		X.compare_u_v();

		return true;
	}

public:
	bool check(const uint32_t k, const uint32_t n, engine & engine, const bool bench = false, const bool checkRes = false, const uint64_t r64 = 0)
	{
		uint32_t a = 0;
		if (!arith::proth_prime_quad_nonres(k, n, 3, a))
		{
			std::ostringstream ssr; ssr << k << " * 2^" << n << " + 1 is divisible by " << a << std::endl;
			pio::display(ssr.str());
			pio::result(ssr.str());
			if (_isBoinc)
			{
				std::ostringstream sso; sso << k << " * 2^" << n << " + 1 is complete, a = " << a << ", time = " << timer::formatTime(0.0) << std::endl;
				pio::print(sso.str());
			}
			return true;
		}

		gpmp X(k, n, engine, _isBoinc);

		chronometer chrono;
		uint32_t i0;
		const bool found = X.restoreContext(i0, chrono.previousTime);

		std::ostringstream sst; sst << (found ? "Resuming from a checkpoint " : "Testing ");
		sst << k << " * 2^" << n << " + 1, " << X.getDigits() << " digits, size = 2^" << arith::log2(X.getSize())
			<< " x " << X.getDigitBit() << " bits, plan: " << X.getPlanString() << std::endl;
		pio::print(sst.str());

		chrono.resetTime();

		if (!found)
		{
			i0 = 0;
			chrono.previousTime = 0;
			if (!apowk(X, a, k)) return false;
			checkError(X);	// Sync GPU before benchmark
		}

		const uint32_t L = 1 << (arith::log2(n) / 2);

		const uint32_t benchCnt = benchCount(n);
		uint32_t benchIter = benchCnt;
		chrono.resetBenchTime();
		chrono.resetRecordTime();

		if (_isBoinc) boinc_fraction_done(double(i0) / double(n));

		// X = X^(2^(n - 1))
		for (uint32_t i = i0 + 1; i < n; ++i)
		{
			X.square();

			if (--benchIter == 0)
			{
				if (_isBoinc) boinc_fraction_done(double(i) / double(n));
				else
				{
					if (bench) checkError(X);	// Sync GPU
					const double elapsedTime = chrono.getBenchTime();
					const double mulTime = elapsedTime / benchCnt, estimatedTime = mulTime * (n - i);
					std::ostringstream ssb; ssb << std::setprecision(3) << " " << i * 100.0 / n << "% done, "
						<< timer::formatTime(estimatedTime) << " remaining, " <<  mulTime * 1e3 << " ms/mul.";
					if (bench)
					{
						ssb << std::endl;
						pio::display(ssb.str());
						return true;
					}
					ssb << "        \r";
					pio::display(ssb.str());
					chrono.resetBenchTime();
				}
				benchIter = benchCnt;
			}

			// Robert Gerbicz error checking algorithm
			// u is d(t) and v is u(0). They must be set before the loop
			// if (i == n - 1) X.set_bug();	// test
			if ((i & (L - 1)) == 0)
			{
				X.Gerbicz_step();
			}

			if (i % 1024 == 0)
			{
				if (_isBoinc)
				{
					BOINC_STATUS status;
					boinc_get_status(&status);
					bool quit = boincQuitRequest(status);
					if (quit || (status.suspended != 0))
					{
						checkError(X);
						X.saveContext(i, chrono.getElapsedTime());
					}
					if (quit) return false;
						
					if (status.suspended != 0)
					{
						std::ostringstream ss_s; ss_s << std::endl << "BOINC client is suspended." << std::endl;
						pio::print(ss_s.str());

						while (status.suspended != 0)
						{
							std::this_thread::sleep_for(std::chrono::seconds(1));
							boinc_get_status(&status);
							if (boincQuitRequest(status)) return false;
						}

						std::ostringstream ss_r; ss_r << "BOINC client is resumed." << std::endl;
						pio::print(ss_r.str());
					}

					if (boinc_time_to_checkpoint() != 0)
					{
						checkError(X);
						X.saveContext(i, chrono.getElapsedTime());
						boinc_checkpoint_completed();
					}
				}
				else
				{
					const double elapsedTime = chrono.getRecordTime();
					if (elapsedTime > 600)
					{
						checkError(X);
						X.saveContext(i, chrono.getElapsedTime());
						chrono.resetRecordTime();
					}
				}
			}

			if (_quit)
			{
				checkError(X);
				X.saveContext(i, chrono.getElapsedTime());
				return false;
			}
		}

		uint64_t res64;
		const bool isPrime = X.isMinusOne(res64);
		checkError(X);

		// Gerbicz last check point is i % L == 0 and i >= n - L
		// It is extended to i % L == 0 and i >= n
		for (uint32_t i = n; true; ++i)
		{
			X.square();

			if ((i & (L - 1)) == 0)
			{
				X.Gerbicz_check(L);
				checkError(X);
				break;
			}
		}

		if (_isBoinc) boinc_fraction_done(1.0);

		const std::string res = (isPrime) ? "                        " : std::string(", RES64 = ") + res64String(res64);
		const std::string runtime = timer::formatTime(chrono.getElapsedTime());

		std::ostringstream ssr; ssr << k << " * 2^" << n << " + 1 is " << (isPrime ? "prime" : "composite")
			 << ", a = " << a << ", time = " << runtime << res << std::endl;

		pio::display(std::string("\r") + ssr.str());
		pio::result(ssr.str());

		if (_isBoinc)
		{
			std::ostringstream sso; sso << k << " * 2^" << n << " + 1 is complete, a = " << a << ", time = " << runtime << std::endl;
			pio::print(sso.str());
		}

		if (checkRes && (res64 != r64))
		{
			std::ostringstream ss; ss << res64String(res64) << " != " << res64String(r64);
			throw std::runtime_error(ss.str());
		}

		return true;
	}

public:
	bool check_gfn(const uint32_t k, const uint32_t n, engine & engine)
	{
		gpmp X(k, n, engine, false);

		chronometer chrono;
		uint32_t i0;
		const bool found = false;	// TODO: X.restoreContext(i0, chrono.previousTime);

		std::ostringstream sst; sst << "GFN divisibility: " << std::endl;
		sst << (found ? "Resuming from a checkpoint " : "Testing ");
		sst << k << " * 2^" << n << " + 1, " << X.getDigits() << " digits, size = 2^" << arith::log2(X.getSize())
			<< " x " << X.getDigitBit() << " bits, plan: " << X.getPlanString() << std::endl;
		pio::print(sst.str());

		chrono.resetTime();

		if (!found)
		{
			i0 = 0;
			chrono.previousTime = 0;
			X.init(2, 2);
			X.copy_x_v();
		}

		const uint32_t L = 1 << (arith::log2(n) / 2);

		const uint32_t benchCnt = benchCount(n);
		uint32_t benchIter = benchCnt;
		chrono.resetBenchTime();
		chrono.resetRecordTime();

		uint32_t fermat_m = 0;

		// X = X^(2^(n - 1))
		for (uint32_t i = i0 + 1; i < n; ++i)
		{
			X.square();

			if (--benchIter == 0)
			{
				const double elapsedTime = chrono.getBenchTime();
				const double mulTime = elapsedTime / benchCnt, estimatedTime = mulTime * (n - i);
				std::ostringstream ssb; ssb << std::setprecision(3) << " " << i * 100.0 / n << "% done, "
					<< timer::formatTime(estimatedTime) << " remaining, " <<  mulTime * 1e3 << " ms/mul.";
				ssb << "        \r";
				pio::display(ssb.str());
				chrono.resetBenchTime();
				benchIter = benchCnt;
			}

			// Robert Gerbicz error checking algorithm
			// u is d(t) and v is u(0). They must be set before the loop
			// if (i == n - 1) X.set_bug();	// test
			if ((i & (L - 1)) == 0)
			{
				X.Gerbicz_step();
			}

			if (i + 50 >= n)
			{
				if (fermat_m == 0)
				{
					uint64_t res64;
					if (X.isMinusOne(res64))
					{
						checkError(X);
						fermat_m = i;
					}
				}
			}
			else
			{
				if (i % 1024 == 0)
				{
					const double elapsedTime = chrono.getRecordTime();
					if (elapsedTime > 600)
					{
						checkError(X);
						X.saveContext(i, chrono.getElapsedTime());
						chrono.resetRecordTime();
					}
				}

				if (_quit)
				{
					checkError(X);
					X.saveContext(i, chrono.getElapsedTime());
					return false;
				}
			}
		}

		for (uint32_t i = n; true; ++i)
		{
			X.square();

			if ((i & (L - 1)) == 0)
			{
				X.Gerbicz_check(L);
				checkError(X);
				break;
			}
		}

		const std::string runtime = timer::formatTime(chrono.getElapsedTime());

		std::ostringstream ssr; ssr << k << " * 2^" << n << " + 1 ";
		if (fermat_m != 0)
		{
			ssr << "divides F(" << fermat_m << ")";
		}
		else 
		{
			ssr << "doesn't divide any Fermat number";
		}
		ssr << ", time = " << runtime << std::endl;

		pio::display(std::string("\r") + ssr.str());
		pio::result(ssr.str());

		return true;
	}

protected:
	struct number
	{
		uint32_t k, n;
		uint64_t res64;
		number(const uint32_t k, const uint32_t n, const uint64_t res64 = 0) : k(k), n(n), res64(res64) {}
	};

public:
	void bench(engine & engine)
	{
		std::vector<number>	benchList;
		benchList.push_back(number(7649,     1553995));		// PPSE
		benchList.push_back(number(595,      2833406));		// PPS
		benchList.push_back(number(13,       5523860));		// DIV
		benchList.push_back(number(6679881,  6679881));		// Cullen
		benchList.push_back(number(3,       10829346));		// 321
		benchList.push_back(number(99739,   14019102));		// ESP
		benchList.push_back(number(168451,  19375200));		// PSP
		benchList.push_back(number(10223,   31172165));		// SOB

		for (const auto & b : benchList) if (!check(b.k, b.n, engine, true)) return;
	}

	friend class proth_test;
};
