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

	static const uint32_t ord2_max = 30;

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

private:
	static void printStatus(gpmp & X, const bool found, const uint32_t k, const uint32_t n)
	{
		std::ostringstream ss; ss << (found ? "Resuming from a checkpoint " : "Testing ");
		ss << k << " * 2^" << n << " + 1, " << X.getDigits() << " digits, size = 2^" << arith::log2(X.getSize())
			<< " x " << X.getDigitBit() << " bits, plan: " << X.getPlanString() << std::endl;
		pio::print(ss.str());
	}

private:
	static void printProgress(chronometer & chrono, const uint32_t i, const uint32_t n, const uint32_t benchCnt)
	{
		const double elapsedTime = chrono.getBenchTime();
		const double mulTime = elapsedTime / benchCnt, estimatedTime = mulTime * (n - i);
		std::ostringstream ss; ss << std::setprecision(3) << " " << i * 100.0 / n << "% done, "
			<< timer::formatTime(estimatedTime) << " remaining, " <<  mulTime * 1e3 << " ms/mul.        \r";
		pio::display(ss.str());
		chrono.resetBenchTime();
	}

public:
	bool apowk(gpmp & X, const uint32_t a, const uint32_t k) const
	{
		// X = a^k, left-to-right algorithm
		bool s = false;
		X.init(a, a);					// x = a, u = a
		X.setMultiplicand();			// x = a, tu = NTT(u)
		for (int b = 0; b < 32; ++b)
		{
			if (s) X.square();			// x = iNTT(NTT(x)^2)

			if ((k & (uint32_t(1) << (31 - b))) != 0)
			{
				if (s) X.mul();			// x = iNTT(NTT(x).tu)
				s = true;
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
	bool check(const uint32_t k, const uint32_t n, engine & engine, const bool checkRes = false, const uint64_t r64 = 0)
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
		const bool found = X.restoreContext(i0, chrono.previousTime, "p");
		printStatus(X, found, k, n);

		chrono.resetTime();

		if (!found)
		{
			i0 = 0;
			chrono.previousTime = 0;
			// X = a^k
			if (!apowk(X, a, k)) return false;
			checkError(X);
		}

		const uint32_t L = 1 << (arith::log2(n) / 2);

		const uint32_t benchCnt = benchCount(n);
		uint32_t benchIter = benchCnt;
		chrono.resetBenchTime();
		chrono.resetRecordTime();

		if (_isBoinc) boinc_fraction_done(double(i0) / double(n));

		// X = X^{2^{n - 1}}
		for (uint32_t i = i0 + 1; i < n; ++i)
		{
			X.square();

			if (--benchIter == 0)
			{
				if (_isBoinc) boinc_fraction_done(double(i) / double(n));
				else printProgress(chrono, i, n, benchCnt);
				benchIter = benchCnt;
			}

			// Robert Gerbicz error checking algorithm
			// u is d(t) and v is u(0). They must be set before the loop
			// if (i == n - 1) X.set_bug();	// test
			if ((i & (L - 1)) == 0) X.Gerbicz_step();

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
						X.saveContext(i, chrono.getElapsedTime(), "p");
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
						X.saveContext(i, chrono.getElapsedTime(), "p");
						boinc_checkpoint_completed();
					}
				}
				else
				{
					const double elapsedTime = chrono.getRecordTime();
					if (elapsedTime > 600)
					{
						checkError(X);
						X.saveContext(i, chrono.getElapsedTime(), "p");
						chrono.resetRecordTime();
					}
				}
			}

			if (_quit)
			{
				checkError(X);
				X.saveContext(i, chrono.getElapsedTime(), "p");
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
	bool check_order(const uint32_t k, const uint32_t n, const uint32_t a, engine & engine)
	{
		std::ostringstream sst; sst << "Multiplicative order of " << a << ": " << std::endl;
		pio::print(sst.str());

		gpmp X(k, n, engine, false);

		const std::string ext = std::string("o_") + std::to_string(a);
		chronometer chrono;
		uint32_t i0;
		const bool found = X.restoreContext(i0, chrono.previousTime, ext.c_str());
		printStatus(X, found, k, n);

		chrono.resetTime();

		if (!found)
		{
			i0 = 0;
			chrono.previousTime = 0;
			// X = a
			X.init(a, 0);
		}

		const uint32_t benchCnt = benchCount(n);
		uint32_t benchIter = benchCnt;
		chrono.resetBenchTime();
		chrono.resetRecordTime();

		for (uint32_t i = i0 + 1; i <= n - ord2_max; ++i)
		{
			X.square();

			if (--benchIter == 0)
			{
				printProgress(chrono, i, n, benchCnt);
				benchIter = benchCnt;
			}

			if (i % 1024 == 0)
			{
				const double elapsedTime = chrono.getRecordTime();
				if (elapsedTime > 600)
				{
					checkError(X);
					X.saveContext(i, chrono.getElapsedTime(), ext.c_str());
					chrono.resetRecordTime();
				}
			}

			if (_quit)
			{
				checkError(X);
				X.saveContext(i, chrono.getElapsedTime(), ext.c_str());
				return false;
			}
		}

		X.copy_x_v();
		// X = a^{2^n}
		for (uint32_t i = 0; i < ord2_max; ++i) X.square();
		X.swap_x_v();

		// X = a^{k.2^{n - ord2_max}}, V = a^{2^n}
		X.pow(k);

		if (X.isOne()) throw std::runtime_error("Multiplicative order computation failed!");

		std::vector<std::pair<uint32_t, uint32_t>> fac_e, fac;
		arith::factor(k, fac);

		uint32_t e = n - ord2_max;
		while (!X.isOne())
		{
			X.square();
			++e;
		}
		if (e != 0) fac_e.push_back(std::make_pair(2, e));

		for (const auto & f : fac)
		{
			const uint32_t pi = f.first, ei = f.second;

			uint64_t E = k;
			for (uint32_t i = 0; i < ei; ++i) E /= pi;

			X.copy_v_x();
			if (E > 1) X.pow(E);

			uint32_t e = 0;
			while (!X.isOne())
			{
				X.pow(pi);
				++e;
			}
			if (e != 0) fac_e.push_back(std::make_pair(pi, e));
		}

		const std::string runtime = timer::formatTime(chrono.getElapsedTime());

		std::ostringstream ssr; ssr << "p = " << k << " * 2^" << n << " + 1, ord_p(" << a << ") = " << fac_e[0].first << "^" << fac_e[0].second;
		for (size_t i = 1; i < fac_e.size(); ++i)
		{
			ssr << " * " << fac_e[i].first;
			if (fac_e[i].second > 1) ssr << "^" << fac_e[i].second;
		}
		ssr << ", time = " << runtime << std::endl;

		pio::display(std::string("\r") + ssr.str());
		pio::result(ssr.str());

		return true;
	}

private:
	bool _check_gfn(gpmp & X, const uint32_t k, const uint32_t n, const uint32_t a)
	{
		const std::string ext = std::string("f_") + std::to_string(a);
		chronometer chrono;
		uint32_t i0;
		const bool found = X.restoreContext(i0, chrono.previousTime, ext.c_str());
		if (a == 2) printStatus(X, found, k, n);

		chrono.resetTime();

		if (!found)
		{
			i0 = 0;
			chrono.previousTime = 0;
			// X = a
			X.init(a, 0);
		}

		const uint32_t benchCnt = benchCount(n);
		uint32_t benchIter = benchCnt;
		chrono.resetBenchTime();
		chrono.resetRecordTime();

		uint32_t m = 0;

		// X = X^{2^n}
		for (uint32_t i = i0 + 1; i <= n; ++i)
		{
			X.square();

			if (--benchIter == 0)
			{
				printProgress(chrono, i, n, benchCnt);
				benchIter = benchCnt;
			}

			if (i + ord2_max >= n)
			{
				if (i + ord2_max == n)
				{
					checkError(X);
					X.saveContext(i, chrono.getElapsedTime(), ext.c_str());
				}

				if (m == 0)
				{
					uint64_t res64;
					if (X.isMinusOne(res64))
					{
						checkError(X);
						m = i;
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
						X.saveContext(i, chrono.getElapsedTime(), ext.c_str());
						chrono.resetRecordTime();
					}
				}

				if (_quit)
				{
					checkError(X);
					X.saveContext(i, chrono.getElapsedTime(), ext.c_str());
					return false;
				}
			}
		}

		X.pow(k);
		bool isOne = X.isOne();
		checkError(X);
		if (!isOne) throw std::runtime_error("GFN divisibility test failed!");

		const std::string runtime = timer::formatTime(chrono.getElapsedTime());

		std::ostringstream ssr; ssr << k << " * 2^" << n << " + 1 ";
		if (a == 2)
		{
			if (m != 0) ssr << "divides F_" << m;
			else ssr << "doesn't divide any Fermat number";
		}
		else
		{
			if (m != 0) ssr << "divides F_" << m << "(" << a << ")";
			else ssr << "doesn't divide any F_m(" << a << ")";

		}
		ssr << ", time = " << runtime << std::endl;

		pio::display(std::string("\r") + ssr.str());
		pio::result(ssr.str());
		return true;
	}

private:
	bool _check_gfn(gpmp & X, const uint32_t k, const uint32_t n, const uint32_t a1, const uint32_t e1, const uint32_t a2, const uint32_t e2)
	{
		const std::string ext1 = std::string("f_") + std::to_string(a1);
		const std::string ext2 = std::string("f_") + std::to_string(a2);

		bool ok = true;
		uint32_t i0; double time;
		ok &= X.restoreContext(i0, time, ext1.c_str(), false);
		ok &= (i0 + ord2_max == n);
		if (e1 > 1) X.pow(e1);
		if (a2 > 1)
		{
			X.copy_x_u();
			ok &= X.restoreContext(i0, time, ext2.c_str(), false);
			ok &= (i0 + ord2_max == n);
			if (!ok) throw std::runtime_error("GFN divisibility test failed!");
			if (e2 > 1) X.pow(e2);
			X.setMultiplicand();
			X.mul();
		}

		uint32_t m = 0;

		// X = X^{2^n}
		for (uint32_t i = n - ord2_max + 1; i <= n; ++i)
		{
			X.square();

			if (m == 0)
			{
				uint64_t res64;
				if (X.isMinusOne(res64))
				{
					checkError(X);
					m = i;
				}
			}
		}

		X.pow(k);
		bool isOne = X.isOne();
		checkError(X);
		if (!isOne) throw std::runtime_error("GFN divisibility test failed!");

		uint32_t a = 1;
		for (uint32_t i = 0; i < e1; ++i) a *= a1;
		for (uint32_t i = 0; i < e2; ++i) a *= a2;

		std::ostringstream ssr; ssr << k << " * 2^" << n << " + 1 ";
		if (m != 0) ssr << "divides F_" << m << "(" << a << ")";
		else ssr << "doesn't divide any F_m(" << a << ")";
		ssr << std::endl;

		pio::display(std::string("\r") + ssr.str());
		pio::result(ssr.str());
		return true;
	}

public:
	bool check_gfn(const uint32_t k, const uint32_t n, engine & engine)
	{
		std::ostringstream sst; sst << "GFN divisibility: " << std::endl;
		pio::print(sst.str());

		gpmp X(k, n, engine, false);

		if (!_check_gfn(X, k, n, 2)) return false;
		if (!_check_gfn(X, k, n, 3)) return false;
		if (!_check_gfn(X, k, n, 5)) return false;
		if (!_check_gfn(X, k, n, 7)) return false;
		if (!_check_gfn(X, k, n, 11)) return false;

		if (!_check_gfn(X, k, n, 2, 1, 3, 1)) return false;	// 6
		if (!_check_gfn(X, k, n, 2, 3, 1, 1)) return false;	// 8
		if (!_check_gfn(X, k, n, 2, 1, 5, 1)) return false;	// 10
		if (!_check_gfn(X, k, n, 2, 2, 3, 1)) return false;	// 12

		return true;
	}

	friend class proth_test;
};
