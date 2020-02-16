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

private:
	volatile bool _quit = false;
	bool _isBoinc = false;

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

private:
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

public:
	bool validate(const uint32_t k, const uint32_t n, const uint32_t L, engine & engine)
	{
		const uint32_t a = 3;
		gpmp X(k, n, engine, _isBoinc, false);

		std::ostringstream sst;
		sst << "Testing " << k << " * 2^" << n << " + 1, size = 2^" << arith::log2(X.getSize()) << " x " << X.getDigitBit() << " bits" << std::endl;
		pio::display(sst.str());

		const size_t cntSq = X.getPlanSquareSeqCount(), cntP2i = X.getPlanPoly2intCount();

		for (size_t j = 0, cnt = std::max(cntSq, cntP2i); j < cnt; ++j)
		{
			X.setPlanSquareSeq(j % cntSq);
			X.setPlanPoly2intFn(j % cntP2i);

			pio::display(X.getPlanString());

			if (!apowk(X, a, k)) return false;
			checkError(X);

			for (uint32_t i = 1; i < L * L; ++i)
			{
				X.square();
				if ((i & (L - 1)) == 0) X.Gerbicz_step();
				if (_quit) return false;
			}

			X.square();
			X.Gerbicz_check(L);
			checkError(X);
			std::ostringstream ss; ss << " valid" << std::endl;
			pio::display(ss.str());
		}

		return true;
	}

private:
	struct number
	{
		uint32_t k, n;
		uint64_t res64;
		number(const uint32_t k, const uint32_t n, const uint64_t res64 = 0) : k(k), n(n), res64(res64) {}
	};

public:
	void test_prime(engine & engine, const bool bench = false)
	{
		std::vector<number>	primeList;
		primeList.push_back(number(1035, 301));
		primeList.push_back(number(955, 636));
		primeList.push_back(number(969, 1307));
		primeList.push_back(number(1139, 2641));
		primeList.push_back(number(1035, 5336));
		primeList.push_back(number(965, 10705));
		primeList.push_back(number(1027, 21468));		// size = 2^11
		primeList.push_back(number(1109, 42921));		// size = 2^12
		primeList.push_back(number(1085, 85959));		// size = 2^13  64-bit 32-bit
		primeList.push_back(number(1015, 171214));		// size = 2^14, 0.071  0.070  256_16 sq_64 p2i_16_16
		primeList.push_back(number(1197, 343384));		// size = 2^15, 0.097  0.094  256_16 sq_128 p2i_8_32
		primeList.push_back(number(1089, 685641));		// size = 2^16, 0.163  0.156  256_8 sq_256 p2i_8_64
		primeList.push_back(number(1005, 1375758));		// size = 2^17, 0.288  0.280  256_8 sq_512 p2i_8_64
		primeList.push_back(number(1089, 2746155));		// size = 2^18, 0.537  0.523  256_8 sq_1024 p2i_8_64
		primeList.push_back(number(45, 5308037));		// size = 2^19, 1.03   1.01   256_8 sq_2048 p2i_8_32
		primeList.push_back(number(6679881, 6679881));	// size = 2^20, 2.04   1.96   256_8 256_8 sq_16 p2i_8_32
		primeList.push_back(number(3, 10829346));		// size = 2^21, 4.13   3.95   256_16 256_8 sq_32 p2i_8_32
		primeList.push_back(number(10223, 31172165));	// size = 2^22, 8.37   8.00   256_16 256_8 sq_64 p2i_8_32

		for (const auto & p : primeList) if (!check(p.k, p.n, engine, bench, true)) return;
	}

	void test_composite(engine & engine, const bool bench = false)
	{
		std::vector<number>	compositeList;
		compositeList.push_back(number(536870911,    298, 0x35461D17F60DA78Aull));
		compositeList.push_back(number(536870905,    626, 0x06543644B033FF0Cull));
		compositeList.push_back(number(536870411,   1307, 0x7747B2D2351394EFull));
		compositeList.push_back(number(536850911,   2631, 0x3A08775B698EEB34ull));
		compositeList.push_back(number(536870911,   5336, 0xDA3B38B4E68F0445ull));
		compositeList.push_back(number(536770911,  10705, 0x030EECBE0A5E77A6ull));
		compositeList.push_back(number(526870911,  21432, 0xD86853C587F1D537ull));
		compositeList.push_back(number(436870911,  42921, 0x098AD2BD01F485BCull));
		compositeList.push_back(number(535970911,  85942, 0x2D19C7E7E7553AD6ull));
		compositeList.push_back(number(536860911, 171213, 0x99EFB220EE2289A0ull));
		compositeList.push_back(number(536870911, 343386, 0x5D6A1D483910E48Full));
		compositeList.push_back(number(536870911, 685618, 0x84C7E4E7F1344902ull));

		// check residues
		for (const auto & c : compositeList) if (!check(c.k, c.n, engine, bench, true, c.res64)) return;
	}

	void test_gfn(engine & engine)
	{
		std::vector<number>	gfnDivList;
		gfnDivList.push_back(number(332436749, 9865));
		gfnDivList.push_back(number(5, 23473));
		gfnDivList.push_back(number(165, 49095));
		gfnDivList.push_back(number(189, 90061));
		gfnDivList.push_back(number(3, 213321));
		gfnDivList.push_back(number(3, 382449));

		for (const auto & d : gfnDivList) if (!check_gfn(d.k, d.n, engine)) return;
	}

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

	void validation(engine & engine)
	{
		if (!validate(536870911, 3000000, 4, engine)) return;
		for (uint32_t n = 15000; n < 100000000; n *= 2) if (!validate(536870911, n, 16, engine)) return;
	}

	void profile(const uint32_t k, const uint32_t n, engine & engine)
	{
		gpmp X(k, n, engine, _isBoinc, true, true);
		const size_t count = 1000;
		for (size_t i = 0; i < count; ++i) X.square();
		std::ostringstream ss; ss << "Size = " << X.getSize() << std::endl;
		pio::display(ss.str());
		engine.displayProfiles(count);

		// Size = 1048576 (DIV)
		// - sub_ntt256_16: 1, 13.5 %, 272001 (272001)
		// - ntt256_4: 1, 13.8 %, 277615 (277615)
		// - intt256_4: 1, 13.8 %, 279031 (279031)
		// - intt256_16: 1, 14.7 %, 296904 (296904)
		// - square16: 1, 11.1 %, 223983 (223983)
		// NTT: 66.9 %
		// - poly2int0_8_32: 1, 9.98 %, 201411 (201411)
		// - poly2int1_8: 1, 5.34 %, 107726 (107726)
		// - poly2int2: 1, 0.135 %, 2721 (2721)
		// POLY2INT: 15.5 %
		// - reduce_upsweep64: 2, 2.17 %, 43754 (21877)
		// - reduce_downsweep64: 2, 3.2 %, 64479 (32239)
		// - reduce_topsweep128: 1, 0.221 %, 4468 (4468)
		// - reduce_i: 1, 5.42 %, 109407 (109407)
		// - reduce_o: 1, 6.52 %, 131654 (131654)
		// - reduce_f: 1, 0.134 %, 2703 (2703)
		// REDUCE: 17.6 %

		// Size = 4194304 (SOB)
		// - sub_ntt256_16: 1, 12.8 %, 1053504 (1053504)
		// - ntt256_8: 1, 13.1 %, 1080382 (1080382)
		// - intt256_8: 1, 13 %, 1073405 (1073405)
		// - intt256_16: 1, 13.9 %, 1140741 (1140741)
		// - square64: 1, 15.8 %, 1302400 (1302400)
		// NTT: 68.6 %
		// - poly2int0_16_32: 1, 11.2 %, 923495 (923495)
		// - poly2int1_16: 1, 3.19 %, 262220 (262220)
		// - poly2int2: 1, 0.0329 %, 2705 (2705)
		// POLY2INT: 14.5 %
		// - reduce_upsweep64: 2, 1.87 %, 153516 (76758)
		// - reduce_downsweep64: 2, 2.89 %, 238101 (119050)
		// - reduce_topsweep512: 1, 0.0635 %, 5222 (5222)
		// - reduce_i: 1, 5.22 %, 429361 (429361)
		// - reduce_o: 1, 6.79 %, 558722 (558722)
		// - reduce_f: 1, 0.0321 %, 2637 (2637)
		// REDUCE: 16.9 %
	}
};
