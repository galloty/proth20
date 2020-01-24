/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "ocl.h"
#include "arith.h"
#include "gpmp.h"
#include "timer.h"

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

	void quit() { _quit = true; }

private:
	volatile bool _quit = false;

private:
	static std::string res64String(const uint64_t res64)
	{
		std::stringstream ss; ss << std::uppercase << std::hex << std::setfill('0') << std::setw(16) << res64;
		return ss.str();
	}

private:
	static void checkError(gpmp & X)
	{
		if (X.getError() != 0) throw std::runtime_error("GPU error detected!");
	}

public:
	static void apowk(gpmp & X, const uint32_t a, const uint32_t k)
	{
		// X = a^k, left-to-right algorithm
		bool s = false;
		X.init(a);						// x = 1, u = a
		X.setMultiplicand();			// x = 1, tu = NTT(u)
		for (int b = 0; b < 32; ++b)
		{
			if (s) X.square();			// x = iNTT(NTT(x)^2)

			if ((k & (uint32_t(1) << (31 - b))) != 0)
			{
				s = true;
				X.mul();				// x = iNTT(NTT(x).tu)
			}
		}
		X.norm();
		X.copy_x_v();

		// X = a^k, right-to-left algorithm
		X.init(a);						// x = 1, u = a
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
		}
		X.norm();
		X.copy_x_u();
		X.compare_u_v();
	}

public:
	bool check(const uint32_t k, const uint32_t n, engine & engine, const bool bench = false, const bool checkRes = false, const uint64_t r64 = 0)
	{
		uint32_t a = 0;
		if (!arith::proth_prime_quad_nonres(k, n, 3, a))
		{
			std::cout << k << " * 2^" << n << " + 1 is divisible by " << a << std::endl;
			return true;
		}

		gpmp X(k, n, engine);

		chronometer chrono;
		uint32_t i0;
		const bool found = X.restoreContext(i0, chrono.previousTime);

		std::cout << (found ? "Resuming from a checkpoint " : "Testing ");
		std::cout << k << " * 2^" << n << " + 1, " << X.getDigits() << " digits, size = 2^" << arith::log2(X.getSize()) << " x " << X.getDigitBit() << " bits" << std::endl;

		chrono.resetTime();

		if (!found)
		{
			i0 = 0;
			chrono.previousTime = 0;
			apowk(X, a, k);
			checkError(X);	// Sync GPU before benchmark
		}

		const uint32_t L = 1 << (arith::log2(n) / 2);

		const uint32_t benchCount = (n < 100000) ? 50000 : 50000000 / (n / 1000);
		uint32_t benchIter = benchCount;
		chrono.resetBenchTime();
		chrono.resetRecordTime();

		// X = X^(2^(n - 1))
		for (uint32_t i = i0 + 1; i < n; ++i)
		{
			X.square();

			if (--benchIter == 0)
			{
				if (bench) checkError(X);	// Sync GPU
				const double elapsedTime = chrono.getBenchTime();
				const double mulTime = elapsedTime / benchCount, estimatedTime = mulTime * (n - i);
				std::cout << std::setprecision(3) << " " << i * 100.0 / n << "% done, "
					<< timer::formatTime(estimatedTime) << " remaining, " <<  mulTime * 1e3 << " ms/mul.";
				if (bench)
				{
					std::cout << std::endl;
					return true;
				}
				std::cout << "        \r" << std::flush;
				benchIter = benchCount;
				chrono.resetBenchTime();
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

		const std::string res = (isPrime) ? "                        " : std::string(", RES64 = ") + res64String(res64);

		std::stringstream ss; ss << k << " * 2^" << n << " + 1 is " << (isPrime ? "prime" : "composite")
			 << ", a = " << a << ", time = " << timer::formatTime(chrono.getElapsedTime()) << res << std::endl;

		std::cout << "\r" << ss.str();
		std::ofstream resFile("presults.txt", std::ios::app);
		if (resFile.is_open())
		{
			resFile << ss.str();
			resFile.close();
		}

		if (checkRes)
		{
			if (res64 != r64) std::cout << "Error: " << res64String(res64) << " != " << res64String(r64) << std::endl;
		}

		return true;
	}

public:
	bool validate(const uint32_t k, const uint32_t n, engine & engine)
	{
		const uint32_t a = 3;
		gpmp X(k, n, engine);

		std::cout << "Testing " << k << " * 2^" << n << " + 1, size = 2^" << arith::log2(X.getSize()) << " x " << X.getDigitBit() << " bits ";

		apowk(X, a, k);
		checkError(X);

		const uint32_t L = 64;

		for (uint32_t i = 1; i < L * L; ++i)
		{
			X.square();
			if ((i & (L - 1)) == 0) X.Gerbicz_step();
			if (_quit) return false;
		}

		X.square();
		X.Gerbicz_check(L);
		checkError(X);
		std::cout << "valid" << std::endl;
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
		primeList.push_back(number(1027, 21468));		// size = 2^11, square32
		primeList.push_back(number(1109, 42921));		// size = 2^12, square64
		primeList.push_back(number(1085, 85959));		// size = 2^13, square128	64-bit 32-bit
		primeList.push_back(number(1015, 171214));		// size = 2^14, square256,  0.073  0.072
		primeList.push_back(number(1197, 343384));		// size = 2^15, square512,  0.107  0.101
		primeList.push_back(number(1089, 685641));		// size = 2^16, square1024, 0.170  0.168
		primeList.push_back(number(1005, 1375758));		// size = 2^17, square32,   0.315  0.306	square2048	0.300
		primeList.push_back(number(1089, 2746155));		// size = 2^18, square64,   0.580  0.556	square4096	0.594
		primeList.push_back(number(45, 5308037));		// size = 2^19, square128,  1.09   1.06
		primeList.push_back(number(6679881, 6679881));	// size = 2^20, square256,  2.15
		primeList.push_back(number(3, 10829346));		// size = 2^21, square512,  4.43
		primeList.push_back(number(10223, 31172165));	// size = 2^22, square1024, 8.96

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
		for (uint32_t n = 15000; n < 100000000; n *= 2) if (!validate(536870911, n, engine)) return;
	}

	void profile(const uint32_t k, const uint32_t n, engine & engine)
	{
		gpmp X(k, n, engine, true);
		const size_t count = 1000;
		for (size_t i = 0; i < count; ++i) X.square();
		std::cout << "Size = " << X.getSize() << std::endl;
		engine.displayProfiles(count);

		// Size = 524288
		// - sub_ntt64: 1, 11.6 %, 124732 (124732)
		// - ntt64: 1, 10.3 %, 109968 (109968)
		// - intt64: 2, 22.3 %, 238608 (119304)
		// - square128: 1, 21.1 %, 225818 (225818)
		// NTT: 65.3 %
		// - poly2int0: 1, 12.1 %, 130266 (130266)
		// - poly2int1: 1, 3.99 %, 42799 (42799)
		// POLY2INT: 16.1 %
		// - reduce_upsweep64: 2, 2.38 %, 25499 (12749)
		// - reduce_downsweep64: 2, 3.37 %, 36128 (18064)
		// - reduce_topsweep64: 1, 0.355 %, 3807 (3807)
		// - reduce_i: 1, 5.16 %, 55344 (55344)
		// - reduce_o: 1, 7.13 %, 76430 (76430)
		// - reduce_f: 1, 0.274 %, 2943 (2943)
		// REDUCE: 18.6 %

		// Size = 2097152
		// - sub_ntt64: 1, 10.6 %, 466566 (466566)
		// - ntt64: 1, 12 %, 525783 (525783)
		// - intt64: 2, 23.2 %, 1018108 (509054)
		// - square512: 1, 23.9 %, 1050873 (1050873)
		// NTT: 69.7 %
		// - poly2int0: 1, 10.9 %, 480425 (480425)
		// - poly2int1: 1, 3.41 %, 149497 (149497)
		// POLY2INT: 14.3 %
		// - reduce_upsweep64: 2, 1.81 %, 79589 (39794)
		// - reduce_downsweep64: 2, 2.83 %, 124330 (62165)
		// - reduce_topsweep256: 1, 0.103 %, 4507 (4507)
		// - reduce_i: 1, 4.91 %, 215512 (215512)
		// - reduce_o: 1, 6.18 %, 271477 (271477)
		// - reduce_f: 1, 0.065 %, 2853 (2853)
		// REDUCE: 16.0 %
	}
};
