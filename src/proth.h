/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "ocl.h"
#include "gpmp.h"
#include "timer.h"

class proth
{
private:
	static int jacobi(const uint64_t x, const uint64_t y)	// y is an odd number
	{
		uint64_t m = x, n = y;

		int k = 1;
		while (true)
		{
			if (m == 0) return 0;	// (0/n) = 0

			// (2/n) = (-1)^((n^2-1)/8)
			bool odd = false;
			while (m % 2 == 0) { m /= 2; odd = !odd; }
			if (odd && (n % 8 != 1) && (n % 8 != 7)) k = -k;

			if (m == 1) return k;	// (1/n) = 1

			// (m/n)(n/m) = -1 iif m == n == 3 (mod 4)
			if ((m % 4 == 3) && (n % 4 == 3)) k = -k;
			const uint64_t t = n; n = m; m = t;

			m %= n;	// (m/n) = (m mod n / n)
		}
	}

private:
	static std::string res64String(const uint64_t res64)
	{
		std::stringstream ss; ss << std::uppercase << std::hex << std::setfill('0') << std::setw(16) << res64;
		return ss.str();
	}

public:
	static bool check(const uint32_t k, const uint32_t n, ocl::Device & device, const bool bench = false, const bool checkRes = false, const uint64_t r64 = 0)
	{
		const Timer::Time startTime = Timer::currentTime();

		// Proth's theorem: a such that (a/P) = -1
		// Note that P = k*2^n + 1 and a is odd => (a/P) * (P/a) = 1 if P = 1 (mod 4)
		// Then (P/a) = (P mod a / a)
		uint32_t a = 3;
		for (; a < 10000; a += 2)
		{
			uint32_t pmoda = k % a;
			if (pmoda == 0) continue;
			for (uint32_t i = 0; i < n; ++i) { pmoda += pmoda; if (pmoda >= a) pmoda -= a; }
			pmoda += 1; if (pmoda >= a) pmoda -= a;
			if (pmoda <= 1) continue;
			if (jacobi(pmoda, a) == -1) break;
		}
		if (a >= 10000) return false;

		gpmp X(k, n, device);

		std::cout << "Testing " << k << " * 2^" << n << " + 1, " << X.getDigits() << " digits (size = "	<< X.getSize() << ")," << std::flush;

		// X *= a^k, left-to-right algorithm
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
		X.copy_x_v();

		// X *= a^k, right-to-left algorithm
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
		X.compare_x_v();

		if (X.getError() != 0)	// Sync GPU before benchmark
		{
			std::cout << " error detected!" << std::endl;
			return false;
		}

		// X = X^(2^(n - 1))
		Timer::Time startBenchTime = Timer::currentTime();
		const uint32_t benchCount = (n < 100000) ? 20000 : 10000000 / (n / 1000);
		for (uint32_t i = 0; i < n - 1; ++i)
		{
			X.square();
			if (i == benchCount - 1)
			{
				X.getError();
				const double elapsedTime = Timer::diffTime(Timer::currentTime(), startBenchTime);
				const double mulTime = elapsedTime / benchCount, estimatedTime = mulTime * n;
				std::cout << " estimated time is " << Timer::formatTime(estimatedTime) << ", " << std::setprecision(3) << mulTime * 1e3 << " ms/mul." << std::flush;
				if (bench)
				{
					std::cout << std::endl;
					return true;
				}
			}
		}

		uint64_t res64;
		const bool isPrime = X.isMinusOne(res64);
		if (X.getError() != 0)
		{
			std::cout << " error detected!" << std::endl;
			return false;
		}

		const double elapsedTime = Timer::diffTime(Timer::currentTime(), startTime);

		const std::string res = (isPrime) ? "                        " : std::string(", RES64 = ") + res64String(res64);

		std::stringstream ss; ss << k << " * 2^" << n << " + 1 is " << (isPrime ? "prime" : "composite") << ", a = " << a << ", " 
			<< X.getDigits() << " digits (size = "	<< X.getSize() << "), time = " << Timer::formatTime(elapsedTime) << res << std::endl;

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
			return false;
		}

		return true;
	}

private:
	struct Number
	{
		uint32_t k, n;
		uint64_t res64;
		Number(const uint32_t k, const uint32_t n, const uint64_t res64 = 0) : k(k), n(n), res64(res64) {}
	};

public:
	static void test_prime(ocl::Device & device, const bool bench = false)
	{
		std::vector<Number>	primeList;
		primeList.push_back(Number(1035, 301));
		primeList.push_back(Number(955, 636, 2));
		primeList.push_back(Number(969, 1307));
		primeList.push_back(Number(1139, 2641));
		primeList.push_back(Number(1035, 5336));
		primeList.push_back(Number(965, 10705));
		primeList.push_back(Number(1027, 21468));	// size = 2k,   square32
		primeList.push_back(Number(1109, 42921));	// size = 4k,   square64
		primeList.push_back(Number(1085, 85959));	// size = 8k,   square128	64-bit 32-bit
		primeList.push_back(Number(1015, 171214));	// size = 16k,  square256,  0.072  0.072
		primeList.push_back(Number(1197, 343384));	// size = 32k,  square512,  0.105  0.101
		primeList.push_back(Number(1089, 685641));	// size = 64k,  square1024, 0.177  0.168
		primeList.push_back(Number(1005, 1375758));	// size = 128k, square32,   0.319  0.306
		primeList.push_back(Number(1089, 2746155));	// size = 256k, square64,   0.571  0.556
		primeList.push_back(Number(45, 5308037));	// size = 512k, square128,  1.09   1.06  ms

		for (const auto & p : primeList) proth::check(p.k, p.n, device, bench, true);
	}

	static void test_composite(ocl::Device & device, const bool bench = false)
	{
		std::vector<Number>	compositeList;
		compositeList.push_back(Number(536870911,    298, 0x35461D17F60DA78Aull));
		compositeList.push_back(Number(536870905,    626, 0x06543644B033FF0Cull));
		compositeList.push_back(Number(536870411,   1307, 0x7747B2D2351394EFull));
		compositeList.push_back(Number(536850911,   2631, 0x3A08775B698EEB34ull));
		compositeList.push_back(Number(536870911,   5336, 0xDA3B38B4E68F0445ull));
		compositeList.push_back(Number(536770911,  10705, 0x030EECBE0A5E77A6ull));
		compositeList.push_back(Number(526870911,  21432, 0xD86853C587F1D537ull));	// size = 2k
		compositeList.push_back(Number(436870911,  42921, 0x098AD2BD01F485BCull));	// size = 4k
		compositeList.push_back(Number(535970911,  85942, 0x2D19C7E7E7553AD6ull));	// size = 8k
		compositeList.push_back(Number(536860911, 171213, 0x99EFB220EE2289A0ull));	// size = 16k
		compositeList.push_back(Number(536870911, 343386, 0x5D6A1D483910E48Full));	// size = 32k
		compositeList.push_back(Number(536870911, 685618, 0x84C7E4E7F1344902ull));	// size = 64k

		// check residues
		for (const auto & c : compositeList) proth::check(c.k, c.n, device, bench, true, c.res64);
	}

	static void bench(ocl::Device & device)
	{
		std::vector<Number>	benchList;
		benchList.push_back(Number(7649,     1553995));		// PPSE
		benchList.push_back(Number(595,      2833406));		// PPS
		benchList.push_back(Number(45,       5308037));		// DIV
		// benchList.push_back(Number(6679881,  6679881));		// Cullen
		// benchList.push_back(Number(3,       10829346));		// 321
		// benchList.push_back(Number(99739,   14019102));		// ESP
		// benchList.push_back(Number(168451,  19375200));		// PSP
		// benchList.push_back(Number(10223,   31172165));		// SOB

		for (const auto & b : benchList) proth::check(b.k, b.n, device, true);
		std::cout << std::endl;
		for (const auto & b : benchList) proth::check(b.k, b.n, device);
	}

	static void profile(const uint32_t k, const uint32_t n, ocl::Device & device)	// ocl_profile must be defined (ocl.h)
	{
		gpmp X(k, n, device);
		const size_t pCount = 1000;
		for (size_t i = 0; i < pCount; ++i) X.square();
		std::cout << "Size = " << X.getSize() << std::endl;
		device.displayProfiles(pCount);

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
	}
};
