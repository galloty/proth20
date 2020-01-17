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

		gpmp X(k, n, device);	// X = 1

		std::cout << "Testing " << k << " * 2^" << n << " + 1, " << X.getDigits() << " digits (size = "	<< X.getSize() << ")," << std::flush;

		// X *= a^k, left-to-right algorithm
		bool s = false;
		X.setMultiplicand(a);
		for (int b = 0; b < 32; ++b)
		{
			if (s) X.square();

			if ((k & (uint32_t(1) << (31 - b))) != 0)
			{
				s = true;
				X.mul();
			}
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
		const int err = X.getError();
		if (err != 0)
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
		compositeList.push_back(Number(9999, 299,    0xB073C97A2450454Full));
		compositeList.push_back(Number(21, 636,      0x4FD4F9FE4C6E7C1Bull));
		compositeList.push_back(Number(4769, 1307,   0x8B5F4C7215F37871ull));
		compositeList.push_back(Number(9671, 2631,   0x0715EDFC4814B64Aull));
		compositeList.push_back(Number(19, 5336,     0x614B05AC60E508A0ull));
		compositeList.push_back(Number(963, 10705,   0x232BF76A98040BA3ull));
		compositeList.push_back(Number(6189, 21469,  0x88D2582BDDE8E7CAull));	// size = 2k
		compositeList.push_back(Number(2389, 42922,  0xE427B88330D2EE8Cull));	// size = 4k
		compositeList.push_back(Number(1295, 85959,  0x53D33CD949CC31DBull));	// size = 8k
		compositeList.push_back(Number(9273, 171214, 0xAEC1A38C0C4B1D98ull));	// size = 16k
		compositeList.push_back(Number(8651, 343387, 0xB832D18693CCB6BCull));	// size = 32k
		compositeList.push_back(Number(9999, 685619, 0xB151FAF87B6977C2ull));	// size = 64k

		// check residues
		for (const auto & c : compositeList) proth::check(c.k, c.n, device, bench, true, c.res64);
	}

	static void bench(ocl::Device & device)
	{
		// 1,375,000 - 2,750,000: size = 256k	PPSE, PPS
		// 2,750,000 - 5,500,000: size = 512k	DIV

		std::vector<Number>	benchList;
		benchList.push_back(Number(9501, 1553584));
		benchList.push_back(Number(1101, 2832061));
		benchList.push_back(Number(45, 5308037));

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
		// - sub_ntt64: 1, 11.3 %, 125281 (125281)
		// - ntt64: 1, 11.7 %, 129282 (129282)
		// - intt64: 2, 22.5 %, 249309 (124654)
		// - square128: 1, 20.6 %, 228637 (228637)
		// NTT: 66.1 %
		// - poly2int0: 1, 12.2 %, 135665 (135665)
		// - poly2int1: 1, 3.53 %, 39125 (39125)
		// POLY2INT: 15.7 %
		// - reduce_upsweep64: 2, 2.34 %, 25922 (12961)
		// - reduce_downsweep64: 2, 3.26 %, 36184 (18092)
		// - reduce_topsweep256: 1, 0.419 %, 4642 (4642)
		// - reduce_i: 1, 5 %, 55429 (55429)
		// - reduce_o: 1, 6.9 %, 76526 (76526)
		// - reduce_f: 1, 0.261 %, 2898 (2898)
		// REDUCE: 18.2 %
	}
};
