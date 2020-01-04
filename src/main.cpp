/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include "ocl.h"
#include "gpmp.h"
#include "timer.h"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>

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

class Application
{
private:
	static std::string resString(const uint64_t res64)
	{
		std::stringstream ss; ss << std::uppercase << std::hex << std::setfill('0') << std::setw(16) << res64;
		return ss.str();
	}

private:
	static uint64_t check(const uint32_t k, const uint32_t n, ocl::Device & device, const bool bench = false)
	{
		std::cout << "Testing " << k << " * 2^" << n << " + 1" << std::flush;

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
		if (a >= 10000) return 0;

		gpmp X(k, n, device);

		// X *= a^k, left-to-right algorithm
		bool s = false;
		for (int b = 0; b < 32; ++b)
		{
			if (s) X.square();

			if ((k & (uint32_t(1) << (31 - b))) != 0)
			{
				s = true;
				X.mul(a);
			}
		}

		// X = X^(2^(n - 1))
		Timer::Time startBenchTime = Timer::currentTime();
		const uint32_t benchCount = 1000;
		for (uint32_t i = 0; i < n - 1; ++i)
		{
			X.square();
			if (i == benchCount - 1)
			{
				X.getError();
				const double elapsedTime = Timer::diffTime(Timer::currentTime(), startBenchTime);
				const double mulTime = elapsedTime / benchCount, estimatedTime = mulTime * n;
				std::cout << ": estimated time is " << Timer::formatTime(estimatedTime) << ", " << std::setprecision(3) << mulTime * 1e3 << " ms/mul." << std::flush;
				if (bench)
				{
					std::cout << std::endl;
					return 0;
				}
			}
		}

		uint64_t res64;
		const bool isPrime = X.isMinusOne(res64);
		const int err = X.getError();

		const double elapsedTime = Timer::diffTime(Timer::currentTime(), startTime);

		const std::string res = (isPrime) ? "" : std::string(", RES64 = ") + resString(res64);

		std::cout << "\r" << k << " * 2^" << n << " + 1 is " << (isPrime ? "prime" : "composite") << ", a = " << a << ", err = " << err
			<< ", " << X.getDigits() << " digits (size = "	<< X.getSize() << "), time = " << Timer::formatTime(elapsedTime) << res << std::endl;

		return res64;
	}

public:
	Application() {}
	virtual ~Application() {}

public:
	static void run()
	{
		std::cout << "proth20 0.0.1" << std::endl;
		std::cout << "Copyright (c) 2020, Yves Gallot" << std::endl;
		std::cout << "proth20 is free source code, under the MIT license." << std::endl << std::endl;

		struct Number
		{
			uint32_t k, n;
			std::string res64;
			Number(const uint32_t k, const uint32_t n, const std::string & res64 = "1") : k(k), n(n), res64(res64) {}
		};

		std::vector<Number>	primeList;
		primeList.push_back(Number(1035, 301));
		primeList.push_back(Number(955, 636));
		primeList.push_back(Number(969, 1307));
		primeList.push_back(Number(1139, 2641));
		primeList.push_back(Number(1035, 5336));
		primeList.push_back(Number(965, 10705));
		primeList.push_back(Number(1027, 21468));
		primeList.push_back(Number(1109, 42921));
		primeList.push_back(Number(1085, 85959));
		primeList.push_back(Number(1015, 171214));
		primeList.push_back(Number(1197, 343384));
		primeList.push_back(Number(1089, 685641));
		primeList.push_back(Number(1005, 1375758));
		primeList.push_back(Number(1089, 2746155));	// 1.14 ms
		primeList.push_back(Number(45, 5308037));	// 2.34 ms

		std::vector<Number>	compositeList;
		compositeList.push_back(Number(9999, 299, "B073C97A2450454F"));
		compositeList.push_back(Number(21, 636, "4FD4F9FE4C6E7C1B"));
		compositeList.push_back(Number(4769, 1307, "8B5F4C7215F37871"));
		compositeList.push_back(Number(9671, 2631, "0715EDFC4814B64A"));
		compositeList.push_back(Number(19, 5336, "614B05AC60E508A0"));
		compositeList.push_back(Number(963, 10705, "232BF76A98040BA3"));
		compositeList.push_back(Number(6189,21469, "88D2582BDDE8E7CA"));
		compositeList.push_back(Number(2389, 42922, "E427B88330D2EE8C"));
		compositeList.push_back(Number(1295, 85959, "53D33CD949CC31DB"));
		compositeList.push_back(Number(9273, 171214, "AEC1A38C0C4B1D98"));
		compositeList.push_back(Number(8651, 343387, "B832D18693CCB6BC"));

		ocl::Engine engine;
		engine.displayDevices();

		ocl::Device device0(engine, 0);

		// test Intel GPU
		// ocl::Device device1(engine, 1);
		// check(1199, 2755, device1);

		// profile
		// gpmp X(45, 5308037, device0);
		// X.square();
		// device0.displayProfiles();

		// bench
		for (const auto & p : primeList) check(p.k, p.n, device0, true);

		// check residues
		for (const auto & c : compositeList)
		{
			const std::string res64 = resString(check(c.k, c.n, device0));
			if (res64 != c.res64) std::cout << "Error: " << res64 << " != " << c.res64 << std::endl;
		}

		// test primes
		for (const auto & p : primeList) check(p.k, p.n, device0);

		// too large
		check(3, 5505020, device0);
	}
};

int main()
{
	try
	{
		Application::run();
	}
	catch (const std::runtime_error & e)
	{
		std::cerr << " error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
