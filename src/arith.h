/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

struct arith
{
	static constexpr int log2(const size_t n) { return (n > 1) ? 1 + log2(n >> 1) : 0; }

	static constexpr uint32_t invert(const uint32_t n, const uint32_t m)
	{
		int64_t s0 = 1, s1 = 0, d0 = n % m, d1 = m;
		
		while (d1 != 0)
		{
			const int64_t q = d0 / d1;
			d0 -= q * d1;
			const int64_t t1 = d0; d0 = d1; d1 = t1;
			s0 -= q * s1;
			const int64_t t2 = s0; s0 = s1; s1 = t2;
		}
		
		if (d0 != 1) return 0;

		return uint32_t((s0 < 0) ? s0 + m : s0);
	}

	static int jacobi(const uint64_t x, const uint64_t y)
	{
		uint64_t m = x, n = y;

		int k = 1;
		while (m != 0)
		{
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

		return n;	// x and y are not coprime, return their gcd
	}

	static bool proth_prime_quad_nonres(const uint32_t k, const uint32_t n, const uint32_t a_start, uint32_t & ra)
	{
		// Proth's theorem: a such that (a/P) = -1
		// Note that P = k*2^n + 1 and a is odd => (a/P) * (P/a) = 1 if P = 1 (mod 4)
		// Then (P/a) = (P mod a / a)

		uint32_t a = a_start;
		for (; a < (1u << 31); a += 2)
		{
			// unnecessary but this is a condition of LLR.exe and we must check ths same 'a'
			bool isPrime = true;
			for (uint32_t d = 3; d < a; d += 2) if (a % d == 0) isPrime = false;
			if (!isPrime) continue;

			uint32_t pmoda = k % a;
			if (pmoda == 0) continue;
			for (uint32_t i = 0; i < n; ++i) { pmoda += pmoda; if (pmoda >= a) pmoda -= a; }
			pmoda += 1; if (pmoda >= a) pmoda -= a;

			if (pmoda == 0) { ra = a; return false; }
			if (pmoda == 1) continue;

			const int jac = arith::jacobi(pmoda, a);
			if (jac > 1) { ra = jac; return false; }

			if (jac == -1) break;
		}

		if (a >= (1u << 31)) { ra = 0; return false; }
		ra = a;
		return true;
	}

	static void factor(const uint32_t k, std::vector<std::pair<uint32_t, uint32_t>> & fac) 
	{
		uint32_t m = k;

		for (uint32_t p = 3; p < 65536; p += 2)
		{
			if (m % p == 0)
			{
				uint32_t e = 0;
				do
				{
					m /= p;
					++e;	
				}
				while (m % p == 0);

				fac.push_back(std::make_pair(p, e));
				if (m == 1) return;
			}
		}

		if (m != 1) fac.push_back(std::make_pair(m, 1));
	}
};
