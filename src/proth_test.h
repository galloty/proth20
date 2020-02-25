/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "proth.h"

class proth_test
{
private:
	struct number
	{
		uint32_t k, n;
		uint64_t res64;
		number(const uint32_t k, const uint32_t n, const uint64_t res64 = 0) : k(k), n(n), res64(res64) {}
	};

public:
	static bool validate(proth & p, const uint32_t k, const uint32_t n, const uint32_t L, engine & engine)
	{
		const uint32_t a = 3;
		gpmp X(k, n, engine, false, false);

		std::ostringstream sst;
		sst << "Testing " << k << " * 2^" << n << " + 1, size = 2^" << arith::log2(X.getSize()) << " x " << X.getDigitBit() << " bits" << std::endl;
		pio::display(sst.str());

		const size_t cntSq = X.getPlanSquareSeqCount(), cntP2i = X.getPlanPoly2intCount();

		for (size_t j = 0, cnt = std::max(cntSq, cntP2i); j < cnt; ++j)
		{
			X.setPlanSquareSeq(j % cntSq);
			X.setPlanPoly2intFn(j % cntP2i);

			pio::display(X.getPlanString());

			if (!p.apowk(X, a, k)) return false;
			p.checkError(X);

			for (uint32_t i = 1; i < L * L; ++i)
			{
				X.square();
				if ((i & (L - 1)) == 0) X.Gerbicz_step();
				if (p._quit) return false;
			}

			X.square();
			X.Gerbicz_check(L);
			p.checkError(X);
			std::ostringstream ss; ss << " valid" << std::endl;
			pio::display(ss.str());
		}

		return true;
	}

public:
	static void test_prime(proth & p, engine & engine)
	{
		std::vector<number> primeList;
		primeList.push_back(number(1035u, 301u));
		primeList.push_back(number(955u, 636u));
		primeList.push_back(number(969u, 1307u));
		primeList.push_back(number(1139u, 2641u));
		primeList.push_back(number(1035u, 5336u));
		primeList.push_back(number(965u, 10705u));
		primeList.push_back(number(1027u, 21468u));			// size = 2^11
		primeList.push_back(number(1109u, 42921u));			// size = 2^12
		primeList.push_back(number(1085u, 85959u));			// size = 2^13  64-bit 32-bit
		primeList.push_back(number(1015u, 171214u));		// size = 2^14, 0.070  0.070  256_16 sq_64 p2i_16_16
		primeList.push_back(number(1197u, 343384u));		// size = 2^15, 0.100  0.095  256_16 sq_128 p2i_8_32
		primeList.push_back(number(1089u, 685641u));		// size = 2^16, 0.160  0.158  256_8 sq_256 p2i_8_64
		primeList.push_back(number(1005u, 1375758u));		// size = 2^17, 0.275  0.270  256_8 sq_512 p2i_8_64
		primeList.push_back(number(1089u, 2746155u));		// size = 2^18, 0.510  0.500  256_8 sq_1024 p2i_8_64
		primeList.push_back(number(45u, 5308037u));			// size = 2^19, 0.978  0.955   256_8 sq_2048 p2i_8_32
		primeList.push_back(number(6679881u, 6679881u));	// size = 2^20, 1.94   1.87   256_8 256_8 sq_16 p2i_8_32
		primeList.push_back(number(3u, 10829346u));			// size = 2^21, 3.90   3.76   256_16 256_8 sq_32 p2i_8_32
		primeList.push_back(number(10223u, 31172165u));		// size = 2^22, 7.95   7.60   256_16 256_8 sq_64 p2i_8_32

		for (const auto & pr : primeList) if (!p.check(pr.k, pr.n, engine, true)) return;
	}

public:
	static void test_composite(proth & p, engine & engine)
	{
		std::vector<number> compositeList;
		compositeList.push_back(number(536870911u,    298u, 0x35461D17F60DA78Aull));
		compositeList.push_back(number(536870905u,    626u, 0x06543644B033FF0Cull));
		compositeList.push_back(number(536870411u,   1307u, 0x7747B2D2351394EFull));
		compositeList.push_back(number(536850911u,   2631u, 0x3A08775B698EEB34ull));
		compositeList.push_back(number(536870911u,   5336u, 0xDA3B38B4E68F0445ull));
		compositeList.push_back(number(536770911u,  10705u, 0x030EECBE0A5E77A6ull));
		compositeList.push_back(number(526870911u,  21432u, 0xD86853C587F1D537ull));
		compositeList.push_back(number(436870911u,  42921u, 0x098AD2BD01F485BCull));
		compositeList.push_back(number(535970911u,  85942u, 0x2D19C7E7E7553AD6ull));
		compositeList.push_back(number(536860911u, 171213u, 0x99EFB220EE2289A0ull));
		compositeList.push_back(number(536870911u, 343386u, 0x5D6A1D483910E48Full));
		compositeList.push_back(number(536870911u, 685618u, 0x84C7E4E7F1344902ull));

		// check residues
		for (const auto & c : compositeList) if (!p.check(c.k, c.n, engine, true, c.res64)) return;
	}

public:
	static void test_order(proth & p, engine & engine)
	{
		std::vector<number> orderList;
		orderList.push_back(number(385u, 7598u));			// 2^7596 * 5 * 11
		orderList.push_back(number(387u, 23127u));			// 2^23126 * 3^2
		orderList.push_back(number(357u, 35567u));			// 2^35564
		orderList.push_back(number(357u, 54392u));			// 2^54390 * 17
		orderList.push_back(number(315u, 99512u));			// 2^99511 * 3^2 * 5 * 7

		for (const auto & o : orderList) if (!p.check_order(o.k, o.n, 2, engine)) return;
	}

public:
	static void test_gfn(proth & p, engine & engine)
	{
		std::vector<number> gfnDivList;
		gfnDivList.push_back(number(332436749u, 9865u));
		gfnDivList.push_back(number(5u, 23473u));
		gfnDivList.push_back(number(1199u, 44201u));
		gfnDivList.push_back(number(165u, 49095u));
		gfnDivList.push_back(number(189u, 90061u));
		gfnDivList.push_back(number(3u, 213321u));
		gfnDivList.push_back(number(3u, 382449u));

		for (const auto & d : gfnDivList) if (!p.check_gfn(d.k, d.n, engine)) return;
	}

public:
	static void validation(proth & p, engine & engine)
	{
		// if (!validate(536870911, 3000000, 4, engine)) return;
		for (uint32_t n = 15000; n < 100000000; n *= 2) if (!validate(p, 536870911, n, 16, engine)) return;
	}

public:
	static void profile(const uint32_t k, const uint32_t n, engine & engine)
	{
		gpmp X(k, n, engine, false, true, true);
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
