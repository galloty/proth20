/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>

#include "boinc.h"

class pio
{
private:
	struct deleter { void operator()(const pio * const p) { delete p; } };

public:
	pio() {}
	virtual ~pio() {}

	static pio & getInstance()
	{
		static std::unique_ptr<pio, deleter> pInstance(new pio());
		return *pInstance;
	}

public:
	void setBoinc(const bool isBoinc) { _isBoinc = isBoinc; }

private:
	bool _isBoinc = false;

private:
	// print: console: cout, boinc: stderr
	void _print(const std::string & str) const
	{
		if (_isBoinc) { std::fprintf(stderr, str.c_str()); }
		else { std::cout << str; }
	}

private:
	// display: console: cout, boinc: -
	void _display(const std::string & str) const
	{
		if (!_isBoinc) { std::cout << str << std::flush; }
	}

private:
	// error: normal: cerr, boinc: stderr
	void _error(const std::string & str, const bool fatal) const
	{
		if (_isBoinc) { std::fprintf(stderr, str.c_str()); if (fatal) boinc_finish(EXIT_FAILURE); }
		else { std::cerr << str; }
	}

private:
	// result: normal: 'presult.txt' file, boinc: 'out' file
	bool _result(const std::string & str) const
	{
		if (_isBoinc)
		{
			char out_path[512];
			boinc_resolve_filename("out", out_path, sizeof(out_path));
			FILE * const out_file = boinc_fopen(out_path, "w");
			if (out_file == nullptr) throw std::runtime_error("Cannot write results to out file");
			fprintf(out_file, str.c_str());
			fclose(out_file);
			return true;
		}
		std::ofstream resFile("presults.txt", std::ios::app);
		if (!resFile.is_open()) return false;
		resFile << str;
		resFile.close();
		return true;
	}

public:
	static void print(const std::string & str) { getInstance()._print(str); }
	static void display(const std::string & str) { getInstance()._display(str); }
	static void error(const std::string & str, const bool fatal = false) { getInstance()._error(str, fatal); }
	static bool result(const std::string & str) { return getInstance()._result(str); }
};
