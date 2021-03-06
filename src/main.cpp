/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include "pio.h"
#include "ocl.h"
#include "proth.h"
#include "proth_test.h"
#include "boinc.h"

#include <cstdlib>
#include <stdexcept>
#include <vector>

#if defined (_WIN32)
#include <Windows.h>
#else
#include <signal.h>
#endif

class application
{
private:
	struct deleter { void operator()(const application * const p) { delete p; } };

private:
	static void quit(int)
	{
		proth::getInstance().quit();
	}

private:
#if defined (_WIN32)
	static BOOL WINAPI HandlerRoutine(DWORD)
	{
		quit(1);
		return TRUE;
	}
#endif

public:
	application()
	{
#if defined (_WIN32)	
		SetConsoleCtrlHandler(HandlerRoutine, TRUE);
#else
		signal(SIGTERM, quit);
		signal(SIGINT, quit);
#endif
	}

	virtual ~application() {}

	static application & getInstance()
	{
		static std::unique_ptr<application, deleter> pInstance(new application());
		return *pInstance;
	}

private:
	static std::string header(const bool nl = false)
	{
		const char * const sysver =
#if defined(_WIN64)
			"win64";
#elif defined(_WIN32)
			"win32";
#elif defined(__linux__)
#ifdef __x86_64
			"linux64";
#else
			"linux32";
#endif
#elif defined(__APPLE__)
			"macOS";
#else
			"unknown";
#endif

		std::ostringstream ssc;
#if defined(__GNUC__)
		ssc << " gcc-" << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#elif defined(__clang__)
		ssc << " clang-" << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#endif

		std::ostringstream ss;
		ss << "proth20 0.9.1 " << sysver << ssc.str() << std::endl;
		ss << "Copyright (c) 2020, Yves Gallot" << std::endl;
		ss << "proth20 is free source code, under the MIT license." << std::endl;
		if (nl) ss << std::endl;
		return ss.str();
	}

private:
	static std::string usage()
	{
		std::ostringstream ss;
		ss << "Usage: proth20 [options]  options may be specified in any order" << std::endl;
		ss << "  -q \"k*2^n+1\"            test expression (default primality)" << std::endl;
		ss << "  -o <a>                  compute the multiplicative order of a modulo k*2^n+1" << std::endl;
		ss << "  -f                      Fermat and Generalized Fermat factor test" << std::endl;
		ss << "  -d <n> or --device <n>  set device number=<n> (default 0)" << std::endl;
		ss << "  -v or -V                print the startup banner and immediately exit" << std::endl;
#ifdef BOINC
		ss << "  -boinc                  operate as a BOINC client app" << std::endl;
#endif
		ss << std::endl;
		return ss.str();
	}

public:
	void run(const std::vector<std::string> & args)
	{
		bool bBoinc = false;
#ifdef BOINC
		for (const std::string & arg : args) if (arg == "-boinc") bBoinc = true;
#endif
		pio::getInstance().setBoinc(bBoinc);

		if (bBoinc)
		{
			const int retval = boinc_init();
			if (retval != 0)
			{
				std::ostringstream ss; ss << "boinc_init returned " << retval;
				throw std::runtime_error(ss.str());
			}
		}

		// if -v or -V then print header to stderr and exit
		for (const std::string & arg : args)
		{
			if ((arg[0] == '-') && ((arg[1] == 'v') || (arg[1] == 'V')))
			{
				pio::error(header());
				if (bBoinc) boinc_finish(EXIT_SUCCESS);
				return;
			}
		}

		pio::print(header(true));

		if (args.empty()) pio::print(usage());	// print usage, display devices and exit

		ocl::platform platform;
		platform.displayDevices();

		bool bPrime = false, bOrder = false, bGFN = false;
		uint32_t k = 0, n = 0, a = 0;
		size_t d = 0;
		// parse args
		for (size_t i = 0, size = args.size(); i < size; ++i)
		{
			const std::string & arg = args[i];

			if (arg == "-f") bGFN = true;
			else if (arg.substr(0, 2) == "-q")
			{
				const std::string exp = ((arg == "-q") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				auto k_end = exp.find('*');
				if (k_end != std::string::npos) k = std::atoi(exp.substr(0, k_end).c_str());
				auto n_start = exp.find('^'), n_end = exp.find('+');
				if ((n_start != std::string::npos) && (n_end != std::string::npos)) n = std::atoi(exp.substr(n_start + 1, n_end).c_str());
				if (k > 0) while (k % 2 == 0) { k /= 2; ++n; }				
				if ((k < 3) || (n < 32)) throw std::runtime_error("invalid expression");

				if (k > 99999999) throw std::runtime_error("k > 99999999 is not supported");
				if (n > 99999999) throw std::runtime_error("n > 99999999 is not supported");

				bPrime = true;
			}
			else if (arg.substr(0, 2) == "-o")
			{
				const std::string dev = ((arg == "-o") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				a = std::atoi(dev.c_str());
				if (a < 2) throw std::runtime_error("-o: invalid integer a");
				bOrder = true;
			}
			else if (arg.substr(0, 2) == "-d")
			{
				const std::string dev = ((arg == "-d") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				d = std::atoi(dev.c_str());
				if (d >= platform.getDeviceCount()) throw std::runtime_error("invalid device number");
			}
		}

		proth & p = proth::getInstance();
		p.setBoinc(bBoinc);

		if (bPrime)
		{
			engine engine(platform, d);
			if (bOrder) p.check_order(k, n, a, engine);
			else if (bGFN) p.check_gfn(k, n, engine);
			else p.check(k, n, engine);
		}

		if (bBoinc) boinc_finish(EXIT_SUCCESS);

		// gpmp::printRanges(10000);

		// engine engine0(platform, 0);
		// test Intel GPU
		// engine engine1(platform, 1);
		// test CPU
		// engine engine2(platform, 2);

		// function profiling
		// proth_test::profile(13, 5523860, engine0);		// DIV
		// proth_test::profile(10223, 31172165, engine0);	// SOB

		// test
		// proth_test::test_composite(p, engine0);
		// proth_test::test_prime(p, engine0);
		// proth_test::test_order(p, engine0);
		// proth_test::test_gfn(p, engine0);

		// validation
		// proth_test::validation(p, engine0);
		// proth_test::validation(p, engine1);
	}
};

int main(int argc, char * argv[])
{
	try
	{
		application & app = application::getInstance();

		std::vector<std::string> args;
		for (int i = 1; i < argc; ++i) args.push_back(argv[i]);
		app.run(args);
	}
	catch (const std::runtime_error & e)
	{
		std::ostringstream ss; ss << std::endl << "error: " << e.what() << "." << std::endl;
		pio::error(ss.str(), true);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
