/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include "pio.h"
#include "ocl.h"
#include "proth.h"
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
		std::ostringstream ss;
		ss << "proth20 0.3.0" << std::endl;
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
		ss << "  -q \"k*2^n+1\"            test expression primality" << std::endl;
		ss << "  -d <n> or --device <n>  set device number=<n> (default 0)" << std::endl;
		ss << "  -b                      run benchmark" << std::endl;
		ss << "  -v or -V                print the startup banner and immediately exit" << std::endl;
		ss << "  -boinc                  operate as a BOINC client app" << std::endl;
		ss << std::endl;
		return ss.str();
	}

public:
	void run(const std::vector<std::string> & args)
	{
		bool bBoinc = false;
		for (const std::string & arg : args) if (arg == "-boinc") bBoinc = true;
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

		bool bPrime = false, bBench = false;
		uint32_t k = 0, n = 0;
		size_t d = 0;
		// parse args
		for (size_t i = 0, size = args.size(); i < size; ++i)
		{
			const std::string & arg = args[i];

			if (arg == "-b") bBench = true;
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
			else if (arg.substr(0, 2) == "-d")
			{
				const std::string dev = ((arg == "-d") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				d = std::atoi(dev.c_str());
				if (d >= platform.getDeviceCount()) throw std::runtime_error("invalid device number");
			}
		}

		proth & p = proth::getInstance();
		p.setBoinc(bBoinc);

		if (bBench)
		{
			engine engine(platform, d);
			p.bench(engine);
		}

		if (bPrime)
		{
			engine engine(platform, d);
			p.check(k, n, engine);
		}

		if (bBoinc) boinc_finish(EXIT_SUCCESS);

		// gpmp::printRanges(10000);

		// engine engine0(platform, 0);
		// test Intel GPU
		// engine engine1(platform, 1);

		// function profiling
		// p.profile(13, 5523860, engine0);		// DIV
		// p.profile(10223, 31172165, engine0);	// SOB

		// bench
		// p.bench(engine0);
		// p.test_prime(engine0, true);

		// full test
		// p.test_composite(engine0);
		// p.test_prime(engine0);

		// validation
		// p.validation(engine0);
		// p.validation(engine1);
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
		std::ostringstream ss; ss << std::endl << "error: " << e.what() << std::endl;
		pio::error(ss.str(), true);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
