/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include "ocl.h"
#include "proth.h"

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

public:
	void run(const std::vector<std::string> & args)
	{
		std::cout << "proth20 0.2.0" << std::endl;
		std::cout << "Copyright (c) 2020, Yves Gallot" << std::endl;
		std::cout << "proth20 is free source code, under the MIT license." << std::endl;

		// if -v or -V exit
		for (const std::string & arg : args)
		{
			if ((arg[0] == '-') && ((arg[1] == 'v') || (arg[1] == 'V'))) return;
		}
		std::cout << std::endl;

		if (args.empty())	// print usage, display devices and exit
		{
			std::cout << "Usage: proth20 [options]  options may be specified in any order" << std::endl;
			std::cout << "  -q \"k*2^n+1\"            test expression primality" << std::endl;
			std::cout << "  -d <n> or --device <n>  set device number=<n> (default 0)" << std::endl;
			std::cout << "  -b                      run benchmark" << std::endl;
			std::cout << "  -v or -V                print the startup banner and immediately exit" << std::endl;
			std::cout << std::endl;
		}

		ocl::platform platform;
		platform.displayDevices();
		std::cout << std::endl;

		bool bBench = false, bPrime = false;
		uint32_t k = 0, n = 0;
		size_t d = 0;
		// parse args
		for (size_t i = 0, size = args.size(); i < size; ++i)
		{
			const std::string & arg = args[i];

			if (arg == "-b") bBench = true;
			if (arg.substr(0, 2) == "-q")
			{
				const std::string exp = ((arg == "-q") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				auto k_end = exp.find('*');
				if (k_end != std::string::npos) k = std::atoi(exp.substr(0, k_end).c_str());
				auto n_start = exp.find('^'), n_end = exp.find('+');
				if ((n_start != std::string::npos) && (n_end != std::string::npos)) n = std::atoi(exp.substr(n_start + 1, n_end).c_str());
				if ((k < 3) || (n < 32)) throw std::runtime_error("invalid expression");

				if (k > 99999999) throw std::runtime_error("k > 99999999 is not supported");
				if (n > 99999999) throw std::runtime_error("n > 99999999 is not supported");

				bPrime = true;
			}
			if (arg.substr(0, 2) == "-d")
			{
				const std::string dev = ((arg == "-d") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				d = std::atoi(dev.c_str());
				if (d >= platform.getDeviceCount()) throw std::runtime_error("invalid device number");
			}
		}

		if (bBench)
		{
			engine engine(platform, d);
			proth::getInstance().bench(engine);
		}

		if (bPrime)
		{
			engine engine(platform, d);
			proth::getInstance().check(k, n, engine);
		}

		// gpmp::printRanges(10000);

		// engine engine0(platform, 0);
		// test Intel GPU
		// engine engine1(platform, 1);

		// proth & p = proth::getInstance();

		// function profiling
		// p.profile(45, 5308037, engine0);
		// p.profile(99739, 14019102, engine0);

		// bench
		// p.bench(engine0);
		// p.test_prime(engine0, true);
		// p.test_composite(engine0, true);

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
		std::cerr << std::endl << "error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
