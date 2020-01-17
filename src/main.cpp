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

class Application
{
public:
	Application() {}
	virtual ~Application() {}

public:
	static void run(const std::vector<std::string> & args)
	{
		std::cout << "proth20 0.1.0" << std::endl;
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

		ocl::Engine engine;
		engine.displayDevices();
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
				if ((k < 3) || (n == 0)) throw std::runtime_error("invalid expression");

				if (k >= 250000) throw std::runtime_error("max k = 249999");
				if (n > 5500000) throw std::runtime_error("max n = 5500000");

				bPrime = true;
			}
			if (arg.substr(0, 2) == "-d")
			{
				const std::string dev = ((arg == "-d") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				d = std::atoi(dev.c_str());
				if (d >= engine.getDeviceCount()) throw std::runtime_error("invalid device number");
			}
		}

		if (bBench)
		{
			ocl::Device device(engine, d);
			proth::bench(device);
		}

		if (bPrime)
		{
			ocl::Device device(engine, d);
			proth::check(k, n, device);
		}

		ocl::Device device0(engine, 0);

		// test Intel GPU
		// ocl::Device device1(engine, 1);
		// proth::check(1199, 2755, device1);

		// profile: ocl_profile must be defined (ocl.h)
		// proth::profile(45, 5308037, device0);

		// bench
		// proth::test_prime(device0, true);
		// proth::test_composite(device0, true);

		// true test
		proth::test_composite(device0);
		proth::test_prime(device0);

		// too large
		// proth::check(3, 5505020, device0);
	}
};

int main(int argc, char * argv[])
{
	try
	{
		std::vector<std::string> args;
		for (int i = 1; i < argc; ++i) args.push_back(argv[i]);
		Application::run(args);
	}
	catch (const std::runtime_error & e)
	{
		std::cerr << "error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
