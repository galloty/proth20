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
		std::cout << "proth20 is free source code, under the MIT license." << std::endl << std::endl;

		// parse args

		// if -v -V exit

		// if no arg usage

		ocl::Engine engine;
		engine.displayDevices();

		ocl::Device device0(engine, 0);

		// test Intel GPU
		// ocl::Device device1(engine, 1);
		// proth::check(1199, 2755, device1);

		// profile: ocl_profile must be defined (ocl.h)
		// proth::profile(45, 5308037, device0);

		// bench
		proth::test_prime(device0, true);
		// proth::test_composite(device0, true);

		// true test
		proth::test_composite(device0);
		proth::test_prime(device0);

		// too large
		proth::check(3, 5505020, device0);
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
		std::cerr << " error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
