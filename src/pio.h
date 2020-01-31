/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <string>
#include <iostream>
#include <fstream>

struct pio
{
	// print: console: cout, boinc: cerr
	static void print(const std::string & str) { std::cout << str; }
	// display: console: cout, boinc: -
	static void display(const std::string & str) { std::cout << str << std::flush; }
	// error: normal: cerr, boinc: cerr
	static void error(const std::string & str) { std::cerr << str; }
	// result: normal: 'presult.txt' file, boinc: 'out' file
	static bool result(const std::string & str)
	{
		std::ofstream resFile("presults.txt", std::ios::app);
		if (!resFile.is_open()) return false;
		resFile << str;
		resFile.close();
		return true;
	}
};
