/*
Copyright 2020, Yves Gallot

proth20 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <string>
#include <iostream>

struct pio
{
	// print: console: cout, boinc: cerr
	static void print(const std::string & str) { std::cout << str; }
	// display: console: cout, boinc: -
	static void display(const std::string & str) { std::cout << str; }
// error: normal: cerr, boinc: cerr
// res: normal: file presult.txt, boinc: "out" file


};
