# proth20
An OpenCL implementation of Proth's Theorem

## About

**proth20** is an [OpenCL™](https://www.khronos.org/opencl/) application.  
It performs a fast primality test for numbers of the form *k*·2<sup>*n*</sup> + 1 with [Proth's Theorem](https://en.wikipedia.org/wiki/Proth%27s_theorem).  
[Proth.exe](https://primes.utm.edu/programs/gallot/) was created by Yves Gallot in 1998. It is a single-threaded CPU program that found many prime number records about 20 years ago.  
proth20 is a new highly optimised GPU application, created in 2020.

## Build

A preliminary version of proth20 is available for tests. Optimization of the algorithms is completed; all of them are implemented on GPU.  
Any number of the form *k*·2<sup>*n*</sup> + 1 such that 3 &le; *k* < 100,000,000 and 1 &le; *n* < 100,000,000 can be tested.  
This version was compiled with gcc 8.1 and tested on Windows. But it can be built on Linux or OS X.  
An OpenCL SDK is not required. OpenCL header files are included in the project and the application is linked with the dynamic OpenCL library of the OS.

## TODO

- test BOINC interface
- compute the multiplicative order of *a* modulo *k*·2<sup>*n*</sup> + 1
- check the divisibility of Fermat and generalized Fermat numbers
