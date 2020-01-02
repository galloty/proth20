# proth20
An OpenCL implementation of Proth's Theorem

## About

**proth20** is an [OpenCL™](https://www.khronos.org/opencl/) application.  
It performs a fast primality test for numbers of the form *k*·2<sup>n</sup> + 1 with [Proth's Theorem](https://en.wikipedia.org/wiki/Proth%27s_theorem).  
[Proth.exe](https://primes.utm.edu/programs/gallot/) was created by Yves Gallot in 1998. It is a single-threaded CPU program that found many prime number records 20 years ago.  
proth20 is a new highly optimised GPU application, created in 2020.

## Build

A **preliminary** version of proth20 is available for tests. It will evolve rapidly. Parallelization of the algorithms is completed; they are implemented on GPU.  
It was built with gcc 8.1 on Windows but it can be built on Linux or OS&nbspX.

## TODO

- reduce the number of kernels
- optimize kernels
- add command line arguments
- save & restore context
- add BOINC interface
- extend the limit > 1600000-digit numbers
