#!/bin/bash
nvcc -arch=sm_50 -std=c++11 main.cpp mini-gmp.c primetest.cu -o PrimesCUDA -lcuda
