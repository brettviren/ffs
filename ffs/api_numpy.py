#!/usr/bin/env python
'''
Thin wrapper around numpy
'''

import numpy

name = "numpy"

def random(shape, dtype="float32"):
    '''
    Return a random array
    '''
    r = numpy.random.rand(*shape)
    if dtype.startswith('i'):
        r *= 1e9
    return numpy.array(r, dtype=dtype)

def gpu(arr):
    '''
    Copy numpy array to numpy array as proxy for GPU transfer
    '''
    return numpy.array(arr)

def cpu(arr):
    '''
    Copy numpy array to numpy array as proxy for CPU transfer
    '''
    return numpy.array(arr)

def fft(arr):
    '''
    Return N-D FFT of arr
    '''
    for axis in range(arr.ndim):
        arr = numpy.fft.fft(arr, axis=axis)
    return arr

def ifft(arr):
    '''
    Return N-D inverse FFT of arr
    '''
    for axis in range(arr.ndim):
        arr = numpy.fft.ifft(arr, axis=axis)
    return arr
