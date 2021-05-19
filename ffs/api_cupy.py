#!/usr/bin/env python
'''
Thin wrapper around cupy
'''

import cupy

def random(shape, dtype='float32'):
    '''
    Return a random array
    '''
    r = cupy.random.rand(*shape)
    if dtype.startswith('i'):
        r *= 1e9
    return cupy.array(r, dtype=dtype)

def gpu(arr):
    '''
    Return a version of arr on the GPU
    '''
    return cupy.array(arr)

def cpu(arr):
    '''
    Return a version of arr on the CPU as Numpy.
    '''
    return arr.get()

def fft(arr):
    '''
    Return FFT of arr, must be cupy array
    '''
    for axis in range(arr.ndim):
        arr = cupy.fft.fft(arr, axis=axis)
    return arr

def ifft(arr):
    '''
    Return inverse FFT of arr, must be cupy array
    '''
    for axis in range(arr.ndim):
        arr = cupy.fft.ifft(arr, axis=axis)
    return arr
