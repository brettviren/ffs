#!/usr/bin/env python3
'''
Fast Fourier Spin

Stress test GPUs for FFT and other things.
'''

import time
import click
import numpy

@click.group()
@click.option("-m","--module", default="numpy",
              type=click.Choice(["cupy","numpy"]),
              help="Set GPU interface module")
@click.pass_context
def cli(ctx, module):
    from importlib import import_module
    ctx.obj = import_module(f'ffs.api_{module}')


@cli.command()
@click.option("--copy/--no-copy", default=True,
              help="Whether to round-trip between CPU and GPU")
@click.option("-s","--shape", default="1000,1000",
              help="Array shape")
@click.option("-c","--count", type=int, default=100,
              help="Number of cycles")
@click.option("-d","--dtype", default="float32",
              help="The array dtype")
@click.pass_context
def fft(ctx, copy, shape, count, dtype):
    '''
    Run FFTs on GPU, transfering back/forth
    '''
    shape = tuple(map(int, shape.split(',')))
    arr = ctx.obj.random(shape, dtype=dtype)
    arr = ctx.obj.cpu(arr)
    nbytes = arr.nbytes
    t0 = time.time()
    if copy:
        for num in range(count):
            a = ctx.obj.gpu(arr)
            b = ctx.obj.fft(a)
            c = ctx.obj.ifft(b)
            d = ctx.obj.cpu(c)
    else:
        a = ctx.obj.gpu(arr)
        for num in range(count):
            b = ctx.obj.fft(a)
            a = ctx.obj.ifft(b)

    t1 = time.time()
    dt = t1-t0
    hz = num/dt
    mbps = 1e-6*num*nbytes/dt
    print(f'{dt:.3f} s, {hz:.1f} Hz, {mbps:.1f} MByte/sec copy:{copy}')


@cli.command()
@click.option("-s","--shape", default="1000,1000",
              help="Array shape")
@click.option("-c","--count", type=int, default=100,
              help="Number of cycles")
@click.option("-d","--dtype", default="float32",
              help="The array dtype")
@click.pass_context
def copy(ctx, shape, count, dtype):
    '''
    Transfering array back/forth from CPU/GPU RAM.
    '''
    shape = tuple(map(int, shape.split(',')))
    arr = ctx.obj.random(shape, dtype=dtype)
    arr = ctx.obj.cpu(arr)
    nbytes = arr.nbytes
    t0 = time.time()
    for num in range(count):
        a = ctx.obj.gpu(arr)
        d = ctx.obj.cpu(a)
        
    t1 = time.time()
    dt = t1-t0
    hz = num/dt
    mbps = 1e-6*num*nbytes/dt
    print(f'{dt:.3f} s, {hz:.1f} Hz, {mbps:.1f} MByte/sec')



@cli.command()
@click.option("-s","--shape", default="1000,1000",
              help="Array shape")
@click.option("-c","--count", type=int, default=100,
              help="Number of cycles")
@click.option("-d","--dtype", default="float32",
              help="The array dtype")
@click.pass_context
def rand(ctx, shape, count, dtype):
    '''
    Generate random numbers on GPU.
    '''
    shape = tuple(map(int, shape.split(',')))
    t0 = time.time()
    for num in range(count):
        arr = ctx.obj.random(shape, dtype=dtype)
        
    t1 = time.time()
    dt = t1-t0
    hz = num/dt
    mbps = 1e-6*num*arr.nbytes/dt
    print(f'{dt:.3f} s, {hz:.1f} Hz, {mbps:.1f} MByte/sec')


@cli.command()
@click.argument("plog")
@click.pass_context
def proflog(ctx, plog):
    '''
    Dump a Python profile binary log.

    >>> python -m cProfile -o fft.bin $(which ffs) -m cupy fft
    >>> ffs proflog fft.bin
    '''
    import pstats
    from pstats import SortKey
    p = pstats.Stats(plog)
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()    

def main():
    cli(obj=None)

if '__main__' == __name__:
    main()
