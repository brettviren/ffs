#!/usr/bin/env python3
'''
Fast Fourier Spin

Stress test GPUs for FFT and other things.
'''
import os
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


@cli.command()
@click.argument("outdir")
@click.pass_context
def campaign(ctx, outdir):
    '''
    Generate into outdir files to run a campaign.
    '''
    multis = (1, 2, 3, 10)
    dtypes = ("float16", "float32", "float64")

    # Do more counts of the smaller array shapes to approximately use
    # similar time for each shape.
    shapes = ("100,100", "1000,1000", "10000,1000")
    counts = (10000, 1000, 100)

    tests = ("copy", "rand", "fft")

    outdir = os.path.realpath(outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    sm_fname = f'{outdir}/shoreman.sh'
    smurl = 'https://raw.githubusercontent.com/brettviren/shoreman/master/shoreman.sh'
    if not os.path.exists(sm_fname):
        from urllib.request import urlretrieve
        urlretrieve(smurl, sm_fname)
        os.chmod(sm_fname, 0o755)

    module=ctx.obj.name

    main_log = f'{module}-campaign.log'
    main_fname = f'{outdir}/{module}-campaign.sh'
    main = open(main_fname, 'w')
    main.write(f"""#!/bin/sh
set -e
echo "logging to {main_log}"
date > {main_log}
hostname >> {main_log}
lscpu >> {main_log}
nvidia-smi --query >> {main_log}

set +x
    """)

    test_cli = "ffs -m {module} {test} --shape {shape} --count {count} --dtype {dtype}"

    from itertools import product as pd
    outer = pd(multis, dtypes, zip(shapes,counts), tests)
    for multi,dtype,(shape,count),test in outer:
        shape_name = shape.replace(",", "x")

        name = f'{module}-{test}-{shape_name}-{dtype}'
        pf = f'{name}.procfile'
        gf = f'{name}-gpu.log'
        cf = f'{name}-cpu.log'
        jf = f'{name}-job.log'
        main.write(f'./shoreman.sh {pf} "" {cf} {gf}  >> {main_log} 2>&1\n')
        with open(f'{outdir}/{pf}', 'w') as procfile:
            cli = test_cli.format(**locals())
            for num in range(multi):
                procfile.write(f'{name}-{num}: {cli}\n')
    main.close()
    os.chmod(main_fname, 0o755)
    print(f'cd {outdir} && ./{module}-campaign.sh')



def main():
    cli(obj=None)

if '__main__' == __name__:
    main()
