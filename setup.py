import setuptools
ver_globals = {}
with open("ffs/version.py") as fp:
    exec(fp.read(), ver_globals)
version = ver_globals["version"]
setuptools.setup(
    name="ffs",
    version=version,
    author="Brett Viren",
    author_email="brett.viren@gmail.com",
    description="Fast Fourier Spin",
    url="https://brettviren.github.io/ffs",
    packages=setuptools.find_packages(),
    python_requires='>=3.5',
    install_requires=[
        "click",         # CLI
        "numpy",
    ],
    extras_require={
        "cupy": [ "cupy" ],
        "torch": ["torch"],
    },
    entry_points = dict(
        console_scripts = [
            'ffs = ffs.__main__:main',
        ]
    ),
)
