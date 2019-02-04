# python3 setup.py build_ext --inplace

from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension

ext_modules = [
    Extension("mixtures",
              sources=["mixtures.pyx"],
              libraries=["m"]  # Unix-like specific
              )
]


setup(
    ext_modules = cythonize(
        "mixtures.pyx",
        annotate=True))
