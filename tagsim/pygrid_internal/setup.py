import os, sys


def is_platform_windows():
    return sys.platform == "win32"


def is_platform_mac():
    return sys.platform == "darwin"


def is_platform_linux():
    return sys.platform.startswith("linux")


try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize
import numpy

sourcefiles = ["./src/c_grid.pyx"]

include_dirs = [".", "./src", numpy.get_include()]
library_dirs = [".", "./src"]
if is_platform_windows():
    extra_compile_args = ['/openmp']
    extra_link_args = []
else:
    extra_compile_args = ["-fopenmp"]
    extra_link_args = ["-fopenmp"]

extensions = [
    Extension(
        "c_grid",
        sourcefiles,
        language="c++",
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(name="c_grid",
    ext_modules=cythonize(extensions,
        compiler_directives={'language_level' : sys.version_info[0]})
        )

