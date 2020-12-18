from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
import sys

def is_platform_windows():
    return sys.platform == "win32"


def is_platform_mac():
    return sys.platform == "darwin"


def is_platform_linux():
    return sys.platform.startswith("linux")


if is_platform_windows():
    extra_compile_args = ['/openmp']
    extra_link_args = []
else:
    extra_compile_args = ["-fopenmp"]
    extra_link_args = ["-fopenmp"]



# --- interp2d compile ---

print(' ')
print('***********************************')
print('******** Building interp2d ********')
print('***********************************')
print(' ')

ext_modules = [Extension("interp2d",
                         ["src_interp/interp2d.pyx"],
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_link_args,
                         include_dirs=[numpy.get_include()])]

setup(
  name = 'interp2d',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)


# --- interp_temp2d compile ---

print(' ')
print('***********************************')
print('******** Building interp_temp2d ********')
print('***********************************')
print(' ')

ext_modules = [Extension("interp_temp2d",
                         ["src_interp/interp_temp2d.pyx"],
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_link_args,
                         include_dirs=[numpy.get_include()])]

setup(
  name = 'interp_temp2d',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)


# --- pygrid3 compile ---

print(' ')
print('**********************************')
print('******** Building pygrid3 ********')
print('**********************************')
print(' ')

ext_modules = [Extension("pygrid_internal.c_grid",
                         ["./pygrid_internal/src/c_grid.pyx"],
                         language="c++",
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_link_args,
                         include_dirs=["./pygrid_internal/src/", numpy.get_include()],
                         library_dirs=["./pygrid_internal/src/"])]

setup(
  name = 'c_grid',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules 
)