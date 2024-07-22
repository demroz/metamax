from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='hankel_cpp',
      ext_modules=[cpp_extension.CppExtension('hankel_cpp', ['hankel.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
