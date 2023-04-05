from setuptools import setup, Extension
import pybind11 
# python3 -m pybind11 --includes
functions_module = Extension(  
    name ='AEPacking',  
    sources = ['main.cpp'],  
    include_dirs = [pybind11.get_include()],
    extra_compile_args=["-fopenmp", "-std=c++11", "-O3"], # log4cxx is not currently used
    extra_link_args=["-fopenmp", "-std=c++11", "-O3"],
    language='c++',
)  
  
setup(ext_modules = [functions_module])