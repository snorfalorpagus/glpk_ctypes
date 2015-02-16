#!/usr/bin/env python

from distutils.core import setup

setup(name='glpk_ctypes',
      version='0.1',
      description='Wrapper for GLPK written using ctypes',
      author='Joshua Arnott',
      author_email='josh@snorfalorpagus.net',
      url='',
      packages=['glpk_ctypes'],
      data_files=[('.', 'glpk_4_55_w32.dll')]
     )
