#!/usr/bin/env python

# import os
# import shutil
# import sys
from setuptools import setup, find_packages

version = "0.1.0"
readme = open("README.md").read()
requirements = ["torch==1.0.0"]

setup(
    name="sincnetconv",
    version="0.1.0",
    author="Angus Turner",
    author_email="angus@wearepopgun.com",
    description="a novel Convolutional Neural Network (CNN) that encourages the first layer to "
                "discover more meaningful filters by exploiting parametrized sinc functions",
    license='MIT',
    packages=['sincnetconv']
)