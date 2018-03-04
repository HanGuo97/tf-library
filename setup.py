import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="TFLibrary",
    version="0.1",
    packages=find_packages(),
    description="Tensorflow helper library",
    long_description=read("README.md"),
    install_requires=[
        "numpy", "tensorflow>=1.5.0",
    ],
)
