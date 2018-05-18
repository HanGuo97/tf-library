import os
import platform
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def tensorflow_dependency():
    if platform.system() == "Linux":
        return "tensorflow-gpu>=1.8.0"
    else:
        return "tensorflow>=1.8.0"


setup(
    name="TFLibrary",
    version="0.1",
    packages=find_packages(),
    description="Tensorflow helper library",
    long_description=read("README.md"),
    install_requires=[
        "numpy", tensorflow_dependency(),
    ],
)
