from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name='octopus',
    version='0.1',
    description='A Pytorch-based framework',
    long_description=long_description,
    author='Gary Lai',
    packages=find_packages(),
    install_requires=["torch<1.10",
                      "fastai==2.4.1", "fastcore==1.3.20", "dill", "fastprogress"]
)
