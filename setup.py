from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()


with open("requirements.txt") as f:
    requirements = [req.strip() for req in f]

setup(
    name='torchspider',
    version='0.0.7',
    description='A Pytorch-based framework',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Gary Lai',
    author_email="glai9665@gmail.com",
    packages=['torchspider'],
    package_dir={'': 'src'},
    install_requires=requirements,
    url="https://github.com/ghlai9665/octopus",
    python_requires=">=3.6",
    classifiers=["License :: OSI Approved :: Apache Software License",
                 "Programming Language :: Python :: 3"],
    license='Apache v2.0'
)
