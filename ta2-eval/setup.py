import sys
from setuptools import setup, find_packages

PACKAGE_NAME = 'ta2-eval'
MINIMUM_PYTHON_VERSION = 3, 6

def check_python_version():
    """
    Exit when the Python version is too low.
    """
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))

with open('requirements.txt', 'r') as f:
    install_requires = list()
    dependency_links = list()
    for line in f:
        re = line.strip()
        if re:
            install_requires.append(re)

check_python_version()

setup(name=PACKAGE_NAME,
      version='0.1',
      description='D3M TA2s',
      url='https://github.com/plai-group/ubc_primitives/tree/ta2-eval',
      author='UBC-TA2',
      maintainer_email='tonyjos@cs.ubc.ca',
      maintainer='Tony Joseph',
      license='Copyright (C) UBC PLAI LAB, Inc - All Rights Reserved',
      packages=find_packages(exclude=['datasets', 'test_datasets', 'outputs', '*_runs']),
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=install_requires,
      )

