# UBC PLAI-lab TA-1 primitives

![version](https://img.shields.io/badge/version-0.1.0-green.svg)

Project repository for TA-1 primitives for [D3M](https://www.darpa.mil/program/data-driven-discovery-of-models) program.


## Install

1. Get docker image which contains all dependencies and requirements:
```
docker pull registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9
```
Updated [docker images](https://docs.datadrivendiscovery.org/docker.html)


2. Run Docker image (change the local path of the cloned repo):
```
sudo docker run --rm\
    -v ./local/path/ubc_primitives:/ubc_primitives\
    -i -t registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9 /bin/bash
```


3. Inside docker install python package:
```
cd /ubc_primitives
pip3 install -e .
```

## Available TA-1 primitives

1. **Bag-of-Characters**

2. **Bag-of-Words**

3. **Canonical Correlation Forests Classifier**

4. **Canonical Correlation Forests Regressor**

5. **Multilayer Perceptron Classifier**

6. **Multilayer Perceptron Regressor**

7. **Convolutional Neural Network**

8. **GoogleNet CNN**

9. **MobileNet CNN**

10. **ResNet CNN**

11. **VGGNet CNN**

12. **Principal Component Analysis**

13. **K-Means**

14. **Semantic Type Inference**



## Affiliations
| ![alt-text-1](./logo/plai.jpeg "PLAI-LAB")  | ![alt-text-2](./logo/darpa.png "DARPA-D3M") |
|:---:|:---:|
