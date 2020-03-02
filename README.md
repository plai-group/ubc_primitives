# UBC PLAI-lab TA-1 primitives

[version](https://img.shields.io/badge/version-0.0.1-green.svg)

Project repository for TA-1 primitives for [D3M](https://www.darpa.mil/program/data-driven-discovery-of-models)


## Running the code in docker

1. Get docker image with contains all dependencies:
```
docker pull registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9
```
Updated [docker images](https://docs.datadrivendiscovery.org/docker.html)

2. Run Docker image (change the local path of the cloned repo):
```
sudo docker run --rm\
    -v ./local/path/ubc_primitives:/ubc_primitives\
    -i -t registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9
```

3. Inside docker install python package:
```
cd /ubc_primitives

pip3 install -e .
```
