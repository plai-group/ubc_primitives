# UBC PLAI-Group TA-1 primitives

![version](https://img.shields.io/badge/version-0.2.0-green.svg)
![docker](https://img.shields.io/badge/Docker-v2020.5.18-blue)

Project repository for TA-1 primitives contributed by the [UBC PLAI Group](https://plai.cs.ubc.ca) to the [DARPA Data Driven Discovery Models (D3M) Program](https://www.darpa.mil/program/data-driven-discovery-of-models).


## Install

1. Get docker image which contains all dependencies and requirements:
```
docker pull registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.5.18
```
Updated [docker images](https://docs.datadrivendiscovery.org/docker.html)


2. Run Docker image (change the local path of the cloned repo):
```
sudo docker run --rm\
    -v ./local/path/ubc_primitives:/ubc_primitives\
    -i -t registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.5.18 /bin/bash
```


3. Inside docker install python package:
```
cd /ubc_primitives
pip3 install -e .
```

## TA-1 primitives

View all primitive [description](./docs/README.md)

---
### Acknowledgement
This material is based upon work supported by the United States Air Force Research Laboratory (AFRL) under the Defense Advanced Research Projects Agency (DARPA) Data Driven Discovery Models (D3M) program (Contract No. FA8750-19-2-0222).
The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of DARPA or the U.S. Government.
| ![alt-text-2](./logo/ubc.png "UBC") | ![alt-text-1](./logo/plai.jpeg "PLAI-LAB") | ![alt-text-2](./logo/darpa.png "DARPA-D3M") |
|:---:|:---:|:---:|
