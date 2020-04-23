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

## TA-1 primitives

1. **Bag of Characters**
  - Used to extract features describing the distribution of characters in a column/rows.
  - Sample Usage: [Bag of Characters](https://github.com/plai-group/ubc_primitives/blob/master/pipelines/boc_pipeline.py) pipeline.

2. **Bag of Words**
  - Used to extract features describing 27 global statistical features.
  - Sample Usage: [Bag of Words](https://github.com/plai-group/ubc_primitives/blob/master/pipelines/bow_pipeline.py) pipeline.

3. **Canonical Correlation Forests Classifier**
  - A new decision tree ensemble method for classification. CCFs naturally accommodate multiple outputs, provide a similar computational complexity to random forests.
  - Standalone open-sourced implementation of [CCFs](https://github.com/plai-group/ccfs-python).
  - Sample Usage: [CCFs Classifier](https://github.com/plai-group/ubc_primitives/blob/master/pipelines/ccfsClfy_pipeline.py) pipeline.

4. **Canonical Correlation Forests Regressor**
  - A new decision tree ensemble method for regression. CCFs naturally accommodate multiple outputs, provide a similar computational complexity to random forests.
  - Standalone open-sourced implementation of [CCFs](https://github.com/plai-group/ccfs-python).
  - Sample Usage: [CCFs Regressor](https://github.com/plai-group/ubc_primitives/blob/master/pipelines/ccfsReg_pipeline.py) pipeline.

5. **Multilayer Perceptron Classifier**
  - A feed-forward neural network classification primitive using PyTorch. It can be configured with input and output dimensions, number of layers `depth` Hyperparam, and number of units in each layer except the last one `width` Hyperparam.
  - Sample Usage: [MLP Classifier](https://github.com/plai-group/ubc_primitives/blob/master/pipelines/mlpClfy_pipeline.py) pipeline.

6. **Multilayer Perceptron Regressor**
  - A feed-forward neural network regression primitive using PyTorch. It can be configured with input dimensions, number of layers `depth` Hyperparam, and number of units in each layer except the last one `width` Hyperparam.
  - Sample Usage: [MLP Regressor](https://github.com/plai-group/ubc_primitives/blob/master/pipelines/mlpReg_pipeline.py) pipeline.


7. **Convolutional Neural Network**
  - Convolutional Neural Network primitive can be used to extract deep features from images. It can be used as a pre-trained feature extractor, to extract features from
convolutional layers or the fully connected layers by setting `include_top` Hyperparam.
  - It can also be fine-tunned to fit (classification/regression) new data, by setting `feature_extraction` Hyperparam to False and `output_dim` Hyperparam to specify output dimension.
  - Available pre-trained CNN models are: VGG-16, VGG-16 with Batch-Norm , GoogLeNet, ResNet-34, and MobileNet. All available models are pre-trained on ImageNet.
  - Sample Usage: [CNN](https://github.com/plai-group/ubc_primitives/blob/master/pipelines/cnn_pipeline.py) pipeline.

8. **GoogleNet CNN**
  - GoogleNet Convolutional Neural Network primitive. Unlike the CNN primitive, the overhead metadata installation is much faster, as it only consists of single CNN architecture.
  - It can be used as a pre-trained feature extractor, to extract features from
  convolutional layers or the fully connected layers by setting `include_top` Hyperparam.
  - It can also be fine-tunned to fit (classification/regression) new data, by setting `feature_extraction` Hyperparam to False and `output_dim` Hyperparam to specify output dimension.
  - Sample Usage: [GoogleNet CNN](https://github.com/plai-group/ubc_primitives/blob/master/pipelines/googlenet_pipeline.py) pipeline.

9. **MobileNet CNN**
  - MobileNet is a light weight Convolutional Neural Network primitive. Unlike the CNN primitive, the overhead metadata installation is much faster, as it only consists of single CNN architecture.
  - It can be used as a pre-trained feature extractor, to extract features from
  convolutional layers or the fully connected layers by setting `include_top` Hyperparam.
  - It can also be fine-tunned to fit (classification/regression) new data, by setting `feature_extraction` Hyperparam to False and `output_dim` Hyperparam to specify output dimension.
  - Sample Usage: [MobileNet CNN](https://github.com/plai-group/ubc_primitives/blob/master/pipelines/mobilenet_pipeline.py) pipeline.

10. **ResNet CNN**
  - ResNet Convolutional Neural Network primitive. Unlike the CNN primitive, the overhead metadata installation is much faster, as it only consists of single CNN architecture.
  - It can be used as a pre-trained feature extractor, to extract features from
  convolutional layers or the fully connected layers by setting `include_top` Hyperparam.
  - It can also be fine-tunned to fit new data, by setting `feature_extraction` Hyperparam to False and `output_dim` Hyperparam to specify output dimension.
  - Sample Usage: [ResNet CNN](https://github.com/plai-group/ubc_primitives/blob/master/pipelines/resnet_pipeline.py) pipeline.

11. **VGGNet CNN**
  - VGGNet Convolutional Neural Network primitive. Unlike the CNN primitive, the overhead metadata installation is much faster, as it only consists of single CNN architecture.
  - It can be used as a pre-trained feature extractor, to extract features from
  convolutional layers or the fully connected layers by setting `include_top` Hyperparam.
  - It can also be fine-tunned to fit (classification/regression) new data, by setting `feature_extraction` Hyperparam to False and `output_dim` Hyperparam to specify output dimension.
  - Sample Usage: [VGGNet CNN](https://github.com/plai-group/ubc_primitives/blob/master/pipelines/vggnet_pipeline.py) pipeline.

12. **Principal Component Analysis**
  - PCA primitive is used to project data into a lower dimensional space, by finding linearly uncorrelated variables called principal components.
  - Sample Usage: [PCA](https://github.com/plai-group/ubc_primitives/blob/master/pipelines/pca_pipeline.py) pipeline.

13. **K-Means**
  - The K-Means algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares.
  - This algorithm requires the number of clusters, `n_clusters` Hyperparam to be specified.
  - Sample Usage: [K-Means](https://github.com/plai-group/ubc_primitives/blob/master/pipelines/kmeans_pipeline.py) pipeline.

14. **Simple-CNAPS**
  - Simple CNAPS is a simple classcovariance-based distance metric, namely the Mahalanobis distance, adopted into a state-of-the-art few-shot learning approach.
  - Sample Usage: [Simple-CNAPS](https://github.com/plai-group/ubc_primitives/blob/master/pipelines/cnaps_pipeline.py) pipeline.
  - The metadataset dataset accompanying this primitive can be downloaded from [Link](https://dl.dropboxusercontent.com/s/4ehm3rpotv0x0s8/LWLL1_metadataset.zip?dl=1)

15. **Semantic Type Inference**
  - A primitive for detecting the semantic type of inputed column data
  - Sample Usage: [SMI](https://github.com/plai-group/ubc_primitives/blob/master/pipelines/smi_pipeline.py) pipeline.


## Affiliations
| ![alt-text-2](./logo/ubc.png "UBC") | ![alt-text-1](./logo/plai.jpeg "PLAI-LAB") | ![alt-text-2](./logo/darpa.png "DARPA-D3M") |
|:---:|:---:|:---:|
