import subprocess
from setuptools import setup, find_packages

# Run
#subprocess.run(["apt", "update"])
#subprocess.run(["apt", "install", "python3.6-gdbm"])

# Get install requirements
with open('requirements.txt', 'r') as f:
    install_requires = list()
    dependency_links = list()
    for line in f:
        re = line.strip()
        if re:
            install_requires.append(re)

setup(name='ubc_primitives',
      version='0.1.0',
      description='Setup primitive build model paths',
      author='UBC',
      url='https://github.com/plai-group/ubc_primitives.git',
      maintainer_email='tonyjos@cs.ubc.ca',
      maintainer='Tony Joseph',
      license='Apache-2.0',
      packages=find_packages(exclude=['pipelines']),
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=install_requires,
      keywords='d3m_primitive',
      entry_points={
          'd3m.primitives': [
              'schema_discovery.profiler.UBC=primitives_ubc.smi:SemanticTypeInfer',
              'feature_extraction.bag_of_characters.UBC=primitives_ubc.boc:BagOfCharacters',
              'feature_extraction.bag_of_words.UBC=primitives_ubc.bow:BagOfWords',
              'feature_extraction.cnn.UBC=primitives_ubc.cnn:ConvolutionalNeuralNetwork',
              'feature_extraction.googlenet.UBC=primitives_ubc.googlenet:GoogleNetCNN',
              'feature_extraction.mobilenet.UBC=primitives_ubc.mobilenet:MobileNetCNN',
              'feature_extraction.resnet.UBC=primitives_ubc.resnet:ResNetCNN',
              'feature_extraction.vggnet.UBC=primitives_ubc.vgg:VGG16CNN',
              'classification.ccfs.UBC=primitives_ubc.clfyCCFS:CanonicalCorrelationForestsClassifierPrimitive',
              'regression.ccfs.UBC=primitives_ubc.regCCFS:CanonicalCorrelationForestsRegressionPrimitive',
              'classification.mlp.UBC=primitives_ubc.clfyMLP:MultilayerPerceptronClassifierPrimitive',
              'regression.mlp.UBC=primitives_ubc.regMLP:MultilayerPerceptronRegressionPrimitive',
              'clustering.kmeans.UBC=primitives_ubc.kmeans:KMeansClusteringPrimitive',
              'dimensionality_reduction.pca.UBC=primitives_ubc.pca:PrincipalComponentAnalysisPrimitive',
              'classification.simpleCnaps.UBC=primitives_ubc.simpleCNAPS:SimpleCNAPSClassifierPrimitive',
              'regression.LinearRegression.UBC=primitives_ubc.linearRegression:LinearRegressionPrimitive',
              'classification.LogisticRegression.UBC=primitives_ubc.logisticRegression:LogisticRegressionPrimitive',
              'operator.DiagonalMVN.UBC=primitives_ubc.diagonalMVN:DiagonalMVNPrimitive',
          ],
      })

