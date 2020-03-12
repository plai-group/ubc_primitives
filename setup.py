from setuptools import setup

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
      url='https://github.com/tonyjo/ubc_primitives.git',
      maintainer_email='tonyjos@cs.ubc.ca',
      maintainer='Tony Joseph',
      license='MIT',
      packages=[
                'primitives',
                'primitives.smi',
                'primitives.smi.weights',
                'primitives.boc',
                'primitives.bow',
                'primitives.cnn',
                'primitives.cnn.cnn_models',
                'primitives.googlenet',
                'primitives.mobilenet',
               ],
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=install_requires,
      keywords='d3m_primitive',
      entry_points={
          'd3m.primitives': [
              'schema_discovery.profiler.UBC=primitives.smi:SemanticTypeInfer',
              'feature_extraction.bag_of_characters.UBC=primitives.boc:BagOfCharacters',
              'feature_extraction.bag_of_words.UBC=primitives.bow:BagOfWords',
              'feature_extraction.cnn.UBC=primitives.cnn:ConvolutionalNeuralNetwork',
              'feature_extraction.googlenet.UBC=primitives.googlenet:GoogleNetCNN',
              'feature_extraction.mobilenet.UBC=primitives.mobilenet:MobileNetCNN',
          ],
      })
