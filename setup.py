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
               ],
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=install_requires,
      keywords='d3m_primitive',
      entry_points={
          'd3m.primitives': [
              'data_transformation.semantic_type.UBC=primitives.smi:SemanticTypeInfer',
          ],
      })
