from setuptools import setup

setup(name='pyunlvrtm',
      version='0.2.1',
      description='Python packages to faciliate the UNL-VRTM model',
      url='https://github.com/xxu2/pyunlvrtm',
      author='Richard Xu',
      author_email='xxu@huksers.unl.edu',
      license='MIT',
      packages=['pyunlvrtm'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
