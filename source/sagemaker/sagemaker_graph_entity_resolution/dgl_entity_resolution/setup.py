from setuptools import setup, find_packages

setup(
   name='dgl_entity_resolution',
   version='1.0',
   description='entity resolution on sagemaker using dgl',
   packages=find_packages(exclude=('test',))
)