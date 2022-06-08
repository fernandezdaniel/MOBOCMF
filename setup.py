from setuptools import setup

setup(name='DGPMF',
     version='0.1',
     author="Daniel Fernandez-Sanchez, Daniel Hernandez-Lobato",
     author_email="daniel.fernandezs@uam.es, daniel.hernandez@uam.es",
     license="Apache License 2.0",
     packages=["DGPMF",
               "mfdgp",
               "mfdgp.acquisition_functions",
               "mfdgp.kernels",
               "mfdgp.likelihoods",
               "mfdgp.mlls",
               "mfdgp.models",
               "mfdgp.utils",
               "experiments"]
)
