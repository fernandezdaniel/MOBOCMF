from setuptools import setup

setup(name='DGPMF',
            version='0.1',
            author="Daniel Fernandez-Sanchez, Daniel Hernandez-Lobato",
            author_email="daniel.fernandezs@uam.es, daniel.hernandez@uam.es",
            license="Apache License 2.0",
            packages=["dgpmf",
                      "dgpmf.kernels",
                      "dgpmf.likelihoods",
                      "dgpmf.models",
                      "dgpmf.acquisition_functions",
                      "utils"]
      )
