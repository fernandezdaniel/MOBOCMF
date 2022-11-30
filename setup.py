from setuptools import setup

setup(name='MOBOCMF',
     version='0.1',
     author="Daniel Fernandez-Sanchez, Daniel Hernandez-Lobato",
     author_email="daniel.fernandezs@uam.es, daniel.hernandez@uam.es",
     license="Apache License 2.0",
     packages=["mobocmf",
               "mobocmf.layers",
               "mobocmf.mlls",
               "mobocmf.models",
               "mobocmf.test_functions",
               "examples",
               "experiments"]
)
