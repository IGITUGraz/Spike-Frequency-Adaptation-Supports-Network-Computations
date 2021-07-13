"""
This file installs the LSNN package.
"""
import re

from setuptools import setup, find_packages

__author__ = "Guillaume Bellec, Darjan Salaj, Anand Subramoney"
__version__ = "1.0.6"


def get_requirements(filename):
    """
    Helper function to read the list of requirements from a file
    """
    dependency_links = []
    with open(filename) as requirements_file:
        requirements = requirements_file.read().strip('\n').splitlines()
    for i, req in enumerate(requirements):

        if ':' in req:
            match_obj = re.match(r"git\+(?:https|ssh|http):.*#egg=(.*)-(.*)", req)
            assert match_obj, "Cannot make sense of url {}".format(req)
            requirements[i] = "{req}=={ver}".format(req=match_obj.group(1), ver=match_obj.group(2))
            dependency_links.append(req)
    return requirements, dependency_links

requirements, dependency_links = get_requirements('requirements.txt')

setup(
    name="LSNN",
    version=__version__,
    packages=find_packages('.'),
    author=__author__,
    author_email="bellec@igi.tugraz.at",
    description="Recurrent Spiking Neural Network (LSNN)",
    provides=['lsnn'],
    install_requires=requirements,
    dependency_links=dependency_links,
)
