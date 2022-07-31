#! /usr/bin/env python
"""A flexible alternative to scikit-learn Pipelines"""

import codecs
import os

from setuptools import find_packages, setup


def parse_requirements(filename):
    # Copy dependencies from requirements file
    with open(filename, encoding="utf-8") as f:
        requirements = [line.strip() for line in f.read().splitlines()]
        requirements = [
            line.split("#")[0].strip()
            for line in requirements
            if not line.startswith("#")
        ]

    return requirements


# get __version__ from _version.py
ver_file = os.path.join("skdag", "_version.py")
with open(ver_file) as f:
    exec(f.read())

DISTNAME = "skdag"
DESCRIPTION = "A flexible alternative to scikit-learn Pipelines"

with codecs.open("README.rst", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

MAINTAINER = "big-o"
MAINTAINER_EMAIL = "big-o@users.noreply.github.com"
URL = "https://github.com/big-o/skdag"
LICENSE = "new BSD"
DOWNLOAD_URL = "https://github.com/scikit-learn-contrib/project-template"
VERSION = __version__
INSTALL_REQUIRES = parse_requirements("requirements.txt")
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
]
EXTRAS_REQUIRE = {
    tgt: parse_requirements(f"requirements_{tgt}.txt")
    for tgt in ["test", "doc"]
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
