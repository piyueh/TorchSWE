#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Install TorchSWE.
"""
import re
import pathlib
import setuptools

rootdir = pathlib.Path(__file__).expanduser().resolve().parent

# basic information
meta = dict(
    name="TorchSWE",
    author="Pi-Yueh Chuang",
    author_email="pychuang@gwu.edu",
    url="https://github.com/piyueh/TorchSWE",
    keywords=["shallow-water equations", "SWE"],
    license="BSD 3-Clause License",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering"
    ],
    license_files=["LICENSE"],
    packages=setuptools.find_packages(),
    entry_points={"console_scripts": ["TorchSWE = torchswe.__main__:main"]}
)

# version and short sescription (read from __init__.py)
with open(rootdir.joinpath("torchswe", "__init__.py"), 'r') as fileobj:
    content = fileobj.read()
    # version
    meta["version"] = re.search(
        r"__version__\s*?=\s*?(?P<version>\S+?)$", content, re.MULTILINE
    ).group("version").strip("\"\'")
    # one line description
    meta["description"] = re.search(
        r"^\"\"\"(?P<desc>\S.*?)$", content, re.MULTILINE
    ).group("desc")

# long  description (read from README.md)
with open(rootdir.joinpath("README.md"), 'r') as fileobj:
    meta["long_description"] = fileobj.read()
    meta["long_description_content_type"] = "text/markdown"

# dependencies
with open(rootdir.joinpath("requirements.txt"), "r") as fileobj:
    deps = fileobj.readlines()
    meta["python_requires"] = ">=3.8"
    meta["install_requires"] = [line.strip() for line in deps]


if __name__ == "__main__":
    setuptools.setup(**meta)
