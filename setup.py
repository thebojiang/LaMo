# Copyright (c) 2026 Applied Intuition, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

with open("requirements.txt", "r", encoding="utf-8") as file:
    requirements = [
        line.strip()
        for line in file.read().splitlines()
        if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("-")
    ]

setup(
    name="lamo",
    version="0.0.1",
    description="Open-source training, evaluation, and inference code for LaMo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LaMo Authors",
    author_email="",
    url="",
    python_requires=">=3.10.0",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={"dev": ["pytest==8.3.2", "ruff==0.1.5"]},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: Unix",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
