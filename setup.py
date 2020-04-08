#!/usr/bin/env python
# Official implementation code for "Lung Nodule Detection and Classification from Thorax CT-Scan Using RetinaNet with Transfer Learning" and "Lung Nodule Texture Detection and Classification Using 3D CNN."
# Adapted from of [medicaldetectiontoolkit](https://github.com/pfjaeger/medicaldetectiontoolkit) and [kinetics_i3d_pytorch](https://github.com/hassony2/kinetics_i3d_pytorch)
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
# ==============================================================================

from distutils.core import setup
from setuptools import find_packages

req_file = "requirements.txt"

def parse_requirements(filename):
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

install_reqs = parse_requirements(req_file)

setup(name='model',
      version='latest',
      packages=find_packages(exclude=['test', 'test.*']),
      install_requires=install_reqs,
      dependency_links=[],
      )