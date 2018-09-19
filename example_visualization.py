# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load MNIST dataset, and generate its n-digit version."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from PIL import Image


def main():
  output_dir = 'data'
  num_digits = 2
  domain_gap = 'number'
  mnist_split = 'train'
  num_visualize = 10

  # Construct dataset directory.
  dataset_dir = os.path.join(output_dir,
                             'dataset_mnist_%d_%s' % (num_digits,
                                                      domain_gap))

  path = os.path.join(dataset_dir, '%s.npz' % mnist_split)
  with open(path, 'r') as f:
    data = np.load(path)
    labels = data['labels']
    images = data['images']

  visualize_dir = os.path.join(dataset_dir, 'visualization')
  if not os.path.exists(visualize_dir):
    os.makedirs(visualize_dir)

  visualize_indices = np.random.choice(range(len(labels)),
                                       num_visualize, replace=False)
  
  for i in visualize_indices:
    im = Image.fromarray(images[i])
    im.save(os.path.join(visualize_dir,
                         'sample_image_%d_label_%d.jpg' % (i, labels[i])))

if __name__ == '__main__':
  main()

