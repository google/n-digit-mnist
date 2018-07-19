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

import argparse
import logging
import os
import struct

import numpy as np


class NDigitMnist(object):
  """Construct and write n-digit MNIST into .npz files."""

  def __init__(self, args):
    self.args = args
    np.random.seed(args.seed)

  def _load_mnist(self, mnist_split):
    """Loads the training or test MNIST data and returns a (images, labels)."""
    mnist_map = {
        'train': {
            'images': 'train-images.idx3-ubyte',
            'labels': 'train-labels.idx1-ubyte',
        },
        'test': {
            'images': 't10k-images.idx3-ubyte',
            'labels': 't10k-labels.idx1-ubyte',
        },
    }[mnist_split]

    with open(os.path.join(self.args.mnist_dir, mnist_map['images'])) as f:
      images = self._decode_mnist(f, 'images')
    with open(os.path.join(self.args.mnist_dir, mnist_map['labels'])) as f:
      labels = self._decode_mnist(f, 'labels')

    return images, labels

  def _decode_mnist(self, f, which_data):
    """Decode raw MNIST dataset into numpy array."""

    if which_data == 'images':
      magic, size, rows, cols = struct.unpack('>IIII', f.read(16))
      if magic != 2051:
        raise ValueError('Machic number mismatch.')
      return np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows, cols)

    elif which_data == 'labels':
      magic, size = struct.unpack('>II', f.read(8))
      if magic != 2049:
        raise ValueError('Machic number mismatch.')
      return np.frombuffer(f.read(), dtype=np.uint8).reshape(size)

  def _print_mnist_images(self, images, labels, num_print=10):
    shuffle_indices = range(len(labels))
    np.random.shuffle(shuffle_indices)
    for i in xrange(num_print):
      self._print_mnist_image(images[shuffle_indices[i]],
                              labels[shuffle_indices[i]])

  def _print_mnist_image(self, image, label):
    """Given an image, prints it as a string array."""
    mapper = np.array(['.', '@'])
    for h_ind in xrange(image.shape[0]):
      for w_ind in xrange(image.shape[1]):
        print(mapper[int(image[h_ind, w_ind] > 0.5)], end='')
      print('')
    print(label)

  def _write_num_classes(self, dataset_dir, num_classes, mnist_split):
    with open(
        os.path.join(dataset_dir, 'num_classes_%s' % mnist_split), 'w') as f:
      f.write(str(num_classes))

  def _write_num_samples(self, dataset_dir, num_samples, mnist_split):
    with open(
        os.path.join(dataset_dir, 'num_samples_%s' % mnist_split), 'w') as f:
      f.write(str(num_samples))

  def _choose_numbers_to_include(self, mnist_split):
    """Recipe for contructing n-digit MNIST.

    Strategy for contructing n-digit MNIST: (1) Make a list of numbers
    to be included in the {train,test} split, (2) Given the list of numbers,
    sample digit images from the original MNIST {train,test} splits to compile
    sample_per_number n-digit images per number - see
    self._compile_number_images_and_labels.

    When domain_gap is 'number', we assure that no number is shared across
    train-test. number_ratio_train percent of numbers will be assigned to train
    split, the rest to the test split.

    Args:
      mnist_split: ['train', 'test'] MNIST split where digit images are sampled.

    Returns:
      sample_per_number: {int} Number of samples per class.
      chosen_numbers: {list} List of numbers to include in the current split.

    Raises:
      ValueError: if chosen domain_gap parameter is not supported or the
        minimal number of samples per class condition is not satisfiable.
    """

    all_numbers = xrange(10 ** self.args.num_digits)
    total_num_sample = (self.args.total_num_train
                        if mnist_split == 'train'
                        else self.args.total_num_test)

    if self.args.domain_gap == 'instance':
      chosen_numbers = all_numbers
      sample_per_number = int(total_num_sample / len(chosen_numbers))
      if (sample_per_number < self.args.min_num_instance_per_number and
          mnist_split == 'train'):
        raise ValueError('Cannot guarantee to have minimal %d samples '
                         'for each class under current configuration' %
                         self.args.min_num_instance_per_number)

    elif self.args.domain_gap == 'number':
      if mnist_split == 'train':
        chosen_numbers = np.random.choice(
            all_numbers,
            int(self.args.number_ratio_train * len(all_numbers) / 100),
            replace=False)
        self._train_numbers = chosen_numbers

      elif mnist_split == 'test_seen':
        chosen_numbers = self._train_numbers.copy()
      else:  # test_unseen
        chosen_numbers = list(
            set(all_numbers).difference(set(self._train_numbers)))

      (sample_per_number, chosen_numbers
      ) = self._trim_num_classes_for_min_instance_constraint(
          total_num_sample, chosen_numbers)

    else:
      raise ValueError('domain_gap should be one of [instance, number]')

    logging.info('split %s total number of samples: %d',
                 mnist_split, total_num_sample)
    logging.info('split %s number of samples per number: %d',
                 mnist_split, sample_per_number)

    return sample_per_number, chosen_numbers

  def _trim_num_classes_for_min_instance_constraint(self, total_num_sample,
                                                    chosen_numbers):
    """Subsample the classes to meet the min_instance_per_class constraint."""

    sample_per_number = int(total_num_sample / len(chosen_numbers))

    if sample_per_number < self.args.min_num_instance_per_number:
      sample_per_number = self.args.min_num_instance_per_number
      target_num_chosen_numbers = int(total_num_sample / sample_per_number)
      chosen_numbers = np.random.choice(chosen_numbers,
                                        target_num_chosen_numbers,
                                        replace=False)
    return sample_per_number, chosen_numbers

  def _collect_image_ids(self, sample_per_number,
                         chosen_numbers, imageset_indices_by_label):
    """Given a list of numbers, construct n-digit images and labels.

    Args:
      sample_per_number: {int} Number of samples per class.
      chosen_numbers: {list} List of integer labels that construct the dataset.
      imageset_indices_by_label:

    Returns:
      image_ids: {list(len(chosen_numbers))} List of original MNIST index arrays
          for each n-digit class. Ingredient for compiling the final dataset.
    """
    image_ids = []

    for number in chosen_numbers:
      number_string = self._number_into_string(number)

      # For each position (digit) in the number, sample (sample_per_number)
      # digit image indices to compile the n-digit images.
      digit_image_indices_all_samples = []
      for digit_char in number_string:
        digit_image_indices_all_samples.append(np.expand_dims(
            np.random.choice(
                imageset_indices_by_label[int(digit_char)],
                sample_per_number, replace=False),
            -1))
      digit_image_indices_all_samples = np.concatenate(
          digit_image_indices_all_samples, 1)
      image_ids.append(digit_image_indices_all_samples)

    return image_ids

  def _compile_number_images_and_labels(self, sample_per_number, chosen_numbers,
                                        image_ids, images):
    """Given a list of numbers, construct n-digit images and labels.

    Args:
      sample_per_number: {int} Number of samples per class.
      chosen_numbers: {list} List of integer labels that construct the dataset.
      image_ids: {list(len(chosen_numbers))} List of original MNIST index arrays
          for each class. Ingredient for compiling the final dataset.
      images: {numpy.darray(shape=(?, 28, 28, 1))} Original MNIST images.

    Returns:
      n_digit_images: {numpy.darray(shape=(?, 28, 28*num_digits, 1))}
          batch of n-digit images.
      n_digit_labels: {numpy.darray(shape=(?))} batch of n-digit labels in the
          range [0, 10**num_digits-1].
    """

    n_digit_images = []
    n_digit_labels = []

    for number, digit_image_indices_all_samples in zip(chosen_numbers,
                                                       image_ids):
      # For each sample in sample_per_number indices samples, construct the
      # n-digit image and label.
      for sample_index in xrange(sample_per_number):
        digit_image_indices = digit_image_indices_all_samples[sample_index]

        number_image = []
        for digit_image_index in zip(digit_image_indices):
          digit_image = images[digit_image_index]
          number_image.append(digit_image)

        n_digit_images.append(np.expand_dims(
            np.concatenate(number_image, 1),
            0))
        n_digit_labels.append(number)

    return np.concatenate(n_digit_images, 0), np.array(n_digit_labels)

  def _number_into_string(self, number):
    """Make e.g. 93 into '093' when num_digits=3."""
    number_string = str(number)
    if len(number_string) < self.args.num_digits:
      number_string = ('0' * (self.args.num_digits - len(number_string))
                       + number_string)
    return number_string

  def _save_mnist_to_npz(self, dataset_dir, mnist_split, images, labels):
    path = os.path.join(dataset_dir, '%s.npz' % mnist_split)
    with open(path, 'w') as f:
      np.savez(f, images=images, labels=labels)

  def load_and_write_mnist(self, mnist_split):
    """Loads MNIST onto memory, transforms it, and writes on disk."""

    # Construct dataset directory.
    dataset_dir = os.path.join(self.args.output_dir,
                               'dataset_mnist_%d_%s' % (self.args.num_digits,
                                                        self.args.domain_gap))

    if not os.path.exists(dataset_dir):
      os.makedirs(dataset_dir)

    # We construct the new {train,test} split from original MNIST {train,test}
    # split respectively, to ensure that no instance of digit is shared.
    original_mnist_split = mnist_split if mnist_split == 'train' else 'test'
    images, labels = self._load_mnist(original_mnist_split)

    if self.args.use_standard_dataset:
      (sample_per_number, chosen_numbers,
       image_ids) = self._load_standard_dataset(mnist_split)

    else:
      sample_per_number, chosen_numbers = self._choose_numbers_to_include(
          mnist_split)

      imageset_indices_by_label = {digit: np.where(labels == digit)[0]
                                   for digit in xrange(10)}
      for digit in xrange(10):
        if len(imageset_indices_by_label[digit]) < sample_per_number:
          raise ValueError('We lack enough number of digit examples to build'
                           'the number images.')

      image_ids = self._collect_image_ids(
          sample_per_number, chosen_numbers, imageset_indices_by_label)

      self._save_standard_dataset(sample_per_number, chosen_numbers,
                                  image_ids, mnist_split)

    n_digit_images, n_digit_labels = self._compile_number_images_and_labels(
        sample_per_number, chosen_numbers, image_ids, images)

    self._write_num_classes(dataset_dir, len(chosen_numbers), mnist_split)
    self._write_num_samples(dataset_dir, len(n_digit_labels), mnist_split)
    self._save_mnist_to_npz(dataset_dir, mnist_split,
                            n_digit_images, n_digit_labels)

  def _save_standard_dataset(self, sample_per_number,
                             chosen_numbers, image_ids, mnist_split):
    with open(os.path.join(self.args.standard_datasets_dir,
                           'dataset_mnist_%d_%s_%s.npz'
                           % (self.args.num_digits,
                              self.args.domain_gap, mnist_split)), 'w') as f:
      np.savez(f, sample_per_number=sample_per_number,
               chosen_numbers=chosen_numbers, image_ids=image_ids)

  def _load_standard_dataset(self, mnist_split):
    with np.load(os.path.join(self.args.standard_datasets_dir,
                              'dataset_mnist_%d_%s_%s.npz'
                              % (self.args.num_digits,
                                 self.args.domain_gap, mnist_split))) as d:
      sample_per_number = d['sample_per_number']
      chosen_numbers = d['chosen_numbers']
      image_ids = d['image_ids']
      return sample_per_number, chosen_numbers, image_ids


def main():
  parser = argparse.ArgumentParser(
      description='Create an n-digit MNIST dataset.')

  parser.add_argument('--num_digits', default=1, type=int,
                      help='Number of concatenated digits per data point.')

  parser.add_argument('--domain_gap', default='instance', type=str,
                      choices=['instance', 'number'],
                      help='How to split training and test sets'
                           'of the n-digit mnist.')

  parser.add_argument('--number_ratio_train', default=70, type=int,
                      help='When domain_gap is number{m}, we decide'
                           'the ratio between the size of the set of numbers'
                           'for train vs test')

  parser.add_argument('--total_num_train', default=100000, type=int,
                      help='Number of training samples per number.')

  parser.add_argument('--total_num_test', default=10000, type=int,
                      help='Number of testing samples per number.')

  parser.add_argument('--min_num_instance_per_number', default=100, type=int,
                      help='Minimal number of instances per n-digit number. '
                           'Needed to ensure minimal positive pairs for metric '
                           'embedding training. If the total_num_train/test is '
                           'too small to ensure min_num_instance_per_number '
                           'for every possible number, subsample the set of '
                           'numbers in the split.')

  parser.add_argument('--use_standard_dataset', dest='use_standard_dataset',
                      default=False, action='store_true',
                      help='Standard dataset for reproducibility. Uses '
                           'a predefined random number table.')

  parser.add_argument('--seed', default=714, type=int,
                      help='Seed for controlling randomness.')

  parser.add_argument('--output_dir', default='data', type=str,
                      help='Directory to write the dataset.')

  parser.add_argument('--mnist_dir', default='data', type=str,
                      help='Directory with the original MNIST dataset.')

  parser.add_argument('--standard_datasets_dir', default='standard_datasets',
                      type=str, help='Standard dataset directory.')

  args = parser.parse_args()

  mnist_writer = NDigitMnist(args)

  if args.domain_gap == 'instance':
    splits = ['train', 'test']
  elif args.domain_gap == 'number':
    splits = ['train', 'test_seen', 'test_unseen']
  for mnist_split in splits:
    mnist_writer.load_and_write_mnist(mnist_split)

if __name__ == '__main__':
  main()

