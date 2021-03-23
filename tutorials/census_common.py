
# These are not necessary in a Python 3-only module.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf1
# import tensorflow_datasets as tfds


def get_logits(features):
  """Given input features, returns the logits from a simple CNN model."""
  logits = tf1.keras.layers.Dense(64).apply(features)

  return logits


def make_input_fn(split, input_batch_size=256, repetitions=-1, tpu=False):
  """Make input function on given MNIST split."""

  def input_fn(params=None):
    """A simple input function."""
    batch_size = params.get('batch_size', input_batch_size)

    def parser(example):
      image, label = example['image'], example['label']
      image = tf.cast(image, tf.float32)
      image /= 255.0
      label = tf.cast(label, tf.int32)
      return image, label

    dataset = tfds.load(name='mnist', split=split)
    dataset = dataset.map(parser).shuffle(60000).repeat(repetitions).batch(
        batch_size)
    # If this input function is not meant for TPUs, we can stop here.
    # Otherwise, we need to explicitly set its shape. Note that for unknown
    # reasons, returning the latter format causes performance regression
    # on non-TPUs.
    if not tpu:
      return dataset

    # Give inputs statically known shapes; needed for TPUs.
    images, labels = tf.data.make_one_shot_iterator(dataset).get_next()
    # return images, labels
    images.set_shape([batch_size, 28, 28, 1])
    labels.set_shape([
        batch_size,
    ])
    return images, labels

  return input_fn
