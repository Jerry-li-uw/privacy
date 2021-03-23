
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import itertools

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# census = pd.read_csv("adult copy.data")

# census.loc[(census.income == '<=50K'),'income'] = 0
# census.loc[(census.income == '>50K'),'income'] = 1

# for name, column in census.items():
#   dtype = column.dtype
#   if dtype == int:
#     column = column.astype(np.float32)

# census_features = census.copy()
# census_labels = census_features.pop('income')
# census_labels = np.asarray(census_labels).astype(np.float32)
# inputs = {}

# for name, column in census_features.items():
#   dtype = column.dtype
#   if dtype == object:
#     dtype = tf.string
#   else:
#     dtype = tf.float32
  
#   inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

# # preprocess numeric inputs
# numeric_inputs = {name:input for name,input in inputs.items()
#                   if input.dtype==tf.float32}

# x = layers.Concatenate()(list(numeric_inputs.values()))
# norm = preprocessing.Normalization()
# norm.adapt(np.array(census[numeric_inputs.keys()]))
# all_numeric_inputs = norm(x)

# preprocessed_inputs = [all_numeric_inputs]

# # preprocess strings
# for name, input in inputs.items():
#   if input.dtype == tf.float32:
#     continue
  
#   lookup = preprocessing.StringLookup(vocabulary=np.unique(census_features[name]))
#   one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())

#   x = lookup(input)
#   x = one_hot(x)
#   preprocessed_inputs.append(x)

# preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

# census_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

# census_dict = {name: np.array(value) 
#                for name, value in census_features.items()}

# def weighted_binary_crossentropy_vector_loss(y_true, y_pred):

#   one_weight = 0.76
#   zero_weight = 0.24
#   bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#   # weighted calc
#   weight_vector = y_true * one_weight + (1 - y_true) * zero_weight
#   weighted_b_ce = tf.multiply(weight_vector, bce(y_true, y_pred))

#   return weighted_b_ce

# def census_model(preprocessing_head, inputs):
#   body = tf.keras.Sequential([
#     layers.Dense(64),
#     layers.Dense(1)
#   ])

#   preprocessed_inputs = preprocessing_head(inputs)
#   result = body(preprocessed_inputs)
#   model = tf.keras.Model(inputs, result)

#   # model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
#   #               optimizer=tf.optimizers.Adam())
#   model.compile(loss=weighted_binary_crossentropy,
#                 optimizer=tf.optimizers.Adam(learning_rate=0.01))
#   return model


#---------------------------------------------
# training using dpsgd

import time

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf1

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from tensorflow_privacy.privacy.optimizers import dp_optimizer
# import census_common as common

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', .01, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('epochs', 10, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 256, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')

FLAGS = flags.FLAGS


def get_logits(features):
  print(features)
  logits = tf1.keras.layers.Dense(64).apply(features)

  return logits


def cnn_model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
  """Model function"""

  logits = get_logits(features)

  # Calculate loss as a vector (to support microbatches in DP-SGD).
  vector_loss = tf1.nn.weighted_cross_entropy_with_logits(
      labels=labels, logits=logits, pos_weight=0.76/0.24)
  # Define mean of loss across minibatch (for reporting through tf.Estimator).
  scalar_loss = tf1.reduce_mean(input_tensor=vector_loss)

  # Configure the training op (for TRAIN mode).
  if mode == tf1.estimator.ModeKeys.TRAIN:
    if FLAGS.dpsgd:
      # Use DP version of GradientDescentOptimizer. Other optimizers are 
      # available in dp_optimizer. Most optimizers inheriting from
      # tf.train.Optimizer should be wrappable in differentially private
      # counterparts by calling dp_optimizer.optimizer_from_args().
      optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
          l2_norm_clip=FLAGS.l2_norm_clip,
          noise_multiplier=FLAGS.noise_multiplier,
          num_microbatches=FLAGS.microbatches,
          learning_rate=FLAGS.learning_rate)
      opt_loss = vector_loss
    else:
      optimizer = tf1.train.GradientDescentOptimizer(
          learning_rate=FLAGS.learning_rate)
      opt_loss = scalar_loss

    global_step = tf1.train.get_global_step()
    train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)

    # In the following, we pass the mean of the loss (scalar_loss) rather than
    # the vector_loss because tf.estimator requires a scalar loss. This is only
    # used for evaluation and debugging by tf.estimator. The actual loss being
    # minimized is opt_loss defined above and passed to optimizer.minimize().
    return tf1.estimator.EstimatorSpec(
        mode=mode, loss=scalar_loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode).
  elif mode == tf1.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        'accuracy':
            tf1.metrics.accuracy(
                labels=labels,
                predictions=tf1.argmax(input=logits, axis=1))
    }
    return tf1.estimator.EstimatorSpec(mode=mode,
                                      loss=scalar_loss,
                                      eval_metric_ops=eval_metric_ops)

#----------------------------------

def train_input_fn():
  census = tf.data.experimental.make_csv_dataset(
      "adult copy.data", batch_size=256,
      label_name="income")
  census_batches = (
      census.cache().repeat().shuffle(32561)
      .prefetch(tf.data.experimental.AUTOTUNE))
  return census_batches

def eval_input_fn():
  census = tf1.data.experimental.make_csv_dataset(
      "adult copy.data", batch_size=256,
      label_name="income")
  census_batches = (
      census.cache().shuffle(32561)
      .prefetch(tf.data.experimental.AUTOTUNE))
  return census_batches

def train_input_fn1():
  titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
  titanic = tf.data.experimental.make_csv_dataset(
      titanic_file, batch_size=32,
      label_name="survived")
  titanic_batches = (
      titanic.cache().repeat().shuffle(500)
      .prefetch(tf.data.experimental.AUTOTUNE))
  return titanic_batches

logging.set_verbosity(logging.INFO)

# Instantiate the tf.Estimator.
census_classifier = tf1.estimator.Estimator(model_fn=cnn_model_fn)

# Training loop.
steps_per_epoch = 32561 // 256
for epoch in range(1, 10 + 1):  # epoches: 10
  start_time = time.time()
  # Train the model for one epoch.
  census_classifier.train(
      input_fn=train_input_fn1,
      steps=steps_per_epoch)
  end_time = time.time()
  logging.info('Epoch %d time in seconds: %.2f', epoch, end_time - start_time)

  # Evaluate the model and print results
  eval_results = census_classifier.evaluate(input_fn=train_input_fn1)
  test_accuracy = eval_results['accuracy']
  print('Test accuracy after %d epochs is: %.3f' % (epoch, test_accuracy))

  # # Compute the privacy budget expended.
  # if FLAGS.dpsgd:
  #   if FLAGS.noise_multiplier > 0.0:
  #     eps, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
  #         60000, FLAGS.batch_size, FLAGS.noise_multiplier, epoch, 1e-5)
  #     print('For delta=1e-5, the current epsilon is: %.2f' % eps)
  #   else:
  #     print('Trained with DP-SGD but with zero noise.')
  # else:
  #   print('Trained with vanilla non-private SGD optimizer')






