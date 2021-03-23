import pandas as pd
import numpy as np
import itertools
import sys 

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
import dp_mod as dp_mod
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import dp_optimizer_adp as adp

tf.get_logger().setLevel('ERROR')
census = pd.read_csv("data/census-income.data")
census = census.sample(n=60000)

census.loc[(census.income == ' - 50000.'),'income'] = 0
census.loc[(census.income == ' 50000+.'),'income'] = 1
neg, pos = np.bincount(census['income'])
total = neg + pos

zero_weight = total / neg / 2
one_weight = total / pos / 2

for name, column in census.items():
  dtype = column.dtype
  if dtype == int:
    column = column.astype(np.float32)

census_features = census.copy()
census_labels = census_features.pop('income')
census_labels = np.asarray(census_labels).astype(np.float32)
inputs = {}

for name, column in census_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32
  
  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

# preprocess numeric inputs
numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = preprocessing.Normalization()
norm.adapt(np.array(census[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

preprocessed_inputs = [all_numeric_inputs]

# preprocess strings
for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue
  
  lookup = preprocessing.StringLookup(vocabulary=np.unique(census_features[name]))
  one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)

preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

census_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

census_dict = {name: np.array(value) 
               for name, value in census_features.items()}

aucs = {}

def weighted_binary_crossentropy(y_true, y_pred):

  one_weight = 0.76
  zero_weight = 0.24
  bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  # weighted calc
  weight_vector = y_true * one_weight + (1 - y_true) * zero_weight
  weighted_b_ce = tf.multiply(weight_vector, bce(y_true, y_pred))

  return tf.math.reduce_mean(weighted_b_ce)


from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from tensorflow_privacy.privacy.optimizers import dp_optimizer
gr_norm = []
clip = float(sys.argv[1])
def census_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid", name="predictions")
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)


  # model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
  #               optimizer=tf.optimizers.Adam(learning_rate=0.0001),
  #               metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
  # GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
  # DPGradientDescentOpt = dp_mod.make_gaussian_optimizer_class(GradientDescentOptimizer)
  # model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
  #               optimizer = DPGradientDescentOpt(
  #                 l2_norm_clip=float(sys.argv[1]),
  #                 noise_multiplier=float(sys.argv[2]),
  #                 num_microbatches=256,
  #                 learning_rate=0.001),
  #               metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
  opt = adp.DPKerasSGDOptimizer(
      l2_norm_clip=clip,
      noise_multiplier=float(sys.argv[2]),
      num_microbatches=1,
      learning_rate=0.05,
      gradient_norm=gr_norm,
      changing_clipping=True 
      )
  model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                optimizer = opt,
                metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
 
  return model

census_model = census_model(census_preprocessing, inputs)
hist = census_model.fit(x=census_dict, y=census_labels, 
                        validation_split=0.4, epochs=20, 
                        class_weight={0:zero_weight, 1:one_weight},
                        batch_size=600,
                        verbose=0)

print(hist.history)


