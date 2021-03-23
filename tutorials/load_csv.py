import pandas as pd
import numpy as np
import itertools

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

census = pd.read_csv("adult copy.data")

census.loc[(census.income == '<=50K'),'income'] = 0
census.loc[(census.income == '>50K'),'income'] = 1

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

def census_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.optimizers.Adam())
  return model



census_model = census_model(census_preprocessing, inputs)
census_model.fit(x=census_dict, y=census_labels, epochs=10)
census_model.save('census_model')

def slices(features):
  for i in itertools.count():
    # For each feature take index `i`
    example = {name:values[i] for name, values in features.items()}
    yield example

for example in slices(census_dict):
  for name, value in example.items():
    print(f"{name:19s}: {value}")
  break




