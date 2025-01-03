import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

!gdown 13zMPxc9Wwg5jxj-W162kpqUjpQ8cdlRR

df = pd.read_csv("/content/pollution.csv")

def encoding(data, name):

    label = LabelEncoder()
    enc = OneHotEncoder()

    numerical_data = label.fit_transform(data[name])
    #y_data = enc.fit_transform(numerical_data.reshape(-1, 1)).toarray()
    data[name] = list(numerical_data)

    return data

def create_dataset(data, batch_size):

    df = data.copy()
    labels = df.pop('Air Quality') #labels will have all itens from target and drop from df

    df = {key: np.asarray(value)[:,tf.newaxis] for key, value in data.items()}

    #for key, value in val.items():
      #dt = {key: np.asarray(value)[:,tf.newaxis]}

    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))

    return ds.shuffle(buffer_size=len(df)).batch(batch_size).prefetch(batch_size)

def normalization(name, data):

  normalizer = layers.Normalization(axis=None)
  feature_ds = data.map(lambda x, y: x[name]) 
  normalizer.adapt(feature_ds)

  return normalizer


def category_encoder(name, dataset, dtype, max_tokens=None):

  if dtype == 'string':
    index = layers.StringLookup(max_tokens=max_tokens)
  else:
    index = layers.IntegerLookup(max_tokens=max_tokens)

  feature_ds = dataset.map(lambda x, y: x[name])
  index.adapt(feature_ds)

  encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

  return lambda feature: encoder(index(feature))

train, val, test = np.split(df.sample(frac=1), [int(0.8*len(df)), int(0.9*len(df))])

bs = 256
train_ds = create_dataset(train, batch_size=bs)
val_ds = create_dataset(val, batch_size=bs)
test_ds = create_dataset(test, batch_size=bs)

all_inputs = []
encoded_features = []

for header in df.columns:

  if header != 'Air Quality':

    x = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = normalization(header, train_ds)
    x1 = normalization_layer(x) #keras tensor
    
    all_inputs.append(x) #keras tensor of input
    encoded_features.append(x1) #keras tensor normalized

all_features = tf.keras.layers.concatenate(encoded_features)
x2 = tf.keras.layers.Dense(32, activation="relu")(all_features)
x2 = tf.keras.layers.Dropout(0.5)(x2)
output = tf.keras.layers.Dense(4, activation='softmax')(x2) 

model = tf.keras.Model(all_inputs, output)
model.compile(optimizer=Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

model.fit(train_ds, epochs=100, validation_data=val_ds)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
