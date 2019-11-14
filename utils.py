from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.python.util import compat
import numpy as np
import pathlib
import os
import re

tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

IMG_WIDTH = 224
IMG_HEIGHT = 224

labels_files = ['0000', '0001', '0100', '0101', '0010', '1000', '1010'] # TODO Change when finish testing
labels_files = ['001001', '001010', '001100', '010001', '010010', '010100', '100001', '100010', '100100']

def get_label(file_path):
  # convert the path to a list of path components
  # parts = tf.strings.split(file_path, os.path.sep)
  print('### GET LABEL ###')
  rep1 = tf.strings.regex_replace(file_path, r'\w+\/frame_\d+_', '')
  rep2 = tf.strings.regex_replace(rep1, r'.jpg$', '')
  
  # parts = re.sub(r'.jpg$', '', re.sub(r'\w+\/frame_\d+_', '', file_path))
  # The second to last is the class-directory
  labels = ['001001', '001010', '001100', '010001', '010010', '010100', '100001', '100010', '100100']
  labels = ['0000', '0001', '0100', '0101', '0010', '1000', '1010']
  # labels = [[label] for label in labels]
  p = tf.strings.split(rep2, result_type='RaggedTensor')
  print(p)
  labels = np.array(labels)
  bbb = tf.cast(file_path, tf.string)
  # print(bbb.numpy())
  print('####')
  print(labels)
  rep2 = tf.strings.strip(rep2)
  print('001001' == labels)
  # print(tf.decode_raw(rep2, tf.uint8))
  print(rep2 == labels)
  return rep2 == labels

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
  #label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img#, label

data_dir = pathlib.Path('output')
list_ds = tf.data.Dataset.list_files(str(data_dir/'*'))


label_names = [item.name for item in data_dir.glob('*')]
labels = list(map(lambda filename: re.sub(r'^frame_\d+_', '', filename), label_names))
label_names = list(map(lambda filename: re.sub(r'.jpg$', '', filename), labels))
# labels = list(map(lambda filename: re.sub(r'\w+\/frame_\d+_', '', filename), label_names))
print(set(label_names))
label_to_index = dict((name, index) for index, name in enumerate(labels_files))
all_image_labels = [label_to_index[label] for label in label_names]
print(all_image_labels[:10])

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
# for f in list_ds.take(5):
#   print(type(f))
#   print(f.numpy())
  # print(f)

# a = list_ds.take(1)
# print(a[0].numpy())
image_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

print(image_label_ds)

for image, label in label_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print(type(label))
  print("Label: ", label.numpy()) #.numpy())