import tensorflow as tf
import numpy as np
import logging
import shutil
import sys
import os

default_dtype = tf.float32

###### Helper classes ---------
class LoopCell(tf.contrib.rnn.RNNCell):
  """ 
  Dummy cell with an empty call/body function that should be overwritten 
  to easy implement a tf.while_loop with dynamic sequence lengths.
  """
  def __init__(self, h_shape, reuse=None, name=None):
    super(LoopCell, self).__init__(_reuse=reuse, name=name)
    # dummy output to make it work with dynamic_rnn
    #dummy_h = tf.zeros((batch_size, entityC_size, roleC_size, entityC_size))
    dummy_h = tf.zeros(h_shape)

    self.out = [dummy_h]

  @property
  def state_size(self):
    return 0 # not executed but function has to be implemented

  @property
  def output_size(self):
    return [o.shape[1:] for o in self.out]

  def build(self, inputs_shape):
    # init variables here
    self.built = True

  def call(self, inputs, state):
    # overwritten in the model part
    return self.out, state


###### Helper Functions ---------
def zeros_init(dtype=default_dtype):
  return tf.zeros_initializer(dtype=dtype)

def ones_init(dtype=default_dtype):
  return tf.ones_initializer(dtype=dtype)

def uniform_init(limit, dtype=default_dtype):
  return tf.random_uniform_initializer(minval=-limit, maxval=limit, dtype=dtype)

def uniform_glorot_init(in_size, out_size, dtype=default_dtype):
  a = np.sqrt(6.0 / (in_size + out_size))
  return tf.random_uniform_initializer(minval=-a, maxval=a, dtype=dtype)

def get_affine_vars(prefix, shape, w_initializer, b_initializer=zeros_init()):
  weights = tf.get_variable(prefix + "_w", shape=shape, initializer=w_initializer)
  bias = tf.get_variable(prefix + "_b", shape=[shape[-1]], initializer=b_initializer)
  return weights, bias

def make_summary(tag, value):
  return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])

def norm(item, active, scope, axes=[-1]):
  """ Layernorm """
  if not active:
    return item
  mean, var = tf.nn.moments(item, axes=axes, keep_dims=True)
  normed = (item - mean) / tf.sqrt(var + 0.0000001)
  
  with tf.variable_scope(scope):
    gain = tf.get_variable("LN_gain", [1], initializer=ones_init())
    bias = tf.get_variable("LN_bias", [1], initializer=zeros_init())
    normed = normed * gain + bias
  return normed

def MLP(inputs, n_networks, equation, input_size, hidden_size, output_size, scope):
  """ compute the output of n distinct 3-layer MLPs in series. """
  with tf.variable_scope(scope):
    outputs = []
    for idx in range(n_networks):
      W1_w, W1_b = get_affine_vars("W1_net"+str(idx), shape=[input_size, hidden_size], w_initializer=uniform_glorot_init(input_size, hidden_size))
      W2_w, W2_b = get_affine_vars("W2_net"+str(idx), shape=[hidden_size, output_size], w_initializer=uniform_glorot_init(hidden_size, output_size))

      hidden = tf.nn.tanh(tf.einsum(equation, inputs, W1_w) + W1_b)
      out = tf.nn.tanh(tf.einsum(equation, hidden, W2_w) + W2_b)

      outputs.append(out)
    return outputs

def get_total_trainable_parameters():
  total_parameters = 0
  for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
      variable_parametes *= dim.value
    total_parameters += variable_parametes
  return total_parameters

def init_logger(log_folder, file_name="output.log"):
  if os.path.exists(log_folder):
    print("WARNING: The results directory (%s) already exists. Remove results directory [y/N]? " % log_folder, end="")
    var = input()
    if var is "y" or var is "Y":
      print("removing directory ...")
      shutil.rmtree(log_folder, ignore_errors=True)
    else:
      print("ERROR: The results directory already exists: %s" % log_folder)
      sys.exit(1)  
  os.makedirs(log_folder)
  log_file_path = os.path.join(log_folder, file_name)

  logger = logging.getLogger("my_logger") # unable to use a new file handler with the tensorflow logger 
  logger.setLevel(logging.DEBUG)
  logger.addHandler(logging.FileHandler(log_file_path))
  logger.addHandler(logging.StreamHandler())
  
  return logger
