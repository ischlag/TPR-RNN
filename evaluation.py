# Loads a pre-trained model, evaluates it, and prints random validation set samples.
# 
# Usage: 
# python3 evaluation.py
#
# 

import tensorflow
import sys 
import os

sys.argv = [sys.argv[0], 0, "tmp"] # force parameters for easy use

from tpr_rnn_graph import * # this import will create the graph

make_eval_tensors()
print()

# print test and validation set performance
print("evaluate a random all-tasks model on the validation and test data:")
full_eval() # evaluate the random initialized model
print()

print("restoring a trained all-tasks model ...")
saver.restore(sess, "pre_trained/model.ckpt")
print()

print("evaluate the trained all-tasks model on the validation and test data:")
full_eval() # evaluate the trained model
print()

print("evaluate the trained all-tasks model on individual tasks:")
eval_every_task()
print()

# show an example
print("printing a random sample from the all-tasks data:")
show_random_sample()

