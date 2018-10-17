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

# print test and validation set performance
print("evaluate a random model:")
full_eval() # evaluate the random initialized model
print()
print("restoring a trained model ...")
saver.restore(sess, "pre_trained/model.ckpt")
print()
print("evaluate the trained model:")
full_eval() # evaluate the trained model
print()

# show an example
print("printing a random sample:")
show_random_sample()

