# Trains a TPR-RNN model from scratch. 
# 
# Usage: 
# python3 train.py [task_id] [log_subfolder]
#  - task_id: 0-20 where 0 is all tasks combined and 1-20 are the 20 bAbI tasks
#  - log_subfolder: the folder inside the logs directory
#
# Example:
# python3 train.py 0 default
#
# Result:
# - Starts training on the all-tasks objective.
# - New log folder in logs/default
#   > Will contain tensorflow event files, terminal output, best model checkpoint, etc.
# 
#

from tpr_rnn_graph import * 

train(steps=1000000, bs=32, terminal_log_every=250, validate_every=500)