# Loads a pre-trained model and generates hierarchically clustered cosine similarity 
# matrices for specific internal representations (like the entities (e1, e2) or the 
# relations (r1, r2, r3).
# 
# You can generate the matrices for a few sentences or for many. 
# Simply vary the number of stories below.
#
# Usage: 
# python3 cluster_analysis.py
#
# 

import tensorflow
import sys 
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sys.argv = [sys.argv[0], 0, "tmp"] # force parameters for easy use

from tpr_rnn_graph import * # this import will create the graph

from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns


# print test and validation set performance
print("evaluate a random model:")
#full_eval() # evaluate the random initialized model
print()
print("restoring a trained model ...")
saver.restore(sess, "pre_trained/model.ckpt")
print()
print("evaluate the trained model:")
#full_eval() # evaluate the trained model
print()


def cos_sim_clustering(item, number_of_stories=1000):
  idxs = np.random.randint(low=1, high=test_epoch_size, size=number_of_stories)
  batch = [raw_test[0][idxs,:,:], 
           raw_test[1][idxs], 
           raw_test[2][idxs,:], 
           raw_test[3][idxs]]
  # r stands for representations
  all_r, all_stories = sess.run([item, story], get_feed_dic(batch))
  all_r = np.reshape(all_r, (-1, all_r.shape[-1]))
  sentences = np.reshape(all_stories, (-1, all_stories.shape[-1]))
  _, indecies = np.unique(sentences, axis=0, return_index=True)
  print("{} unique sentences found in {} random stories.".format(len(indecies), number_of_stories))
  sentences = sentences[indecies]
  r = all_r[indecies]
  C = cosine_similarity(r)
  g = sns.clustermap(C, standard_scale=1, figsize=(20,20))
  return g, sentences


def plot_small_random_sample(item):
  g, sentences = cos_sim_clustering(item, number_of_stories=5)
  g.savefig("small_plot_{}.png".format(item))
  for idx in g.dendrogram_row.reordered_ind:
    print("{:4}: {}".format(idx, translate(sentences[idx])))

plot_small_random_sample(e1)
plot_small_random_sample(e2)
plot_small_random_sample(r1)
plot_small_random_sample(r2)
plot_small_random_sample(r3)
