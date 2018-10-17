# This file loads the data, builds the graph, and provides several other functionality.
# It is made with the use of a jupyter-console/notebook in mind for experimentation.
# The current configuration is the all-task model.# 
# Use one of the other files in order to train from scratch or analyse a trained model.
# 

import tensorflow as tf
import numpy as np
import pprint
import types
import copy
import time
import sys
import os

from preprocessor.reader import parse
from lib import *


###### Hyper Parameters ------------------
c = types.SimpleNamespace()
# user input
c.task_id = int(sys.argv[1])
c.log_keyword = str(sys.argv[2])

# data loading (necessary for task specific symbol_size parameter)
c.data_path = "tasks/en-valid" + "-10k"
raw_train, raw_valid, raw_test, word2id = parse(c.data_path, c.task_id)
id2word = {word2id[k]:k for k in word2id.keys()}
c.vocab_size = len(word2id)

# model parameters
c.symbol_size = c.vocab_size
c.entity_size = 90
c.hidden_size = 40
c.role_size = 20
c.init_limit = 0.10
c.LN = True

# optimizer
c.learning_rate = 0.001
c.beta1 = 0.9
c.beta2 = 0.999
c.max_gradient_norm = 5.0
c.do_warm_up = True
c.warm_up_steps = 50
c.warm_up_factor = 0.1
c.do_decay = True
c.decay_thresh = 0.1
c.decay_factor = 0.5

# other
c.log_folder = "logs/{}/{}/".format(c.log_keyword, str(c.task_id))


###### Experiment Setup ------------------
logger = init_logger(c.log_folder) # will not log tensorflow outputs like GPU infos and warnings
log = lambda *x: logger.debug((x[0].replace('{','{{').replace('}','}}') + "{} " * (len(x)-1)).format(*x[1:]))


###### Load Data ------------------
batch_size = tf.placeholder(tf.int64) # dynamic batch_size

p = np.random.permutation(len(raw_train[0])) # random permutation of the train_data
train_data = tf.data.Dataset.from_tensor_slices((raw_train[0][p],raw_train[1][p],raw_train[2][p],raw_train[3][p])).cache().repeat().batch(batch_size)
valid_data = tf.data.Dataset.from_tensor_slices((raw_valid[0],raw_valid[1],raw_valid[2],raw_valid[3])).cache().repeat().batch(batch_size)
test_data = tf.data.Dataset.from_tensor_slices((raw_test[0],raw_test[1],raw_test[2],raw_test[3])).cache().repeat().batch(batch_size)

train_iterator = train_data.make_initializable_iterator()
valid_iterator = valid_data.make_initializable_iterator()
test_iterator = test_data.make_initializable_iterator()

train_batch = train_iterator.get_next()
valid_batch = valid_iterator.get_next()
test_batch = test_iterator.get_next()

train_epoch_size = raw_train[0].shape[0]
valid_epoch_size = raw_valid[0].shape[0]
test_epoch_size = raw_test[0].shape[0]

# some task specific data attributes
max_story_length = np.max(raw_train[1])
max_sentences = raw_train[0].shape[1]
max_sentence_length = raw_train[0].shape[2]
max_query_length = raw_train[2].shape[1]

# full valid and test data requires too much memory for a single batch
valid_steps = 73
valid_batch_size = valid_epoch_size / 73  # 274
test_steps = 20
test_batch_size = test_epoch_size / 20 # 1000


###### Print Run Config ---------
log("Configuration:")
log(pprint.pformat(c.__dict__))
log("")


###### Graph Structure ---------
with tf.variable_scope("hyper_params", reuse=None, dtype=tf.float32):
  # we have dynamic hyper parameters
  _learning_rate = tf.get_variable("learning_rate", shape=[], trainable=False)
  _beta1 = tf.get_variable("beta1", shape=[], trainable=False)
  _beta2 = tf.get_variable("beta2", shape=[], trainable=False)

  # op to set hyper parameters
  hyper_param_init = [ 
    _learning_rate.assign(c.learning_rate),
    _beta1.assign(c.beta1),
    _beta2.assign(c.beta2)
  ]

with tf.variable_scope("inputs"):
  # story shape: [batch_size, max_sentences, max_sentence_length]
  story = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='story')  # [batch_size, sentences, words]
  story_length = tf.placeholder(dtype=tf.int32, shape=[None], name='story_length') # [batch_size]
  query = tf.placeholder(dtype=tf.int32, shape=[None, None], name='query')  # [batch_size, words]
  answer = tf.placeholder(dtype=tf.int32, shape=[None], name='answer')  # [batch_size]

  _batch_size = tf.shape(story)[0]
  sentence_length = tf.shape(story)[2]

with tf.variable_scope("variables"): # Note, the MLP weights are created in the MLP function.
  # initialize the embeddings  
  word_embedding = tf.get_variable(name="word_embedding", 
                                   shape=[c.vocab_size, c.symbol_size], 
                                   initializer=uniform_init(c.init_limit))
  position = tf.get_variable(name="story_position_embedding", 
                                   shape=[max_sentence_length, c.symbol_size], # [words, symbol_size]
                                   initializer=ones_init()) / max_sentence_length
  Z = tf.get_variable(name="output_embedding", 
                      shape=[c.entity_size, c.vocab_size], 
                      initializer=uniform_glorot_init(c.entity_size, c.vocab_size)) # output projection Z, final transformation

  # initial state of the TPR
  TPR_init = tf.zeros((_batch_size, c.entity_size, c.role_size, c.entity_size))

  # we use a cell and dynamic_rnn instead of a tf.while_loop due to its dynamic squence_length capability
  loopCell = LoopCell((_batch_size, c.entity_size, c.role_size, c.entity_size))

with tf.variable_scope("model"):
  with tf.variable_scope("update_module"):
    # we had problems with embedding_lookup and the optimizer implementation.
    # [batch_size, sentences, words, embedding_size]
    #sentence_emb = tf.nn.embedding_lookup(params=word_embedding, ids=story) 
    sentence_emb = tf.einsum('bswv,ve->bswe', tf.one_hot(story, depth=c.vocab_size), word_embedding)  
    # [batch_size, words, embedding_size]
    #query_emb = tf.nn.embedding_lookup(params=word_embedding, ids=query)  
    query_emb = tf.einsum('bwv,ve->bwe', tf.one_hot(query, depth=c.vocab_size), word_embedding) 

    # summing over the words of a sentence into sentence representations
    # [batch_size, sentences, embedding_size]
    sentence_sum = tf.einsum('bswe,we->bse', sentence_emb, position) # eq. 5 for a normal sentence
    # [batch_size, embedding_size]
    query_sum = tf.einsum('bwe,we->be', query_emb, position) # eq.5 for the question sentence

    # Five MLPs that extract the entity and relation representations
    e1, e2 = MLP(sentence_sum, n_networks=2, equation='bse,er->bsr', input_size=c.symbol_size, 
                   hidden_size=c.hidden_size, output_size=c.entity_size, scope="story_entity")
    r1, r2, r3 = MLP(sentence_sum, n_networks=3, equation='bse,er->bsr', input_size=c.symbol_size, 
                   hidden_size=c.hidden_size, output_size=c.role_size, scope="story_roles")

    # compute part of the tensor update outside the loop for efficency
    # (b)atch, (s)tory, (r)ole, (f)iller
    partial_add_W = tf.einsum('bsr,bsf->bsrf', r1, e2) # part of eq.10
    partial_add_B = tf.einsum('bsr,bsf->bsrf', r3, e1) # part of eq.14

    # perform loop operation using dynamic_rnn (so we can exploit dynamic sequence lengths)  
    def body(inputs, TPR):
      e1, r1, partial_add_W, e2, r2, partial_add_B, r3 = inputs 
      # e1 and e2 are [batch_size, entity_size]
      # r1 and r2 are [batch_size, role_size]
      # TPR is [batch_size, entity_size, role_size, entity_size]

      w_hat = (tf.einsum('be,br,berf->bf', e1, r1, TPR)) # eq. 9
      partial_remove_W = tf.einsum('br,bf->brf', r1, w_hat) # part of eq.10

      m_hat = (tf.einsum('be,br,berf->bf', e1, r2, TPR)) # eq. 11
      partial_remove_M = tf.einsum('br,bf->brf', r2, m_hat) # part of eq.12

      partial_add_M = tf.einsum('br,bf->brf', r2, w_hat) # part of eq.12

      b_hat = (tf.einsum('be,br,berf->bf', e2, r3, TPR)) # eq. 13
      partial_remove_B = tf.einsum('br,bf->brf', r3, b_hat) # part of eq.14
      
      # tensor product obeys a distributive law with the direct sum operation
      # this allows for a more efficient implementation
      # we first add the ops before we go from order 2 to order 3
      write_op = partial_add_W - partial_remove_W
      move_op = partial_add_M - partial_remove_M
      backlink_op = partial_add_B - partial_remove_B
      delta_F = tf.einsum('be,brf->berf', e1, write_op + move_op) \
                + tf.einsum('be,brf->berf', e2, backlink_op) # eq. 6

      # direct sum of the old state with the new ones. Removes old associations and adds new ones. 
      TPR += delta_F # eq. 4

      return [delta_F], TPR

    # we set the body of our empty loop cell and make use of 
    # the dynamic sequence_length capability of dynamic_rnn
    loopCell.call = body
    inputs = (e1, r1, partial_add_W, e2, r2, partial_add_B, r3) # all input tensors are already batch major
    _, TPR = tf.nn.dynamic_rnn(loopCell, inputs, initial_state=TPR_init, sequence_length=story_length)

  with tf.variable_scope("inference_module"):
    # for the question we use the same sentence encoding but different MLPs (these are used in the inference module)
    q_e1, q_e2 = MLP(query_sum, n_networks=2, equation='be,er->br', input_size=c.symbol_size, 
                     hidden_size=c.hidden_size, output_size=c.entity_size, scope="query_entity")
    q_r1, q_r2, q_r3 = MLP(query_sum, n_networks=3, equation='be,er->br', input_size=c.symbol_size, 
                           hidden_size=c.hidden_size, output_size=c.role_size, scope="query_roles")

    ## compute question answer ((b)atch, (e)ntity, (r)ole, (f)iller, (q)ueries)
    # simple association
    one_step_raw = tf.einsum('be,br,berf->bf', q_e1, q_r1, TPR)
    i_1 = norm(one_step_raw, active=c.LN, scope="one_step") # eq. 17

    # transitive inference
    two_step_raw = tf.einsum('be,br,berf->bf', i_1, q_r2, TPR)
    i_2 = norm(two_step_raw, active=c.LN, scope="two_step") # eq. 18

    # third step
    three_step_raw = tf.einsum('be,br,berf->bf', i_2, q_r3, TPR)
    i_3 = norm(three_step_raw, active=c.LN, scope="three_step") # eq. 19

    # it is possible to do some gating but it doesn't give any improvement.
    step_sum = i_1 + i_2 + i_3
    logits = tf.einsum('bf,fl->bl', step_sum, Z, name="logits") # projection into symbol space, eq. 20

with tf.variable_scope("outputs"):
  costs = tf.losses.sparse_softmax_cross_entropy(labels=answer, logits=logits, reduction='none')  # [batch_size, queries]
  cost = tf.reduce_mean(costs) # scalar

  predictions = tf.argmax(logits, axis=-1, output_type=tf.int32, name="predictions")  # [batch_size]

  correct = tf.cast(tf.equal(answer, predictions), tf.float32, name="correct")
  accuracy = tf.reduce_mean(correct, name="accuracy")  # []

log("all trainable tensorflow variables:")
log(pprint.pformat(tf.trainable_variables()))
log("total number of trainable parameters: ", get_total_trainable_parameters())
log("")

###### Optimizer ---------
with tf.variable_scope("optimizer"):
  optimizer = tf.contrib.opt.NadamOptimizer(_learning_rate, beta1=_beta1, beta2=_beta2)
  trainable_vars = tf.trainable_variables()
  gradients = tf.gradients(cost, trainable_vars)
  global_norm = tf.global_norm(gradients) # compute global norm
  clipped_global_norm = tf.where(tf.is_nan(global_norm), c.max_gradient_norm, global_norm) # clip NaN gradients to max norm
  clipped_gradients, gradient_norm = tf.clip_by_global_norm(t_list=gradients, 
                                                            clip_norm=c.max_gradient_norm,
                                                            use_norm=clipped_global_norm) # uses this norm instead of computing it
  train_op = optimizer.apply_gradients(zip(clipped_gradients, trainable_vars))


###### Session and Other---------
with tf.variable_scope("session_and_other"):
  saver = tf.train.Saver()
  merged_summaries = tf.summary.merge_all()
  writer = tf.summary.FileWriter(c.log_folder, graph=tf.get_default_graph())
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  #config.operation_timeout_in_ms=60000
  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())
  sess.run(hyper_param_init)

  # it seems like we have to initialize the iterators here but they are reinitialized when train() or eval() is called
  sess.run(train_iterator.initializer, {batch_size: 1})
  sess.run(test_iterator.initializer, {batch_size: 1})
  sess.run(valid_iterator.initializer, {batch_size: 1})


###### Helper Functions  ---------
_total_steps = 0
_start_time = 0
_best_valid_acc = 0.0
_decay_done = False

# train and performance functions
def get_feed_dic(batch):
  feed_dic = {
    story: batch[0],
    story_length: batch[1],
    query: batch[2],
    answer: batch[3],
  }
  return feed_dic

def train(steps=1000000, bs=128, terminal_log_every=50, validate_every=200):
  global _total_steps
  global _start_time
  global _decay_done
  # ---
  # reinitializing the iterator resets it to the first batch
  sess.run(train_iterator.initializer, {batch_size: bs})
  
  _start_time = time.time()
  _prev_time = _start_time
  _prev_step = 0

  acc_sum, cost_sum, valid_acc, valid_cost = 0, 0, 0, 999

  for i in range(steps):
    # reduce learning rate for warm up period
    if i < c.warm_up_steps and c.do_warm_up:
      sess.run(_learning_rate.assign(c.learning_rate * c.warm_up_factor))

    if i > c.warm_up_steps and c.do_warm_up:
      sess.run(_learning_rate.assign(c.learning_rate))

    # decay learning once
    if valid_cost < c.decay_thresh and c.do_decay and not _decay_done:
      saver.save(sess, os.path.join(c.log_folder, "preReduction.ckpt"))
      sess.run(_learning_rate.assign(c.learning_rate * c.decay_factor))
      _decay_done = True # prevent further decays
    
    # get a batch and run a step
    batch = sess.run(train_batch) # batch size is defined by the iterator initialization
    feed_dic = get_feed_dic(batch)
    query_dic = {
      _learning_rate.name: _learning_rate,
      train_op.name: train_op,
      cost.name: cost,
      accuracy.name: accuracy,
      gradient_norm.name: gradient_norm
    }
    result = sess.run(query_dic, feed_dic)
    cost_sum += result[cost.name]
    acc_sum += result[accuracy.name]

    # if we have NaN values during the warm_up phase we restart training
    if np.isnan(result[cost.name]) and i < c.warm_up_steps:
      log("NaN cost during warmup, reinitializing.")
      _total_steps = 0
      i = 0
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(hyper_param_init)
      continue

    # print and log the current training performance
    if (_total_steps % terminal_log_every == 0) or np.isnan(result[cost.name]) or _total_steps < c.warm_up_steps:
      # track speed
      epochs_seen = (_total_steps * bs) / train_epoch_size
      time_done = (time.time() - _prev_time) / 60.0 # in minutes
      step_speed = (i - _prev_step) / time_done
      stories_speed = step_speed * bs
      _prev_time = time.time()
      _prev_step = i

      log("{:4}: cost={:6.4f}, accuracy={:7.4f}, norm={:06.3f}, lr={:.4f} (epochs={:.1f}, steps/min={:2.0f}, stories/min={:2.0f})".format(
            _total_steps, result[cost.name], result[accuracy.name], result[gradient_norm.name], result[_learning_rate.name],
            epochs_seen, step_speed, stories_speed))
      
      writer.add_summary(make_summary(tag="train/cost", value=result[cost.name]), _total_steps)
      writer.add_summary(make_summary(tag="train/accuracy", value=result[accuracy.name]), _total_steps)

    # print and log the current validation and test performance
    if _total_steps % validate_every == 0 and _total_steps != 0:
      log("")
      log("task: ", c.task_id)
      valid_cost, valid_acc = eval(prefix="valid", batch_source=valid_batch, steps=valid_steps, bs=valid_batch_size)
      test_cost, test_acc = eval(prefix="test", batch_source=test_batch, steps=test_steps, bs=test_batch_size)

      writer.add_summary(make_summary(tag="valid/cost", value=valid_cost), _total_steps)
      writer.add_summary(make_summary(tag="valid/accuracy", value=valid_acc), _total_steps)
      writer.add_summary(make_summary(tag="test/cost", value=test_cost), _total_steps)
      writer.add_summary(make_summary(tag="test/accuracy", value=test_acc), _total_steps)

      log("Total time passed: {:.1f} min".format((time.time() - _start_time) / 60.0))
      log("")

      if valid_acc >= _best_valid_acc:
        saver.save(sess, os.path.join(c.log_folder, "model.ckpt"))

    _total_steps += 1  

  return valid_acc, valid_cost, train_acc, train_cost 

def eval(steps=1, prefix="", batch_source=valid_batch, bs=valid_epoch_size):
  acc_sum, cost_sum = 0.0, 0.0
  etime = time.time()

  sess.run(valid_iterator.initializer, {batch_size: bs})
  sess.run(test_iterator.initializer, {batch_size: bs})

  for j in range(steps):
    batch = sess.run(batch_source)
    feed_dic = get_feed_dic(batch)

    query_dic = {
      cost.name: cost,
      accuracy.name: accuracy,
    }
    result = sess.run(query_dic, feed_dic)
    cost_sum += result[cost.name]
    acc_sum += result[accuracy.name]    

  # the following fetch is only to trigger caching and have the cache warning go away
  try:
    sess.run(batch_source)
  except tf.errors.OutOfRangeError: # single-task eval doesn't have repeat()
    pass

  n_stories = steps * bs
  eval_time = (time.time() - etime) / 60.0
  log("{} evaluation: cost={:.4f}, accuracy={:.4f} ({} stories in {:.1f} min)".format(
          prefix, cost_sum / steps, acc_sum / steps, n_stories, eval_time))
  return cost_sum / steps, acc_sum / steps

def full_eval():
  eval(prefix="valid", batch_source=valid_batch, steps=valid_steps, bs=valid_batch_size)
  eval(prefix="test", batch_source=test_batch, steps=test_steps, bs=test_batch_size)


# print functions
def translate(nparr, id2word=id2word):
  assert (type(nparr) == np.ndarray), "You can only translate numpy arrays"
  old_shape = nparr.shape
  arr = np.reshape(nparr, (-1))
  arr = np.asarray([id2word[x] for x in arr])
  arr = np.reshape(arr, old_shape)
  as_string = np.apply_along_axis(lambda x: " ".join(x), axis=-1, arr=arr)
  return as_string

def show_random_sample():  
  idx = np.random.randint(2000)
  batch = [raw_test[0][np.newaxis,idx,:,:], 
           raw_test[1][np.newaxis,idx], 
           raw_test[2][np.newaxis,idx,:], 
           raw_test[3][np.newaxis,idx]]
  feed_dic = get_feed_dic(batch)

  tensors = [story, query, answer, predictions]
  query_dic = {t.name:t for t in tensors}
  res_dic = sess.run(query_dic, feed_dic)
  for t in tensors:
    print("{}: ".format(t.name))
    print(pprint.pformat(translate(res_dic[t.name])))
    print()


# functions to evaluate on all tasks
eval_valid_test = []

def transform_task(task_id):
  _train, _valid, _test, old_dic = parse(c.data_path, task_id)
  new_train = copy.deepcopy(_train)
  new_valid = copy.deepcopy(_valid)
  new_test = copy.deepcopy(_test)

  for key in sorted(list(old_dic.keys())):
    for i in [0,2,3]:
      new_train[i][ _train[i] == old_dic[key]] = word2id[key]
      new_valid[i][ _valid[i] == old_dic[key]] = word2id[key]
      new_test[i][ _test[i] == old_dic[key]] = word2id[key]

  def pad(A):
    out = []
    _maxS = 130
    _maxW = 12
    out.append(np.zeros((A[0].shape[0], _maxS, _maxW)))
    out[0][:, :A[0].shape[1], :A[0].shape[2]] = A[0]

    out.append(A[1])

    out.append(np.zeros((A[2].shape[0], _maxW)))
    out[2][:, :A[2].shape[1]] = A[2]

    out.append(A[3])
    return out

  sets = []
  sets.append(pad(new_train))
  sets.append(pad(new_valid))
  sets.append(pad(new_test))
  sets.append(word2id)
  return sets

def make_eval_tensors():
  global eval_valid_test
  eval_valid_test = []
  for i in range(1,21):
    _, raw_valid, raw_test, _ = transform_task(i)
    valid_epoch_size = raw_valid[0].shape[0]   
    test_epoch_size = raw_test[0].shape[0]

    valid_data = tf.data.Dataset.from_tensor_slices((raw_valid[0],raw_valid[1],raw_valid[2],raw_valid[3])).batch(valid_epoch_size)
    test_data = tf.data.Dataset.from_tensor_slices((raw_test[0],raw_test[1],raw_test[2],raw_test[3])).batch(test_epoch_size)

    valid_batch = valid_data.make_one_shot_iterator().get_next()
    test_batch = test_data.make_one_shot_iterator().get_next()

    eval_valid_test.append((valid_batch, valid_epoch_size, test_batch, test_epoch_size))

def eval_every_task():
  for idx, (valid_batch, valid_epoch_size, test_batch, test_epoch_size) in enumerate(eval_valid_test):
    prefix = "valid_task_" + str(idx + 1)
    eval(prefix=prefix, batch_source=valid_batch, steps=1, bs=valid_epoch_size)
  print()

  for idx, (valid_batch, valid_epoch_size, test_batch, test_epoch_size) in enumerate(eval_valid_test):
    prefix = "test_task_" + str(idx + 1)
    eval(prefix=prefix, batch_source=test_batch, steps=1, bs=test_epoch_size)
  print()

