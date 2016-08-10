import tensorflow as tf
from BaseMatchLSTM import *
class SimpleMatchLSTM(BaseMatchLSTM):
  '''
  match-LSTM with pre-compute word vector
  using weighted average of word vector along time axis as attention
  '''
  def __init__(self, w2v, hidden_size=128, max_length=100, num_output=3, is_training=True):
    self._vector_size = w2v.shape[1]
    self._hidden_size = hidden_size
    self._max_length = max_length

    premise = tf.placeholder(tf.int32, [None, self._max_length])
    premise_length = tf.placeholder(tf.int32, [None])
    premise_mask = tf.placeholder(tf.float32, [None, self._max_length])

    hypothesis = tf.placeholder(tf.int32, [None, self._max_length])
    hypothesis_length = tf.placeholder(tf.int32, [None])
    hypothesis_mask = tf.placeholder(tf.float32, [None, self._max_length])

    target = tf.placeholder(tf.int64, [None])

    # embedding
    with tf.device("/cpu:0"):
      embedding = tf.get_variable('embedding', [w2v.shape[0], w2v.shape[1]], 
        initializer=tf.constant_initializer(w2v))
      premise_emb = tf.nn.embedding_lookup(embedding, premise)
      hypothesis_emb = tf.nn.embedding_lookup(embedding, hypothesis)

    # mask 0 with embeddings
    multi = [1, 1, w2v.shape[1]]
    premise_emb = premise_emb * tf.tile(tf.expand_dims(premise_mask, -1), multi)
    hypothesis_emb = hypothesis_emb * tf.tile(tf.expand_dims(hypothesis_mask, -1), multi)

    # build up attention
    # use conv2d as weighted sum
    premise_emb = tf.expand_dims(premise_emb, [3]) # [batch * time * embedding * 1]

    attention_list = []
    # weighted average word embedding along time axis
    for i in range(self._max_length):
      attention_weights = tf.get_variable('att_w_'+str(i), 
                                [self._max_length, 1, 1, 1], 
                                initializer=tf.random_uniform_initializer(0.0, 1.0))
      att = tf.nn.conv2d(premise_emb, attention_weights, [1, self._max_length, 1, 1], 'SAME')
      att = tf.squeeze(att, [3]) #[batch * 1 * embedding]
      attention_list.append(att)
      
    attention = tf.concat(1, attention_list)
    
    # match LSTM
    x = tf.concat(2, [hypothesis_emb, attention])

    if is_training:
      x = tf.nn.dropout(x, 0.5)

    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    _, final_state = tf.nn.dynamic_rnn(cell, x, hypothesis_length, dtype=tf.float32)
    # final_state has two parts: final outputs and final state
    # c, h = tf.split(1, 2, final_state)

    proj_w = tf.get_variable('proj_w', [2*hidden_size, num_output], 
                initializer=tf.random_uniform_initializer(-1.0, 1.0))
    proj_b = tf.get_variable('proj_b', [num_output], 
                initializer=tf.constant_initializer(1.0))

    logit = tf.matmul(final_state, proj_w) + proj_b
    predict = tf.argmax(logit, 1)
    hit = tf.reduce_sum(tf.to_int32(tf.equal(predict, target)))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit, target))

    self._premise = premise
    self._premise_length = premise_length
    self._premise_mask = premise_mask
    self._attention = attention
    self._hypothesis = hypothesis
    self._hypothesis_length = hypothesis_length
    self._hypothesis_mask = hypothesis_mask
    self._target = target
    self._logit = logit
    self._loss = loss
    self._predict = predict
    self._hit = hit
    self._train_op = None

    if is_training:
      learning_rate = tf.Variable(0.1, dtype=tf.float32)
      self._learning_rate = learning_rate
      train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
      self._train_op = train_op

  @property
  def attention(self):
    return self._attention

  def assign_learning_rate(self, sess, value):
    sess.run(tf.assign(self._learning_rate, value))