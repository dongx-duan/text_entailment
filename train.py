import tensorflow as tf
import numpy
import reader
from SimpleMatchLSTM import *
# from models.WordAttentionMatchLSTM import  WordAttentionMatchLSTM

with tf.Graph().as_default(), tf.Session() as session:
  with tf.variable_scope('mlstm'):
    model = SimpleMatchLSTM(reader.word_embedding, hidden_size=150, max_length=80)
  with tf.variable_scope('mlstm', reuse=True):
    test_model = SimpleMatchLSTM(reader.word_embedding, hidden_size=150, max_length=80, is_training=False)

  session.run(tf.initialize_all_variables())
  saver = tf.train.Saver()
  # reload_file = './checkpoints/SimpleMatchLSTM_50.ckpt'
  # if reload_file is not None:
  #   saver.restore(session, reload_file)
  batch_size = 256

  for epoch in range(1, 81):
    print "epoch ", epoch
    _train_loss = 0.0
    _train_hit = 0
    count = 0.0
    for start in range(0, len(reader.train_set) - batch_size, batch_size):
      end = min(len(reader.train_set), start + batch_size)
      p, plen, pmask, h, hlen, hmask, y = reader.data_iterator(reader.train_set, start, end)
      _loss, _hit,  _ = session.run([model.loss, model.hit, model.train_op], feed_dict={
              model.premise: p,
              model.premise_length: plen,
              model.premise_mask: pmask,
              model.hypothesis: h,
              model.hypothesis_length: hlen,
              model.hypothesis_mask: hmask,
              model.target: y
          })
      _train_loss += _loss * (end - start)
      _train_hit += _hit
    print "train_loss: ", _train_loss / len(reader.train_set)
    print "train_hit  ", _train_hit , " over ", len(reader.train_set)
    print "train_hit rate: ", float(_train_hit) / len(reader.train_set)
    print ""

    _cv_loss = 0
    _cv_hit = 0
    for start in range(0, len(reader.cv_set) - batch_size, batch_size):
      end = min(len(reader.cv_set), start + batch_size)
      p, plen, pmask, h, hlen, hmask, y = reader.data_iterator(reader.cv_set, start, end)
      _loss, _hit = session.run([test_model.loss, test_model.hit], feed_dict={
              test_model.premise: p,
              test_model.premise_length: plen,
              test_model.premise_mask: pmask,
              test_model.hypothesis: h,
              test_model.hypothesis_length: hlen,
              test_model.hypothesis_mask: hmask,
              test_model.target: y
          })
      _cv_loss += _loss * (end - start)
      _cv_hit += _hit
    print "cv_loss: ", _cv_loss / len(reader.cv_set)
    print "cv_hit  ", _cv_hit , " over ", len(reader.cv_set)
    print "cv_hit rate: ", float(_cv_hit) / len(reader.cv_set) 
    print ""

    _test_loss = 0
    _test_hit = 0
    for start in range(0, len(reader.test_set) - batch_size, batch_size):
      end = min(len(reader.test_set), start + batch_size)
      p, plen, pmask, h, hlen, hmask, y = reader.data_iterator(reader.test_set, start, end)
      _loss, _hit = session.run([test_model.loss, test_model.hit], feed_dict={
            test_model.premise: p,
            test_model.premise_length: plen,
            test_model.premise_mask: pmask,
            test_model.hypothesis: h,
            test_model.hypothesis_length: hlen,
            test_model.hypothesis_mask: hmask,
            test_model.target: y
      })
      _test_loss += _loss * (end - start)
      _test_hit += _hit
    print "test_loss: ", _test_loss
    print "test_hit  ", _test_hit , " over ", len(reader.test_set)
    print "test_hit rate: ", float(_test_hit) / len(reader.test_set)
    print ""
   
    if epoch == 50:
      model.assign_learning_rate(session, 0.01)

    if epoch >= 50 and epoch % 10 == 0:
      saver.save(session, 'checkpoints/' + model.__class__.__name__ + '_' + str(epoch) + '.ckpt')