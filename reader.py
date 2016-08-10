import numpy
import os
import cPickle as pickle
from gensim import models
import json

emb_file = 'word_embedding_dict_100.pkl'
data_file = 'data.pkl'

# load pre-trained word vector
word_to_id, word_embedding = pickle.load(open(emb_file, 'r'))
print 'total words in  word id mapping: ', len(word_to_id)

# Snli corpus
label_id = {"neutral":0, "entailment":1, "contradiction":2}

max_seq_length = {}
max_seq_length[0] = 0

def parse_file(f):
  print "parse ", f
  data = []
  with open(f, 'r') as fin:
    for line in fin:
      obj = json.loads(line.strip())
      if obj['gold_label'] in label_id:
        t = {}
        t['y'] = label_id[obj['gold_label']]
        # remove . , from sentences
        t['p'] = [word_to_id[w.strip(',')] for w in obj['sentence1'].lower().strip().strip('.').split(' ') 
              if w.strip(',') in word_to_id]
        t['h'] = [word_to_id[w.strip(',')] for w in obj['sentence2'].lower().strip().strip('.').split(' ')
              if w.strip(',') in word_to_id]
        max_seq_length[0] = max(max_seq_length[0], max(len(t['p']), len(t['h'])))
        data.append(t)
  print "max sentence length in ", f, max_seq_length[0]
  return data

if not os.path.exists(data_file):
  train_set = parse_file('./corpus/snli_1.0_train.jsonl')
  cv_set = parse_file('./corpus/snli_1.0_dev.jsonl')
  test_set = parse_file('./corpus/snli_1.0_test.jsonl')
  pickle.dump((train_set, cv_set, test_set), open(data_file, 'w'))
else:
  train_set, cv_set, test_set  = pickle.load(open(data_file))
  print "size of train set: ", len(train_set)
  print "size of cv set: ", len(cv_set)
  print "size of test set: ", len(test_set)

  
def data_iterator(data_set, start, end, max_length=80):
  h = []
  hlen = []
  hmask = []
  p = []
  plen = []
  pmask = []
  y = []

  for i in range(start, end):
    _hl = len(data_set[i]['h'])
    hlen.append(_hl)
    h.append(data_set[i]['h'] + [word_to_id['__NONE__']] * (max_length - _hl))
    hmask.append([1]*(_hl + 1) + [0] * (max_length - _hl - 1))

    _pl = len(data_set[i]['p'])
    plen.append(_pl)
    p.append(data_set[i]['p'] + [word_to_id['__NONE__']] * (max_length - _pl) )
    pmask.append([1]*(_pl + 1) + [0] * (max_length - _pl - 1))
    y.append(data_set[i]['y'])

  return (numpy.asarray(p), numpy.asarray(plen), numpy.asarray(pmask, dtype=numpy.float32),
          numpy.asarray(h), numpy.asarray(hlen), numpy.asarray(hmask, dtype=numpy.float32),
          numpy.asarray(y))
  