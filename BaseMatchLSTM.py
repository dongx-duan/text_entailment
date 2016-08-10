class BaseMatchLSTM():
  @property
  def batch_size(self):
    return self._batch_size

  @property
  def max_length(self):
    return self._max_length

  @property
  def premise(self):
    return self._premise

  @property
  def premise_length(self):
    return self._premise_length

  @property
  def premise_mask(self):
    return self._premise_mask

  @property
  def hypothesis(self):
    return self._hypothesis

  @property
  def hypothesis_length(self):
    return self._hypothesis_length

  @property
  def hypothesis_mask(self):
    return self._hypothesis_mask

  @property
  def target(self):
    return self._target

  @property
  def logit(self):
    return self._logit

  @property
  def loss(self):
    return self._loss
  
  @property
  def predict(self):
    return self._predict

  @property
  def hit(self):
    return self._hit

  @property
  def train_op(self):
    return self._train_op
