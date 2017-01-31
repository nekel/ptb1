# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
#
# Lena 1/4/2017
# A very simplified original ptb_word_lm file
# Works "almost interactively" if I import everything from it
# Then run imp() (perhaps not even necessary??)
# followed by one iteration of main(None)
# the results are the same as in original   
#
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Example / benchmark for building a PTB LSTM model.
Lena: using shakespeare file instead of zaremba dataset
sving logits; will see what could be done with them
the first idea is to take argmax for each of 375 lines in logit,
take a corresponding letter and see the resulting sentence

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""

import os,sys,time
import numpy as np, tensorflow as tf
#from tensorflow.models.rnn.ptb import reader
from importlib import reload
from datetime import datetime

MAINdir= 'C:\\users\\Lena\\Anaconda\\envs\\snakes\\'
DATApath='simple-examples\\data'
DATAdir=MAINdir+DATApath
SHAKESPEARE ='C:\\users\\lena\\desktop\\2017NeuralNet\\shakespeare.txt'
WARPEACE='C:\\users\\lena\\desktop\\warpeace_input.txt'
ABStr = ' abcdefghijklmnopqrstuvwxyz\'\n.,?!:;-'
#########0123456789012345678901234567890123456789
PTBimport = 'tensorflow.models.rnn.ptb'

################### Lena: STEP 1

class SmallConfig(object):
  """Small config.
  """
  init_scale = 0.1
  learning_rate = 0.00004 #0.00015  #for Adam optimizer 0.001 #0.25 #0.009  #1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 21   #42
  hidden_size = 300 #200
  embed_size=300
  max_epoch = 1
  max_max_epoch = 10
  keep_prob = 0.5
  lr_decay = 1.
  batch_size = 20 
  vocab_size = 36
  forget_bias=1.0
  input_file=WARPEACE
  # use ptb style embedding: hidden_size, vocab_size
  model_dir=MAINdir+'ptb0_embed300_f1\\'
  logfile='test.01.27.17.txt'
#tv9,vv9,tt9=main(None)
#  fetc=tt9['logits']
#  s=preca_eps(fetc,0.1)
#  print s
#  h=histogr(s)


def imps():
  """
  run this in the beginning of interactive session
  (prior to running main interactively)
  Avoid tf.flags at all costs - no advantages except complications
  """
  if True:
      import os,sys, time, numpy as np, tensorflow as tf
      #from tensorflow.models.rnn.ptb import reader
      from pprint import pprint
      from importlib import reload
      from datetime import datetime
      datestr=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      config = SmallConfig()  ### lena
      MAINdir= 'C:\\users\\Lena\\Anaconda\\envs\\snakes\\'
      PTBdir=MAINdir+'Lib\\site-packages\\tensorflow\\models\\rnn\\ptb\\'
      MODELdir=config.model_dir #MAINdir+'Lena_ptb6\\'
      SHAKESPEARE ='C:\\users\\lena\\desktop\\2017NeuralNet\\shakespeare.txt' 
      ABStr = 'abcdefghijklmnopqrstuvwxyz\' \n.,?!:;-'
      #ABStr = " etaonihsrdlu\nmcwfgyp,b.vk'-!x?jzq;:"
      DATApath='simple-examples\\data\\'
      DATAdir=MAINdir+DATApath
      def data_type():
        return tf.float32
      config = SmallConfig()  ### lena
      eval_config = SmallConfig()  ### lena
      eval_config.batch_size = 1
      eval_config.num_steps = 500
      
 
      #raw_data = reader.ptb_raw_data(DATAdir)  ### lena
      raw_data = ptb_raw_data()  ### lena
      train_data, valid_data, test_data, file_strn = raw_data

      #train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      #ida,ts=reader.ptb_producer(train_data,\
                    #config.batch_size, config.num_steps, name="TrainInput")

def histogr(strn='',filename=MAINdir+'tolst_data.py',abstr=ABStr):
      """
      how many letters of each kind occurs in a given text
      skip letters not in ABStr
      """
      D={}
      if strn=='':
        for lin in open(filename,'r').readlines():
          strn+=lin
      for x in abstr: D[x]=0
      for x in strn.lower():
        if x in abstr:
          D[x]+=1
      zs= [(D[x],x) for x in D.keys()]
      zs.sort()
      zs.reverse()     
      #D1=dict(zip(abstr, range(len(abstr))))
      #D2= dict(zip(range(len(abstr)),abstr))
      return zs


def grep(filename,line, nwin=0, case=False):
      """
      grep-like function
      """
      prstr=''
      if not case: line=line.lower()
      LL= open(filename,'r').readlines()
      if not case:
        LL=[x.lower() for x in LL]
      for n in range(len(LL)-(2*nwin+1)):
        block=LL[n:(n+2*nwin+1)]
        x=block[nwin]
        if x.find(line)>=0:
          for b in block:
            prstr+=b
          prstr+='--------\n'
      return prstr

    
def twocurves(filename=MAINdir+'test.01.24.15'):
      """
      process output of ptb6 for plotting with matplotlib
      afterwards, try plot3(scatter(twocurves))
      filename=MAINdir+'tolst_data.py' or filename=MAINdir+'test.01.24.15'
      """
      LL= open(filename,'r').readlines()
      valids=[x for x in LL if (x.lower().find('epoch:')>=0 and x.lower().find('valid')>=0)]
      trains=[x for x in LL if (x.lower().find('epoch:')>=0 and x.lower().find('train')>=0)]
      vns=[eval(x.split(' ')[-1]) for x in valids]
      tns=[eval(x.split(' ')[-1]) for x in trains]
      return np.array(vns),np.array(tns)
      ##plot3(scatter(twocurves))

def plot3(vns,tns):
      import numpy as np
      import matplotlib.pyplot as plt
      # red dashes, blue squares and green triangles
      plt.plot(vns, 'r--', tns, 'bs')   #, t, t**3, 'g^')
      plt.show()
    
def data_type():
  return  tf.float32

def file_to_char_ids(filename, abstr=ABStr, V=True):
  """
  reads a text file, returns a string and a list of corresponding char ids
  """
  Dchar_to_id=dict(zip(abstr, range(len(abstr)))) #buil vocab D[char]=num
  strn=''
  LL= open(filename,'r').readlines()
  for x in LL:
    strn +=x.lower() 
  if V: print(("\nFile %s has %d characters and %d distinct frequent characters ") 
             % (filename, len(strn), len(abstr)))
  #strn=strn.replace(' t',' ') #!!! for testing only! delete later!!!
  return strn, [Dchar_to_id[x] for x in strn if x in Dchar_to_id]



def print_around_break(filename=SHAKESPEARE):
  """
  print segments of shakespeare file for validation purposes
  pretty useless :)
  """
  strn=''
  LL= open(filename,'r').readlines()
  for x in LL:
    strn +=x.lower() 
  nbreak=int(len(strn)/20)
  print(strn[(17*nbreak-400):(17*nbreak+30)],'-----------------\n')
  print(strn[(19*nbreak-400):(19*nbreak+30)])
  #el=[m.start() for m in re.finditer('elizabeth:', strn)]
  #return el


def ptb_raw_data(filename=WARPEACE, abstr=ABStr):
  """
  load raw data from filename; by default it is now WARPEACE, not SHAKESPEARE
  to count occurences:
  [train_data.count(k) for k in range(len(ABStr))]
  """
  strn, idlist = file_to_char_ids(filename)  #len(idlist) around 3196000
  
  if filename==WARPEACE:
    nbreak1,nbreak2,nbreak3=len(idlist)-120000,len(idlist)-20000,len(idlist)-1000
  if filename==SHAKESPEARE:
    nbreak=int(len(strn)/20)
    nbreak1,nbreak2 = 17*nbreak,19*nbreak
  train_data=idlist[:nbreak1-200]  # 3055816 947155
  #train_data=idlist[:(nbreak1//4)]  # 3055816 947155
  valid_data=idlist[nbreak1:nbreak2-200] #100000    1058585
  valid_data=idlist[nbreak1:(nbreak1+20000)] #100000    1058585
  test_data= idlist[nbreak2+100:-300]  #19800 testing starts from the left!
  return train_data, valid_data,test_data,strn

  
def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
    y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
    return x, y

def save_to_file(var,fname='valid_logit.375x36.txt'):
      """
      This is old stuff: I prefer writing to file directly now
      vlog.shape ---> (375, 36)
      save_to_file(trlog,'train_logit.375x10000.txt')
      save_to_file(vlog,'valid_logit.375x10000.txt')
      save_to_file(telog,'test_logit.10000.txt')
      """
      fid=open(fname,'w')
      for i in range(len(var)):
        fid.write('[',)
        var[i].tofile(fid,",","%6.3f")
        fid.write("],\n")
      fid.close()
  
def preca(fetc):
        """
        predicting carpati from say fetc=vlog=vvs['logits']
        """
        strn=''
        li=[ABStr[v.argmax()] for v in fetc]
        for c in li:
          strn+=c
        return strn

def preca_eps(fetc,eps):
        """
        predicting carpati with a twist from say fetc=vlog=vvs['logits']
        """
        strn=''
        for v in fetc:
          c=ABStr[v.argmax()]
          a=v.copy()
          a.sort()
          aa=[(x-a.mean())/a.std() for x in a]
          if (aa[-1]-aa[-2]>eps):
            c=c.upper()
          else:
            if (not c.isalpha()):  c='?'
          strn+=c
        return strn


def preca_test(fetc,eps):
        """
        predicting carpati with a twist from say fetc=vlog=vvs['logits']
        not using now...
        """
        strn=''
        for v in fetc:
          c='_'
          a=v.copy()
          a.sort()
          aa=[(x-a.mean())/a.std() for x in a]
          if (aa[-1]-aa[-2]>eps):
            c=c.upper()
          else:
            if (not c.isalpha()):  c='?'
          strn+=c
        return strn

      
        
####################################

   
class PTBInput(object):
  """The input data.
    ptb_protucer returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.
    """

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    #self.input_data, self.targets = reader.ptb_producer(
    self.input_data, self.targets = ptb_producer(
        data, batch_size, num_steps, name=name)

##input_
class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    embed_size = config.embed_size

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size,
                  forget_bias=config.forget_bias, state_is_tuple=True) ###
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())   ###

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
        "embedding", [vocab_size, embed_size], dtype=data_type())
#        "embedding",  initializer=tf.eye(vocab_size), dtype=data_type())
#          "embedding", [vocab_size, size], dtype=data_type(),
#            initializer=tf.eye(vocab_size))
      ###embedding=tf.eye(vocab_size)
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state         ###
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):  ###
        if time_step > 0: tf.get_variable_scope().reuse_variables()         ###
        (cell_output, state) = cell(inputs[:, time_step, :], state)          ###
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    #logits = tf.matmul(output, softmax_w) + softmax_b
    self._logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        #[logits],
        [self._logits],
        [tf.reshape(input_.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    self._embedding = embedding

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    ##optimizer = tf.train.GradientDescentOptimizer(self._lr)
    ##optimizer = tf.train.AdamOptimizer(self._lr)
    optimizer = tf.train.AdamOptimizer() #default learning_rate is 0.001) 
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  ##def assign_embedding(self, session):
    ##session.run(embedding, feed_dict = tf.eye(vocab_size)???

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def logits(self):
    return self._logits

  @property
  def embedding(self):
    return self._embedding
  
  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op



class OldSmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 36 #10000
  
class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

def pprint_class(X=SmallConfig):
      """
      print all alphanumeric fields in a given class
      """
      d,it=[],list(X.__dict__.items())
      it.sort()
      for (k,v) in it:
        if isinstance(v,(int,float,str)):
          d+=(k,v)
      return d

    
def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
      "logits": model.logits,
      "embedding": model.embedding
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h
    
# for model=m:
#0 Tensor("Train/Model/zeros:0", shape=(15, 200), dtype=float32)
      #Tensor("Train/Model/zeros_1:0", shape=(15, 200), dtype=float32)
#1 Tensor("Train/Model/zeros_2:0", shape=(15, 200), dtype=float32)
      #Tensor("Train/Model/zeros_3:0", shape=(15, 200), dtype=float32)

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]
    logits = vals["logits"]
    embed = vals["embedding"]
    emmax = [max(x) for x in embed]
      
    costs += cost
    iters += model.input.num_steps
    

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.6f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))
      #print ('iter = ', iters, ', logit is:',logit,'\n*************\n')

  state = vals["final_state"]
  logits = vals["logits"]
  embed = vals["embedding"]
  emmax = [max(x) for x in embed]
      
  print ('iter = ', iters)
  print ('embed diag: '+len(embed)*"%4.2f " % tuple(embed.diagonal()))
  print (('embed max:  '+len(embed)*"%4.2f "+"*******") % tuple(emmax))
      
  return np.exp(costs / iters), vals



def main(_):
  """
  1. initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
  2.with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
        ##tf.scalar_summary("Training Loss", m.cost)
        ##tf.scalar_summary("Learning Rate", m.lr)
        tf.summary.scalar("Training Loss", m.cost)
        tf.summary.scalar("Learning Rate", m.lr)
        gv=tf.global_variables()
        _=[print(eval('v.name')) for v in gv]
        
   <tf.Tensor 'Train/Model/Training_Loss:0' shape=() dtype=string>

  3.with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
        ##tf.scalar_summary("Validation Loss", mvalid.cost)
        tf.summary.scalar("Validation Loss", mvalid.cost)
        print ('divider')
        gv=tf.global_variables()
        _=[print(eval('v.name')) for v in gv]
        
   <tf.Tensor 'Valid/Model/Validation_Loss:0' shape=() dtype=string>

  4. with tf.name_scope("Test"):
      test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

  5. sv = tf.train.Supervisor(logdir=MODELdir)

  6. session=sv.managed_session()
                - wrong type, also, CANNOT do this: 
  session=sv.managed_session().__enter__()
   >>> type(session)
   <class 'tensorflow.python.client.session.Session'>

  7. train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)
  
   Traceback (most recent call last):
  File "<pyshell#157>", line 1, in <module>
    train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)
  File "C:/Users/lena/Anaconda/envs/snakes/ptb2.py", line 301, in run_epoch
    state = session.run(model.initial_state)
   AttributeError: '_GeneratorContextManager' object has no attribute 'run'

  Epoch: 1 Learning rate: 1.000
  0.004 perplexity: 5147.709 speed: 1294 wps
  0.104 perplexity: 837.542 speed: 1668 wps
  0.204 perplexity: 620.831 speed: 1690 wps
  0.304 perplexity: 500.984 speed: 1700 wps
  0.404 perplexity: 431.946 speed: 1705 wps
  0.604 perplexity: 348.731 speed: 1711 wps
  0.703 perplexity: 322.436 speed: 1713 wps
  0.803 perplexity: 301.708 speed: 1714 wps
  0.903 perplexity: 282.535 speed: 1714 wps
  Epoch: 1 Train Perplexity: 268.278
  Epoch: 1 Valid Perplexity: 179.752
  """
  config = SmallConfig()  ### lena
  eval_config = SmallConfig()  ### lena
  eval_config.batch_size = 1
  eval_config.num_steps = 1500
  MODELdir=config.model_dir
  logfile=config.logfile

  fd = open(logfile,'a')
  datestr=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  print('\n******* '+datestr + ' *********\n\n')
  print(('\n******* '+datestr + ' *********'),file=fd)
  print(SmallConfig, pprint_class())
  print((SmallConfig, pprint_class()),file=fd,flush=True)
  if not os.path.exists(MODELdir):   
    print('-- creating '+MODELdir)
    print(('-- creating '+MODELdir), file=fd)
    os.mkdir(MODELdir)
    
  #raw_data = reader.ptb_raw_data(DATAdir)  ### lena
  raw_data = ptb_raw_data(config.input_file)  ### lena
  train_data, valid_data, test_data, _ = raw_data

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.name_scope("Train"):
      #print(train_input.__dict__)
#{'targets': <tf.Tensor 'TrainInput_1/Slice_1:0' shape=(15, 25) dtype=int32>,
#'epoch_size': 2478, 'batch_size': 15,
#'input_data': <tf.Tensor 'TrainInput_1/Slice:0' shape=(15, 25) dtype=int32>,
#'num_steps': 25}
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
        ##tf.scalar_summary("Training Loss", m.cost)
        ##tf.scalar_summary("Learning Rate", m.lr)
        tf.summary.scalar("Training Loss", m.cost)
        tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
        ##tf.scalar_summary("Validation Loss", mvalid.cost)
        tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    sv = tf.train.Supervisor(logdir=MODELdir)
    with sv.managed_session() as session:
      ##m.assign_embedding
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.6f" % (i + 1, session.run(m.lr)))
        #should uncomment for non-ADAM gradient methods
        #fd.write("Epoch: %d Learning rate: %.6f \n" % (i + 1, session.run(m.lr)))
        
        train_perplexity, tvs = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.6f" % (i + 1, train_perplexity))
        print("Epoch: %d Train Perplexity: %.6f" % (i + 1, train_perplexity), \
              file=fd)

  
        valid_perplexity, vvs = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.7f" % (i + 1, valid_perplexity))
        print("Epoch: %d Valid Perplexity: %.7f" % (i + 1, valid_perplexity),\
                 file=fd,flush=True)
        
      test_perplexity,tts = run_epoch(session, mtest)
      prstr=("Test Perplexity: %.6f" % test_perplexity)
      print(prstr)
      print(prstr+'\n', file=fd)

      if MODELdir:
        print("Saving model to %s." % MODELdir)
        print("Saving model to %s." % MODELdir,file=fd)
        sv.saver.save(session, MODELdir, global_step=sv.global_step)
  fetc, fetv=tts['logits'],vvs['logits']
  print(preca_eps(fetc,0.1))
  print (histogr(preca_eps(fetc,0.0001)))
  print(preca_eps(fetc,0.01),file=fd)
  print (histogr(preca_eps(fetc,0.0001)),file=fd)
  print(preca_eps(fetv,0.01),file=fd)
  print (histogr(preca_eps(fetv,0.0001)),file=fd)

  datestr=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  fd.write('\n--finished on '+datestr + ' --------\n\n')
  fd.close()
  print('\n--finished on '+datestr + ' --------\n)')
  return tvs, vvs, tts

if __name__ == "__main__":
  tf.app.run()



def look(name='0'):
    sess = tf.Session()
    #saver.restore(sess, FLAGS.save_path)
    new_saver = tf.train.import_meta_graph('lena\\model.ckpt-'+name+'.meta')
    new_saver.restore(sess, 'lena\\model.ckpt-'+name)
    all_vars = tf.global_variables()
    for v in all_vars:
      print(v.value())
    print(sess.run(all_vars[0]))
 
def rest():
  sess = tf.Session()
  #saver.restore(sess, FLAGS.save_path)
  new_saver = tf.train.import_meta_graph('lena\\model.ckpt-7811.meta')
  new_saver.restore(sess, 'lena\\model.ckpt-7811')
  ##new_saver = tf.train.import_meta_graph('C:\\users\\lena\\anaconda\\envs\\snakes\\lena\\model.ckpt-0.meta')
  ##new_saver.restore(sess, tf.train.latest_checkpoint('.\\lena'))
  all_vars = tf.trainable_variables()
  for v in all_vars:
    print(v.name, ' -- ',v,'---\n')
  val = v.eval(session=sess)
  print(val)
  return (val, all_vars)

def formermain():
  sess = tf.Session()
  #saver.restore(sess, FLAGS.save_path)
  new_saver = tf.train.import_meta_graph('lena\\model.ckpt-7811.meta')
  new_saver.restore(sess, 'lena\\model.ckpt-7811') #need directory without a file here? check!?
  ##new_saver = tf.train.import_meta_graph('C:\\users\\lena\\anaconda\\envs\\snakes\\lena\\model.ckpt-0.meta')
  ##new_saver.restore(sess, tf.train.latest_checkpoint('.\\lena'))
  all_vars = tf.trainable_variables()
  for v in all_vars:
    print(v,'---\n')
  print(v.value()[0])


