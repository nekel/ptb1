import os,sys,time
#import numpy as np, tensorflow as tf
#from importlib import reload
from datetime import datetime

#MAINdir= 'C:\\users\\Lena\\Anaconda\\envs\\snakes\\'
ABStr = ' abcdefghijklmnopqrstuvwxyz\'\n.,?!:;-'
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
  #input_file=WARPEACE
  # use ptb style embedding: hidden_size, vocab_size
  #model_dir=MAINdir+'ptb0_embed300_f1\\'
  logfile='test.01.31.17.txt'
