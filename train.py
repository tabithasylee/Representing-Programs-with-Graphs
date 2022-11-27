from utils import vocabulary_extractor
from model.model import Model
import yaml
import sys
from utils.arg_parser import parse_input_args
import os


def train(task_id):

  with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.Loader)

  checkpoint_path = cfg['checkpoint_path']
  train_path = cfg['train_path']
  val_path   = cfg['val_path']
  token_path = cfg['token_path']

  if True and os.path.isfile(token_path):
    vocabulary = vocabulary_extractor.create_vocabulary_from_corpus(train_path, token_path)
    print("Constructed vocabulary...")
  else:
    vocabulary = vocabulary_extractor.load_vocabulary(token_path)
    print("Loaded vocabulary...")
    

  m = Model(mode='train', task_id=task_id, vocabulary=vocabulary, checkpoint_path=checkpoint_path)
  n_train_epochs = 50

  m.train(train_path=train_path, val_path=val_path, n_epochs=n_train_epochs)
  print("Model trained successfully...")



args = sys.argv[1:]
task_id = parse_input_args(args)

train(task_id)

