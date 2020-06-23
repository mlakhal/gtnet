import os
import json

def makedir(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def load_json(dict_path):
  with open(dict_path) as f:
    dict_data = json.load(f)

  return dict_data
