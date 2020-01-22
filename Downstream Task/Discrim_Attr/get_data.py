import pickle
import numpy
from nltk import word_tokenize

data_dir = "data/"
name = "discrim_attr"


data_train_X = []
data_val_X = []
data_test_X = []

data_train_y = []
data_val_y = []
data_test_y = []


tmp = open('./train.txt').readlines()
for row in tmp:
  vals = row.strip().split(',')
  data_train_X.append(vals[:3])
  data_train_X.append(vals[3])

tmp = open('./validation.txt').readlines()
for row in tmp:
  vals = row.strip().split(',')
  data_val_X.append(vals[:3])
  data_val_y.append(vals[3])

tmp = open('./test.txt').readlines()
for row in tmp:
  vals = row.strip().split(',')
  data_test_X.append(vals[:3])
  data_test_y.append(vals[3])

pickle.dump(data_train_X, open(data_dir+name+"_train_X.p", 'wb'))
pickle.dump(data_train_y, open(data_dir+name+"_train_y.p", 'wb'))
pickle.dump(data_val_X, open(data_dir+name+"_val_X.p", 'wb'))
pickle.dump(data_val_y, open(data_dir+name+"_val_y.p", 'wb'))
pickle.dump(data_test_X, open(data_dir+name+"_test_X.p", 'wb'))
pickle.dump(data_test_y, open(data_dir+name+"_test_y.p", 'wb'))

