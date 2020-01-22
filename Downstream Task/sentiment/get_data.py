import pickle
import numpy
from nltk import word_tokenize

import pytreebank
dataset = pytreebank.load_sst()

data_dir = "data/"
name = "sentiment"



train_X = []
train_Y = []
val_X = []
val_Y = []
test_X = []
test_Y = []

for each_example in dataset['train']:
    if each_example.label in [0,1]:
        train_X.append(word_tokenize(each_example.to_lines()[0]))
        train_Y.append(0)
    elif each_example.label in [3,4]:
        train_X.append(word_tokenize(each_example.to_lines()[0]))
        train_Y.append(1)
    else:
        pass
    
    
for each_example in dataset['dev']:
    if each_example.label in [0,1]:
        val_X.append(word_tokenize(each_example.to_lines()[0]))
        val_Y.append(0)
    elif each_example.label in [3,4]:
        val_X.append(word_tokenize(each_example.to_lines()[0]))
        val_Y.append(1)
    else:
        pass
    
    
for each_example in dataset['test']:
    if each_example.label in [0,1]:
        test_X.append(word_tokenize(each_example.to_lines()[0]))
        test_Y.append(0)
    elif each_example.label in [3,4]:
        test_X.append(word_tokenize(each_example.to_lines()[0]))
        test_Y.append(1)
    else:
        pass

print(len(train_X), len(train_Y),len(val_X),len(val_Y),len(test_X),len(test_Y))

# all_feature_data = [train_X,train_Y,val_X,val_Y,test_X,test_Y]
# print(len(all_feature_data))




pickle.dump(train_X, open(data_dir+name+"_train_X.p", 'wb'))
pickle.dump(train_Y, open(data_dir+name+"_train_y.p", 'wb'))
pickle.dump(val_X, open(data_dir+name+"_val_X.p", 'wb'))
pickle.dump(val_Y, open(data_dir+name+"_val_y.p", 'wb'))
pickle.dump(test_X, open(data_dir+name+"_test_X.p", 'wb'))
pickle.dump(test_Y, open(data_dir+name+"_test_y.p", 'wb'))

