import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import sys
import pickle
import gensim
from sklearn.metrics import accuracy_score

h = .02  # step size in the mesh
embedding_size = None
vectors = None

# python classify.py embedding_src num_classes x_train_pickle y_train_pickle x_val_pickle y_val_pickle x_test_pickle y_test_pickle

# python classify_discrim_attr.py ../embeddings/glove.42B.300d.txt 2 ./data/discrim_attr_train_X.p ./data/discrim_attr_train_y.p ./data/discrim_attr_val_X.p ./data/discrim_attr_val_y.p ./data/discrim_attr_test_X.p ./data/discrim_attr_test_y.p
def loadVectors():
    global embedding_size
    global vectors
    try:
        data = open(sys.argv[1],"r").readlines()
        vectors = {}
        for row in data:
            vals = row.split()
            word = vals[0]
            vals = np.array( [float(val) for val in vals[1:]] )
            vectors[word] = vals
        embedding_size = len(vals)
    except:
        vectors = gensim.models.KeyedVectors.load_word2vec_format(sys.argv[1], binary=True)
        embedding_size = vectors.vectors.shape[1]
    #print("embedding_size = ",embedding_size)
    return vectors


def getFeats(sentence):
    global vectors
    global embedding_size
    if False:
      ret = np.zeros(embedding_size)
      cnt = 0
      for word in sentence:
          if word in vectors:
              ret+=vectors[word]
              cnt+=1
      if cnt>0:
        ret/=cnt
      return ret
    else:
      ret = []
      cnt = 0
      for word in sentence:
          if word in vectors:
              ret.extend([ v for v in vectors[word]])
              cnt+=1
          else:
              ret.extend( [v for v in np.zeros(embedding_size)] )
      ret = np.array(ret)
      return ret

def getOneHot(vals, max_num):
    ret = np.zeros((vals.shape[0], max_num))
    for i,val in enumerate(vals):
        ret[i][val] = 1
    return ret

def trainAndTest(x_splits, y_splits, clf):
    clf.fit(x_splits[0], y_splits[0])
    train_score = clf.score(x_splits[0], y_splits[0])
    val_score = None
    if len(x_splits[1])>0:
        val_score = clf.score(x_splits[1], y_splits[1])
        #print "Val Score = ", val_score
    score = clf.score(x_splits[2], y_splits[2])
    #print "Test Score = ", score
    return train_score,val_score,score

def getSimilarity(e1, e2):
	# cosine similarity
	return np.sum(e1 * e2)/( np.sqrt(np.sum(e1*e1)) * np.sqrt(np.sum(e2*e2)))

def getSimilarityScoreForWords(w1,w2):
	global embeddings
	#print w1
	#print embeddings
	if (w2 not in embeddings) or (w1 not in embeddings) :
		return -1
	finalVector_w1 = embeddings[w1]
	finalVector_w2 = embeddings[w2]
	return getSimilarity(finalVector_w1, finalVector_w2)

def f1_score(evaluation):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in evaluation:
            if i[0] == i[1] and i[1] == 1: 
                tp = tp+1
            if i[0] == i[1] and i[1] == 0: 
                tn = tn+1
            elif i[0] != i[1] and i[1] == 1: 
                fp = fp+1
            elif i[0] != i[1] and i[1] == 0: 
                fn = fn+1
    f1_positives = 0.0
    f1_negatives = 0.0
    if tp>0:
        precision=float(tp)/(tp+fp)
        recall=float(tp)/(tp+fn)
        f1_positives = 2*((precision*recall)/(precision+recall))
    if tn>0:
        precision=float(tn)/(tn+fn)
        recall=float(tn)/(tn+fp)
        f1_negatives = 2*((precision*recall)/(precision+recall))
    if f1_positives and f1_negatives:
        f1_average = (f1_positives+f1_negatives)/2.0
        return f1_average
    else:
        return 0


def main1():
    loadVectors()
    # num_classes = int(sys.argv[2])
    #print "num_classes = ",num_classes

    classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(kernel="linear", C=0.1),
    SVC(kernel="linear", C=1.0),
    SVC(gamma=2, C=1),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    RandomForestClassifier(max_depth=5, n_estimators=50, max_features=10),
    MLPClassifier(alpha=1),
    RandomForestClassifier(n_estimators=20, max_features=10)]



    all_feats = []
    labels = []
    idx = 3
    while idx<8:
        texts = pickle.load( open(sys.argv[idx],"rb") )
        if len(texts)>0:
            feats = np.array( [getFeats(t) for t in texts] )
            #print "feats : ",feats.shape
            all_feats.append( feats )
            idx+=1
            cur_labels = np.array(pickle.load( open(sys.argv[idx],"rb") ) )
            #cur_labels = getOneHot(cur_labels, max(cur_labels)+1)
            labels.append( cur_labels )
            #print "cur_labels : ",cur_labels.shape
            idx+=1
        else:
            idx+=2
            labels.append([])
            all_feats.append([])
    #print("Done loading data")

    best_test = 0.0
    best_clf = None
    best = 0.0
    for clf in classifiers:
        #print "="*33
        #print "clf = ",clf
        score, val_score, test_score = trainAndTest(all_feats, labels, clf)
        best_test = max(best_test, test_score)
        if score>best:
          best = score
          best_clf = clf
    #print("best_test for this split= ", best_test)
    #print "best_test = ", best_test
    #print("best= ", best, best_clf)


def main2():
    loadVectors()
    # num_classes = int(sys.argv[2])
    #print "num_classes = ",num_classes
    # global vectors
    classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(kernel="linear", C=0.1),
    SVC(kernel="linear", C=1.0),
    SVC(gamma=2, C=1),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    RandomForestClassifier(max_depth=5, n_estimators=50, max_features=10),
    MLPClassifier(alpha=1),
    RandomForestClassifier(n_estimators=20, max_features=10)]

    texts = pickle.load( open(sys.argv[7],"rb") )
    
    true_labels = pickle.load( open(sys.argv[8],"rb") )
    final_labels = []
    acc_true = []
    acc_pred = []
    for line_counter, each_text in enumerate(texts):
        word1 = each_text[0]
        word2 = each_text[1]
        word3 = each_text[2]
        #print(each_text)
        if word1 not in vectors or word2 not in vectors or word3 not in vectors:
            continue
        if getSimilarity(vectors[word1], vectors[word3])> getSimilarity(vectors[word2], vectors[word3]):
            final_labels.append([int(true_labels[line_counter]), 1])
            acc_true.append(int(true_labels[line_counter]))
            acc_pred.append(1)
        else:
            final_labels.append([int(true_labels[line_counter]), 0])
            acc_true.append(int(true_labels[line_counter]))
            acc_pred.append(0)
    
    #print(f1_score(final_labels))
    print(accuracy_score(acc_true, acc_pred))
        


# main1()
main2()
