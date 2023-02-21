import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import linear_model, preprocessing
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import scatter
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from stats_count import *
import copy
import sys
import warnings
import argparse
from functools import reduce
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import operator
from sklearn.metrics import classification_report
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from glob import glob


def load_feat(input_dir, subset, filepattern, type):
    feat_list = []
    n_feat = len(glob(f"{input_dir}/{subset}{filepattern}*"))

    for i in range(1, n_feat+1):
        feat = np.load(f"{input_dir}/{subset}{filepattern}_{i}_of_{n_feat}.npy")
        feat_list.append(feat)
    if type == "old_f":
        feat_list = np.concatenate(feat_list, axis=3)
        feat_list = feat_list.transpose(3,0,1,2,4)
    elif type == "templ":
        feat_list = np.concatenate(feat_list, axis=3)
        feat_list = feat_list.transpose(3,0,1,2)
    elif type == "ripser":
        feat_list = np.concatenate(feat_list, axis=2)
        feat_list = feat_list.transpose(2,0,1,3)
    return feat_list

def pred_by_Xy(X_train, y_train, x_test, classifier,type, verbose=False, scale=True):

    if scale:
        scaler  = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)

    classifier.fit(X_train, y_train)
    
    if verbose:
        
        if(type == "intrusion"):
            print("f1-score:", f1_score(y_train, classifier.predict(X_train)))
        
        elif (type == "3Way"):
            print("Acc-score: ", accuracy_score(y_train, classifier.predict(X_train)))
    
    if scale:
        x_test = scaler.transform(x_test)
    
    if(type == "intrusion"):
        return classifier.predict(x_test), f1_score(y_train, classifier.predict(X_train))
    
    return classifier.predict(x_test)


warnings.filterwarnings('ignore')
seed = 42
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser(description = 'Train/test MLP on TDA features')
parser.add_argument("--input_dir", help="input directory of csv", required=True)
parser.add_argument("--feat_dir", help="input directory of TDA features", required=True)
parser.add_argument("--domain", help="Domain of GCDC split", required=True, choices=['clinton', 'yelp', 'enron', 'yahoo'])
parser.add_argument("--no-hat", help="Disable using HAT model.", dest='hat', action='store_false', default=True)

args = parser.parse_args()
print(args)

if args.hat:
    max_tokens_amount  = 4096 # The number of tokens to which the tokenized text is truncated / padded.
    n_layers = 4 # Only 4 cross segment encoder blocks
    model_name = "hierarchical-transformer-base-4096"
else:
    max_tokens_amount  = 256 # The number of tokens to which the tokenized text is truncated / padded.
    n_layers = 12
    model_name = "bert-base-cased"
layers_of_interest = [i for i in range(n_layers)]  # Layers for which attention matrices and features on them are 
                                             # calculated.
max_examples_to_train = 10**10

train_subset = f"{args.domain}_train"
test_subset  = f"{args.domain}_test"
input_dir = args.input_dir  # Name of the directory with .csv file
feat_dir = args.feat_dir


old_features_train = load_feat(feat_dir, train_subset,
        f"_all_heads_{n_layers}_layers_s_e_v_c_b0b1_lists_array_6_thrs_MAX_LEN_{max_tokens_amount}_{model_name}",
        type="old_f")
old_features_test = load_feat(feat_dir, test_subset,
        f"_all_heads_{n_layers}_layers_s_e_v_c_b0b1_lists_array_6_thrs_MAX_LEN_{max_tokens_amount}_{model_name}",
        type="old_f")
ripser_train = load_feat(feat_dir, train_subset,
        f"_all_heads_{n_layers}_layers_MAX_LEN_{max_tokens_amount}_{model_name}_ripser",
        type="ripser")
ripser_test = load_feat(feat_dir, test_subset,
        f"_all_heads_{n_layers}_layers_MAX_LEN_{max_tokens_amount}_{model_name}_ripser",
        type="ripser")
templ_train = load_feat(feat_dir, train_subset,
        f"_all_heads_{n_layers}_layers_MAX_LEN_{max_tokens_amount}_{model_name}_template",
        type="templ")
templ_test = load_feat(feat_dir, test_subset,
        f"_all_heads_{n_layers}_layers_MAX_LEN_{max_tokens_amount}_{model_name}_template",
        type="templ")

"""## Loading data and features"""

try:
    train_data = pd.read_csv(input_dir + train_subset + ".csv")
    test_data = pd.read_csv(input_dir + test_subset + ".csv")
except:
    train_data = pd.read_csv(input_dir + train_subset + ".tsv", delimiter="\t", header=None)
    train_data.columns = ["0", "labels", "2", "sentence"]
    test_data = pd.read_csv(input_dir + test_subset + ".tsv", delimiter="\t")

if "label" in train_data.columns:
    train_data["labels"] = (train_data["label"] == "Ordered").astype(int)
    test_data["labels"] = (test_data["label"] == "Ordered").astype(int)
elif "hasIntrusion" in train_data.columns:
    # Rename hasIntrusion to labels as expected by code
    train_data.rename(columns={'hasIntrusion':'labels'}, inplace=True)
    test_data.rename(columns={'hasIntrusion':'labels'}, inplace=True)
elif "expert_label" in train_data.columns:
    # Rename hasIntrusion to labels as expected by code
    train_data.rename(columns={'expert_label':'labels'}, inplace=True)
    test_data.rename(columns={'expert_label':'labels'}, inplace=True)

y_test = list(map(int, test_data["labels"]))
train_data = train_data[:max_examples_to_train]

X_train = []
for i in range(len(train_data)):
    features = np.concatenate((old_features_train[i].flatten(),
                               ripser_train[i].flatten(),
                               templ_train[i].flatten()))
    X_train.append(features)

train_data, val_data, X_train, X_val = train_test_split(train_data, X_train, test_size=0.1, random_state=seed)

y_train = train_data["labels"]
y_val = val_data["labels"]

X_test = []
for i in range(len(test_data)):
    features = np.concatenate((old_features_test[i].flatten(),
                               ripser_test[i].flatten(),
                               templ_test[i].flatten()))
    X_test.append(features)
y_test = test_data["labels"]

X_train = X_train[:max_examples_to_train]
train_data = train_data[:max_examples_to_train]


assert(len(train_data) == len(X_train))
assert(len(test_data) == len(X_test))


# The classifier with concrete hyperparameters values, which you should insert here.
# For grid search of hyperparameters - see below.

"""## Grid Search of hyperparameters. Use it on the dev/vaild set!

(**Reminder**: Don't tune hyperparameters on the test set, to not overfit hyperparameters. Tune hyperparameters on the dev/valid set, and then use the best ones on the test set.)
"""

C_range = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2]
max_iter_range = [1, 2, 3, 5, 10, 25, 50, 100, 500, 1000, 2000]
print(C_range, max_iter_range)

acc_scores  = dict()
f1_scores  = dict()
results     = dict()
acc_scores_3Way = dict()
results_3Way = dict()

C_range = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
max_iter_range = [1, 2, 3, 5, 10, 25, 50, 100]

s = StandardScaler()
X_train = s.fit_transform(X_train)
X_val = s.transform(X_val)
X_test = s.transform(X_test)

solver = 'lbfgs'

print(f"Dataset Shape : X -> {np.array(X_train).shape} ")
for C in tqdm(C_range):
    for max_iter in max_iter_range:
        classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=max_iter,activation = 'relu',solver=solver,random_state=seed, alpha = C)
        classifier.out_activation_ = 'softmax'

        result= pred_by_Xy(X_train, y_train, X_val, classifier,type)

        results_3Way[(C, max_iter)] = result
        acc_scores_3Way[(C, max_iter)]  = accuracy_score(result, y_val)

"""### Prints the list of hyperparameters and corresponding matthews corcoef / accuracy of LogReg, trained with these parameters"""

for C in tqdm(C_range):
    for max_iter in max_iter_range:
        print("C: ", C, "| max iter:", max_iter, "| val accuracy :", acc_scores_3Way[(C, max_iter)])
print("---")
print("The best accuracy-score:", max(acc_scores_3Way.values()))
print("Best hyperparams:", max(acc_scores_3Way, key=acc_scores_3Way.get))
best_C, best_max_iter = max(acc_scores_3Way, key=acc_scores_3Way.get)

classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=best_max_iter,activation = 'relu',solver=solver,random_state=1, alpha = best_C)
classifier.out_activation_ = 'softmax'
classifier.fit(X_train, y_train)
test_result = classifier.predict(X_test)
test_accuracy = accuracy_score(test_result, y_test)

train_result = classifier.predict(X_train)
train_accuracy = accuracy_score(train_result, y_train)

val_result = classifier.predict(X_val)
val_accuracy = accuracy_score(val_result, y_val )
print("Test results", result)
print("Test accuracy is:", test_accuracy)

final_accuracy = np.array([train_accuracy, val_accuracy, test_accuracy])
print(final_accuracy)

cm = confusion_matrix(y_test, test_result)
print(cm)

