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


def load_feat(input_dir, subset, filepattern):
    feat_list = []
    n_feat = len(glob(f"{input_dir}/{subset}{filepattern}*"))

    for i in range(1, n_feat+1):
        feat = np.load(f"{input_dir}/{subset}{filepattern}_{i}_of_{n_feat}.npy")
        feat_list.append(feat)
    feat_list = np.concatenate(feat_list, axis=3)
    if len(feat_list.shape) == 4:
        feat_list = feat_list.transpose(3,0,1,2)
    elif len(feat_list.shape) == 5:
        feat_list = feat_list.transpose(3,0,1,2,4)
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
np.random.seed(42)
random.seed(42)

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

train_subset = f"{args.domain}_train"
test_subset  = f"{args.domain}_test"
input_dir = args.input_dir  # Name of the directory with .csv file
feat_dir = args.feat_dir


old_features_train = load_feat(feat_dir, train_subset,
        f"_all_heads_{n_layers}_layers_s_e_v_c_b0b1_lists_array_6_thrs_MAX_LEN_{max_tokens_amount}_{model_name}")
old_features_test = load_feat(feat_dir, test_subset,
        f"_all_heads_{n_layers}_layers_s_e_v_c_b0b1_lists_array_6_thrs_MAX_LEN_{max_tokens_amount}_{model_name}")
#ripser_train = load_feat(feat_dir, train_subset,
#        f"_all_heads_{n_layers}_layers_MAX_LEN_{max_tokens_amount}_{model_name}_ripser")
#ripser_test = load_feat(feat_dir, test_subset,
#        f"_all_heads_{n_layers}_layers_MAX_LEN_{max_tokens_amount}_{model_name}_ripser")
templ_train = load_feat(feat_dir, train_subset,
        f"_all_heads_{n_layers}_layers_MAX_LEN_{max_tokens_amount}_{model_name}_template")
templ_test = load_feat(feat_dir, test_subset,
        f"_all_heads_{n_layers}_layers_MAX_LEN_{max_tokens_amount}_{model_name}_template")

breakpoint()

solver  = "lbfgs"
is_dual = False
classifier = linear_model.LogisticRegression(solver=solver)

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
# print(train_data)



train_data = train_data[:max_examples_to_train]


X_train = []
for i in range(len(train_data)):
    features = np.concatenate((old_features_train[:,:,:,i,:].flatten(),
                               ripser_train[:,:,i,:].flatten(),
                               templ_train[:,:,:,i].flatten()))
    X_train.append(features)

train_data, val_data, X_train, X_val = train_test_split(train_data, X_train, test_size=0.1, random_state=42)

y_train = train_data["labels"]
y_val = val_data["labels"]

X_test = []
for i in range(len(test_data)):
    features = np.concatenate((old_features_test[:,:,:,i,:].flatten(),
                               ripser_test[:,:,i,:].flatten(),
                               templ_test[:,:,:,i].flatten()))
    X_test.append(features)
y_test = test_data["labels"]

X_train = X_train[:max_examples_to_train]
train_data = train_data[:max_examples_to_train]



try:
    assert(len(train_data) == len(X_train))
    assert(len(test_data) == len(X_test))
except:
    print("ASSERTION ERROR!!!")


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

# classifier = GaussianNB()
# C_range = [0.0005, 0.001, 0.01, 0.05, 1]
# max_iter_range = [10, 25, 50, 100, 500]
# C_range = [0.0005]
# max_iter_range = [100]
# x_train_np = np.array(X_train)
# y_train_np = np.array(y_train)

# pca_train = PCA(n_components=200)
# pca_train.fit(X_train)
# X_train = pca_train.transform(X_train)

# pca_test = PCA(n_components=200)
# pca_test.fit(X_test)
# X_test = pca_test.transform(X_test)



# print(np.array(X_train).shape)
# pca = PCA().fit(X_train)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.savefig("dummy_name.png")
# plt.show()
C_range = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
max_iter_range = [1, 2, 3, 5, 10, 25, 50, 100]

# C_range = [1]
# max_iter_range = [1]

# model = build_model_using_sequential()
# model.compile(loss='binary_crossentropy', optimizer='adam')
# X_train = StandardScaler().fit_transform(X_train)
# X_val = StandardScaler().fit_transform(X_val)

s = StandardScaler()
X_train = s.fit_transform(X_train)
X_val = s.transform(X_val)
X_test = s.transform(X_test)

print(f"Dataset Shape : X -> {np.array(X_train).shape} ")
for C in tqdm(C_range):
    for max_iter in max_iter_range:
        classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=max_iter,activation = 'relu',solver=solver,random_state=1, alpha = C)
        classifier.out_activation_ = 'softmax'
        # classifier = GaussianNB()
        # classifier = linear_model.LogisticRegression(penalty='l2', C=C, max_iter=max_iter, dual=is_dual, solver=solver)
        #classifier = linear_model.LinearRegression()
        #model.fit(X_train, y_train, epochs=10, verbose=0)
        #classifier = SVR(C=C, epsilon=0.2, max_iter=max_iter)
        #classifier = MLPRegressor(hidden_layer_sizes=(256,128,64),activation="relu" ,random_state=1, max_iter=max_iter)
        #classifier = DecisionTreeClassifier(random_state=0)

        # classifier = SVC(C=C,max_iter=max_iter, kernel = 'rbf' )
        result= pred_by_Xy(X_train, y_train, X_val, classifier,type)
        #print(result)
        # results = permutation_importance(classifier, X_val, y_val, scoring='accuracy')
        # # get importance
        # importance = results.importances_mean
        # # summarize feature importance
        # for i,v in enumerate(importance):
        #     print('Feature: %0d, Score: %.5f' % (i,v))
        

        #result = model.predict(X_val)
        results_3Way[(C, max_iter)] = result
        #print(np.unique(y_val))
        acc_scores_3Way[(C, max_iter)]  = accuracy_score(result, y_val)

"""### Prints the list of hyperparameters and corresponding matthews corcoef / accuracy of LogReg, trained with these parameters"""

for C in tqdm(C_range):
    for max_iter in max_iter_range:
        print("C: ", C, "| max iter:", max_iter, "| val accuracy :", acc_scores_3Way[(C, max_iter)])
print("---")
print("The best accuracy-score:", max(acc_scores_3Way.values()))
print("Best hyperparams:", max(acc_scores_3Way, key=acc_scores_3Way.get))
best_C, best_max_iter = max(acc_scores_3Way, key=acc_scores_3Way.get)

# classifier = linear_model.LogisticRegression(penalty='l2', C=best_C, max_iter=best_max_iter, dual=is_dual, solver=solver)

# classifier = SVC(C=best_C,max_iter=best_max_iter, kernel = 'rbf' )
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
save_path_acc = "Final_Results/Old_Features/MLP/Accuracy/enron.npy"
save_path_cm = "Final_Results/Old_Features/MLP/Confusion_Matrix/enron.npy"
np.save(save_path_acc, final_accuracy)
np.save(save_path_cm, cm)

print(np.load(save_path_acc))
print(np.load(save_path_cm))


# coef_filename = 'linear_coef.npy'
# print(f"Saving coefficients at {coef_filename}")
# np.save(coef_filename, classifier.coef_)

# best_classifier_model = linear_model.LogisticRegression(multi_class='multinomial', penalty='l2', C=0.0005, max_iter=1000, dual=is_dual,
#                                                     solver=solver)
# # best_proba = best_classifier_model.predict_proba(X_test)

# best_result, best_acc, best_proba = pred_by_Xy(X_train, y_train, X_test, best_classifier_model, type)
# # best_proba = best_classifier_model.predict_log_proba(X_test)
# print(best_classifier_model.classes_)
# print("\n--------------------------------------------\n")
# # print(len(best_result))
# # print(len(y_test))
# for i in range(len(best_result)):
#     if(y_test[i] == 2 or True):
#         print(f'{y_test[i]} | {best_result[i]} || {best_proba[i]}')

