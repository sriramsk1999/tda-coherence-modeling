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


cols_included = []
def feature_reduction(data, label, train):

    data = np.array(data)
    global cols_included

    if(train==True):
        mutual_info = mutual_info_classif(data, label)
        d = {index: value for index, value in enumerate(mutual_info)}
        sorted_d = dict( sorted(d.items(), key=operator.itemgetter(1),reverse=True))
        print(list(sorted_d.keys()))
        cols_included = list(sorted_d.keys())[:args.feature_reduction]
    
    #print(cols_included)
    print(data.shape)


    final_tda = np.array([])
    for i in range(len(cols_included)):
        #print(final_tda.shape)
        col = data[:,cols_included[i]]
        col = np.reshape(col,(col.shape[0],-1))
        if(i==0):
            final_tda = col
        else:
            final_tda = np.concatenate((final_tda,col),axis=1)
    
    return final_tda

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


def document_level_accuracy_score(preds, outs, doc_ids):
    df = pd.DataFrame({'doc_id':doc_ids, 'pred':preds, 'out':outs})
    # Group rows by doc_id to get document level predictions
    df = df.groupby(by='doc_id')
    # Document level predictions and outputs
    doc_pred = []
    doc_out = []
    for group in df: # (doc_id, group_dataframe)
        # Compute Boolean OR over group, if one or more 1 (intrusion predicted), then the document level prediction is also 1.
        doc_pred.append(reduce(lambda a,b: a or b, group[1]['pred']))
        doc_out.append(reduce(lambda a,b: a or b, group[1]['out']))
    acc = accuracy_score(doc_out, doc_pred)
    return acc


def accuracy_score_func(result, y_val,Val):

    acc = 0
    thresh1 = args.thresh1
    thresh2 = args.thresh2

    final_label = []
    for i in range(len(result)):
        if(result[i]<=thresh1):
            final_label.append(1)
        elif(result[i]>thresh1 and result[i]<=thresh2):
            final_label.append(2)
        elif(result[i]>thresh2):
            final_label.append(3)
    
    print(final_label)
    acc = accuracy_score(final_label, y_val)
    if(Val==False):
        print(classification_report(final_label, y_val))
    
    return acc


def build_model_using_sequential():
  model = Sequential([
    Dense(1024, kernel_initializer='normal', activation='relu'),
    Dropout(0.2),
    Dense(768, kernel_initializer='normal', activation='relu'),
    Dropout(0.2),
    Dense(256, kernel_initializer='normal', activation='relu'),
    Dense(1, kernel_initializer='normal', activation='relu')
  ])
  return model


warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

parser = argparse.ArgumentParser(description = 'File 3')
parser.add_argument("--IO_dir", help="I/O dir", required=True)
parser.add_argument("--train", help="Train file", required=True)
parser.add_argument("--test", help="Test file", required=True)
parser.add_argument("--type", help="Intrusion or 3-way", required=True)
parser.add_argument("--thresh1", help="Upper bound for low coherence", type=float)
parser.add_argument("--thresh2", help="Upper bound for medium coherence", type=float)
parser.add_argument("--feature_reduction", help="number of features after dimensionality reduction", default=2500, type=int)

args = parser.parse_args()
print(args)

max_examples_to_train = 10**10
max_tokens_amount  = 256 # The number of tokens to which the tokenized text is truncated / padded.
layers_of_interest = [i for i in range(12)]  # Layers for which attention matrices and features on them are 
                                             # calculated. For calculating features on all layers, leave it be
                                             # [i for i in range(12)].

train_subset = args.train
test_subset  = args.test  # dev/valid - for hyperparameters tuning;
                      # test - for final testing after tuning hyperparameters on the dev set.
input_dir = args.IO_dir  # Name of the directory with .csv file
type = args.type


model_path = "bert-base-cased"
# You can use either standard or fine-tuned BERT. If you want to use fine-tuned BERT to your current task, save the
# model and the tokenizer with the commands tokenizer.save_pretrained(output_dir); 
# bert_classifier.save_pretrained(output_dir) into the same directory and insert the path to it here.

old_f_train_file  = input_dir + "features/" + train_subset + \
                    "_all_heads_12_layers_s_e_v_c_b0b1_lists_array_6_thrs_MAX_LEN_256_bert-base-cased.npy"
old_f_test_file   = input_dir + "features/" + test_subset + \
                    "_all_heads_12_layers_s_e_v_c_b0b1_lists_array_6_thrs_MAX_LEN_256_bert-base-cased.npy"
ripser_train_file = input_dir + "features/" + train_subset + \
                    "_all_heads_12_layers_MAX_LEN_256_bert-base-cased_ripser.npy"
ripser_test_file = input_dir + "features/" + test_subset + \
                    "_all_heads_12_layers_MAX_LEN_256_bert-base-cased_ripser.npy"
templ_train_file  = input_dir + "features/" + train_subset + \
                    "_all_heads_12_layers_MAX_LEN_256_bert-base-cased_template.npy"
templ_test_file   = input_dir + "features/" + test_subset + \
                    "_all_heads_12_layers_MAX_LEN_256_bert-base-cased_template.npy"

old_features_train = np.load(old_f_train_file, allow_pickle=True)[:,:,:,:max_examples_to_train,:]
old_features_test  = np.load(old_f_test_file, allow_pickle=True)[:,:,:,:max_examples_to_train,:]

ripser_train = np.load(ripser_train_file, allow_pickle=True)[:,:,:max_examples_to_train,:]
ripser_test  = np.load(ripser_test_file, allow_pickle=True)[:,:,:max_examples_to_train,:]

templ_train = np.load(templ_train_file, allow_pickle=True)[:,:,:,:max_examples_to_train]
templ_test  = np.load(templ_test_file, allow_pickle=True)[:,:,:,:max_examples_to_train]

print(old_features_train.shape)
print(ripser_train.shape)
print(templ_train.shape)

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


# testing for intrusion task
if(type == "intrusion"):
    classifier = linear_model.LogisticRegression(solver=solver)
    for C in tqdm(C_range):
        for max_iter in max_iter_range:
            #classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=max_iter,activation = 'relu',solver=solver,random_state=1, alpha = C)
            classifier = linear_model.LogisticRegression(penalty='l2', C=C, max_iter=max_iter, dual=is_dual,
                                                        solver=solver)

            result, train_f1 = pred_by_Xy(X_train, y_train, X_test, classifier,type)
            results[(C, max_iter)] = result

            f1_scores[(C, max_iter)]  = f1_score(result, y_test)
            acc_scores[(C, max_iter)]  = document_level_accuracy_score(result, y_test, test_data['doc_id'])

    """### Prints the list of hyperparameters and corresponding matthews corcoef / accuracy of LogReg, trained with these parameters"""

    try:
        for C in tqdm(C_range):
            for max_iter in max_iter_range:
                print("C: ", C, "| max iter:", max_iter, "| f1 :", f1_scores[(C, max_iter)], "| accuracy :", acc_scores[(C, max_iter)])
        print("---")
        print("The best f1-score:", max(f1_scores.values()))
        print("The best accuracy-score:", max(acc_scores.values()))

    except:
        print("Data is not labeled")


# testing for 3Way classification task
elif(type == "3Way"):
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
    
