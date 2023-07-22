import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
from stats_count import *
import warnings
import argparse
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from glob import glob
import operator
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class CoherenceClassifier(nn.Module):
    def __init__(self):
        super(CoherenceClassifier, self).__init__()
        self.fc1 = nn.Linear(2500, 64)
        self.fc2 = nn.Linear(64, 3)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x



def feature_reduction_using_mutual_info(train, val, test, train_label, n_features=2500):
    train, val, test = np.array(train), np.array(val), np.array(test)

    mutual_info = mutual_info_classif(train, train_label)
    d = {index: value for index, value in enumerate(mutual_info)}
    sorted_d = dict( sorted(d.items(), key=operator.itemgetter(1),reverse=True))
    cols_included = list(sorted_d.keys())[:n_features]
    
    train = train[:, cols_included]
    val = val[:, cols_included]
    test = test[:, cols_included]
    return train, val, test


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
    feat_list = feat_list.reshape(feat_list.shape[0], -1)
    return feat_list

def scale_feat(train, test):
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test) 
    return train, test

warnings.filterwarnings('ignore')
seed = 42
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser(description = 'Train/test MLP on TDA features')
parser.add_argument("--input_dir", help="input directory of csv", required=True)
parser.add_argument("--feat_dir", help="input directory of TDA features", required=True)
parser.add_argument("--domain", help="Domain of GCDC split", required=True, choices=['clinton', 'yelp', 'enron', 'yahoo'])
parser.add_argument("--classifier_type", help="Type of classifier used", default='logreg', choices=['sklearn_mlp', 'logreg', 'torch_mlp'])
parser.add_argument("--cuda", help="GPU ID", default=1)

args = parser.parse_args()
print(args)

max_tokens_amount  = 256 # The number of tokens to which the tokenized text is truncated / padded.
n_layers = 12
model_name = "roberta-base"
layers_of_interest = [i for i in range(n_layers)]  # Layers for which attention matrices and features on them are 
                                             # calculated.

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
old_features_train, old_features_test = scale_feat(old_features_train, old_features_test)
ripser_train = load_feat(feat_dir, train_subset,
        f"_all_heads_{n_layers}_layers_MAX_LEN_{max_tokens_amount}_{model_name}_ripser",
        type="ripser")
ripser_test = load_feat(feat_dir, test_subset,
        f"_all_heads_{n_layers}_layers_MAX_LEN_{max_tokens_amount}_{model_name}_ripser",
        type="ripser")
ripser_train, ripser_test = scale_feat(ripser_train, ripser_test)
templ_train = load_feat(feat_dir, train_subset,
        f"_all_heads_{n_layers}_layers_MAX_LEN_{max_tokens_amount}_{model_name}_template",
        type="templ")
templ_test = load_feat(feat_dir, test_subset,
        f"_all_heads_{n_layers}_layers_MAX_LEN_{max_tokens_amount}_{model_name}_template",
        type="templ")
templ_train, templ_test = scale_feat(templ_train, templ_test)

"""## Loading data and features"""

train_data = pd.read_csv(f"{input_dir}/{train_subset}.csv")
test_data = pd.read_csv(f"{input_dir}/{test_subset}.csv")
y_test = list(map(int, test_data["expert_label"]))

X_train = []
for i in range(len(train_data)):
    features = np.concatenate((old_features_train[i],
                               ripser_train[i],
                               templ_train[i]))
    X_train.append(features)

train_data, val_data, X_train, X_val = train_test_split(train_data, X_train, test_size=0.1, random_state=seed)

y_train = train_data["expert_label"]
y_val = val_data["expert_label"]

X_test = []
for i in range(len(test_data)):
    features = np.concatenate((old_features_test[i],
                               ripser_test[i],
                               templ_test[i]))
    X_test.append(features)
y_test = test_data["expert_label"]

assert(len(train_data) == len(X_train))
assert(len(test_data) == len(X_test))

X_train, X_val, X_test = feature_reduction_using_mutual_info(X_train, X_val, X_test, y_train)

if args.classifier_type == "sklearn_mlp":
    # The classifier with concrete hyperparameters values, which you should insert here.
    ## Grid Search of hyperparameters. Use it on the dev/vaild set!

    acc_scores  = dict()
    y_val_pred_dict = dict()

    C_range = [0.0005, 0.001, 0.005, 0.01, 0.05]
    max_iter_range = [2, 3, 5, 10, 25]
    print(C_range, max_iter_range)

    solver = 'lbfgs'
    for C in tqdm(C_range):
        for max_iter in max_iter_range:
            classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=max_iter,activation = 'relu',solver=solver,random_state=seed, alpha = C)
            classifier.out_activation_ = 'softmax'
            classifier.fit(X_train, y_train)
            
            y_val_pred = classifier.predict(X_val)
            y_val_pred_dict[(C, max_iter)] = y_val_pred
            acc_scores[(C, max_iter)] = accuracy_score(y_val_pred, y_val)

    """### Prints the list of hyperparameters and corresponding accuracy, trained with these parameters"""

    for C in tqdm(C_range):
        for max_iter in max_iter_range:
            print("C: ", C, "| max iter:", max_iter, "| val accuracy :", acc_scores[(C, max_iter)])
    print("Best hyperparams:", max(acc_scores, key=acc_scores.get))
    best_C, best_max_iter = max(acc_scores, key=acc_scores.get)

    X_train, y_train = np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val])
    classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=best_max_iter,activation = 'relu',solver=solver,random_state=seed, alpha = best_C)
    classifier.out_activation_ = 'softmax'
elif args.classifier_type == "logreg":
    X_train, y_train = np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val])
    classifier = LogisticRegression(C=0.0005, max_iter=1000)
elif args.classifier_type == "torch_mlp":
    model = CoherenceClassifier().to(args.cuda)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    n_epochs = 10
    X_train, y_train = np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val])

    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=16, shuffle=True, num_workers=8)
    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=16, shuffle=True, num_workers=8)
    for epoch_idx in range(n_epochs):
        for batch in tqdm(train_loader):
            x, y = batch
            optimizer.zero_grad()
            x, y = torch.tensor(x, dtype=torch.float32).to(args.cuda), torch.tensor(y-1).to(args.cuda) # y-1 to make it work with torch
            pred_y = model(x)
            loss = loss_fn(pred_y, y)
            loss.backward()
            optimizer.step()

        preds_list = []
        for batch in test_loader:
            x, y = batch
            x, y = torch.tensor(x, dtype=torch.float32).to(args.cuda), torch.tensor(y-1).to(args.cuda) # y-1 to make it work with torch
            pred_y = model(x)
            pred_labels = pred_y.argmax(dim=1)
            preds_list.extend(pred_labels.tolist())
        preds_list = torch.tensor(preds_list)+1 # increment to bring it back in line with gcdc labels
        y_test_tensor = torch.tensor(np.array(y_test))
        acc = (preds_list == y_test_tensor).float().mean()
        print(f"Epoch - {epoch_idx} - Accuracy - {acc}")
    exit()

classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test_pred, y_test)

print("Test results (actual, predicted): ", list(zip(y_test_pred, y_test)))
print("Test accuracy is:", test_accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))