from transformers import BertTokenizer, BertForSequenceClassification, RobertaModel, \
        RobertaTokenizerFast, RobertaForSequenceClassification, RobertaTokenizer, AutoTokenizer, AutoModelForSequenceClassification, LongformerForSequenceClassification
from modelling_hat import HATModelForSequentialSentenceClassification
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
import pandas as pd
import os
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm


def train():
    model.train() 
    
    total_loss, total_accuracy = 0, 0
    for step,batch in enumerate(tqdm(train_dataloader)):
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        preds = model(sent_id, mask)
        model.zero_grad()   
        loss = cross_entropy(preds.logits, labels)
        total_loss = total_loss + loss.item()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(train_dataloader)
    val_total_loss = 0
    model.eval()
    print("Evaluating the model")
    
    for step, batch in enumerate(tqdm(val_dataloader)):
        batch = [r.to(device) for r in batch]
        sent_id_val, mask_val, tda_val, labels_val = batch
        preds_val = model(sent_id_val, mask_val, tda_val)
        loss = cross_entropy(preds_val, labels_val)
        val_total_loss = val_total_loss + loss.item()
        avg_val_loss = val_total_loss/len(val_dataloader)
    
    return avg_loss, avg_val_loss

def test(dataloader):
	model.eval()
	total_loss, total_accuracy = 0, 0
	
	pred_list = []
	for step,batch in enumerate(tqdm(dataloader)):
		batch = [r.to(device) for r in batch]

		model.zero_grad()        
		sent_id, mask, labels = batch
		preds = model(sent_id, mask)
		pred_list.append(preds)
	pred_list = torch.cat(pred_list)
	pred_list = torch.argmax(pred_list, axis = 1)
	return pred_list


parser = argparse.ArgumentParser(description = 'Train/test transformer.')
parser.add_argument("--cuda", help="Cuda Device", required=True)
parser.add_argument("--batch_size", help="Batch Size", default=10, type=int)
parser.add_argument("--epochs", help="number of epochs", default=10, type=int)
parser.add_argument("--data_name", help = "data file name", required = True)
parser.add_argument("--input_dir", help = "directory of csv files", required = True)
parser.add_argument("--model_type", help = "which model to finetune", default='hat', choices=['hat', 'longformer'])
args = parser.parse_args()

input_dir = args.input_dir
data_name = args.data_name

train_subset = f"{input_dir}/{data_name}_train.csv"
test_subset = f"{input_dir}/{data_name}_test.csv"

df_train= pd.read_csv(train_subset)
df_test = pd.read_csv(test_subset)
# Train/val split is not random as wikipedia dataset is ordered -> Original document followed by 3 permutations
train_size = int(0.9 * len(df_train))
df_train, df_val = df_train.iloc[:train_size], df_train.iloc[train_size:]


df_train['expert_label'] = df_train['expert_label'].replace(1, 0).replace(3, 1)
df_val['expert_label'] = df_val['expert_label'].replace(1, 0).replace(3, 1)
df_test['expert_label'] = df_test['expert_label'].replace(1, 0).replace(3, 1)


os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
device = torch.device(f"cuda")

if(args.model_type == "hat"):
    model_path = tokenizer_path = "kiddothe2b/hierarchical-transformer-base-4096"
if(args.model_type == "longformer"):
    model_path = tokenizer_path = "allenai/longformer-base-4096"

MAX_LEN = 4096
model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, do_lower_case=False)

X_train, X_val, X_test, y_train, y_val, y_test = df_train['doc'].tolist(), df_val['doc'].tolist(), df_test['doc'].tolist(), \
    df_train['expert_label'].tolist(), df_val['expert_label'].tolist(), df_test['expert_label'].tolist()


if(args.model_type == "hat"):  
    X_train_tokenized = tokenizer(X_train, add_special_tokens=True, max_length=MAX_LEN, padding='max_length', truncation=True)
    X_val_tokenized = tokenizer(X_val, add_special_tokens=True, max_length=MAX_LEN, padding='max_length', truncation=True)
    X_test_tokenized = tokenizer(X_test, add_special_tokens=True, max_length=MAX_LEN, padding='max_length', truncation=True)
elif(args.model_type == "longformer"):
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=MAX_LEN)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=MAX_LEN)
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=MAX_LEN)

batch_size = args.batch_size
dataloaders = []
y_labels = []
for tokens, labels in [(X_train_tokenized, y_train), (X_val_tokenized, y_val), (X_test_tokenized, y_test)]:
    
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    y = torch.tensor(labels)
    data = TensorDataset(seq, mask, y)
    
    dataloader = DataLoader(data, batch_size=batch_size)
    dataloaders.append(dataloader)
    y_labels.append(y)

# dataloaders and labels
train_dataloader, val_dataloader, test_dataloader = dataloaders
train_y, val_y, test_y = y_labels
    
num_classes = len(np.unique(train_y))
unique_classes = np.unique(train_y)

optimizer = torch.optim.Adam(model.parameters(), lr=(1e-5), eps=1e-8,weight_decay=0.05)
class_wts = compute_class_weight(class_weight = 'balanced', classes = unique_classes, y = train_y.tolist())

weights = torch.tensor(class_wts,dtype=torch.float)
weights = weights.to(device)

print("getting cross_entropy_loss")
cross_entropy  = nn.CrossEntropyLoss(weight=weights) 
epochs = args.epochs

train_losses=[]
val_losses = []
for epoch in range(epochs):
	print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
	train_loss, val_loss = train()
	train_losses.append(train_loss)
	val_losses.append(val_loss)
	preds = test(val_dataloader)
	preds_tr = test(train_dataloader)
	val_acc = (preds.cpu() == val_y).float().mean()
	train_acc = (preds_tr.cpu() == train_y).float().mean()
	print(f"Training Loss: {train_loss}, Validation Loss: {val_loss}, Validation acc: {val_acc}, Train acc: {train_acc}")


# get predictions for test data
with torch.no_grad():
	preds = test(test_dataloader)
	preds = preds.detach().cpu().numpy()
	test_y = test_y.numpy()
	print(list(preds))
	print(list(test_y))

	# model's performance
	print(classification_report(test_y, preds))
	print(preds)
	# confusion matrix
	print(pd.crosstab(test_y, preds))
	print(f"Len : {len(test_y)}")
	final_acc = (test_y == preds).sum() / len(test_y)
	print(f"Final Accuracy = {final_acc}")
