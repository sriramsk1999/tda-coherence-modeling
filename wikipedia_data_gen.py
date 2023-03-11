import json
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import random 
import pandas as pd
import string
import numpy as np
import argparse
import itertools
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def truncate_doc(doc, n_sent):
    ''' Truncate a document to n_sent sentences '''
    txt = sent_tokenize(doc['text'])
    trunc_txt = ' '.join(txt[:n_sent])
    doc['text'] = trunc_txt
    return doc

def global_jumbling(type, perms, docs, output_dir):
    ''' Shuffle complete document to create non-coherent samples. '''
    doc_ids, texts, expert_label = [], [], []

    id = 1
    for i in range(len(docs)):
        document = docs[i]
        doc_ids.append(id)
        texts.append(document['text'])
        expert_label.append(3)
        id+=1

        # Maintain set to prevent duplicates being created
        doc_set = set()
        doc_set.add(document['text'])

        sents_list = sent_tokenize(document['text'])
        for j in range(perms):
            while True:
                random.shuffle(sents_list)
                text_shuffled = ' '.join(sents_list)
                if text_shuffled not in doc_set:
                    doc_set.add(text_shuffled)
                    doc_ids.append(id)
                    texts.append(text_shuffled)
                    expert_label.append(1)
                    id+=1
                    break
    df = pd.DataFrame(list(zip(doc_ids, texts, expert_label)), columns =['doc_id', 'doc', 'expert_label'])
    df.to_csv(f"{output_dir}/wikpedia_global_{type}.csv")

def local_jumbling(type, perms, docs, output_dir, para_size):
    ''' Shuffle paragraphs of  document to create non-coherent samples. '''
    doc_ids, texts, expert_label = [], [], []

    id = 1
    for i in range(len(docs)):
        document = docs[i]
        doc_ids.append(id)
        texts.append(document['text'])
        expert_label.append(3)
        id+=1

        # Maintain set to prevent duplicates being created
        doc_set = set()
        doc_set.add(document['text'])

        sents_list = sent_tokenize(document['text'])
        para_list = np.array_split(sents_list, len(sents_list) // para_size)
        for j in range(perms):
            while True:
                for para in para_list: random.shuffle(para)
                shuffled_sent_list = list(itertools.chain.from_iterable(para_list))
                text_shuffled = ' '.join(shuffled_sent_list)

                if text_shuffled not in doc_set:
                    doc_set.add(text_shuffled)
                    doc_ids.append(id)
                    texts.append(text_shuffled)
                    expert_label.append(1)
                    id+=1
                    break
    df = pd.DataFrame(list(zip(doc_ids, texts, expert_label)), columns =['doc_id', 'doc', 'expert_label'])
    df.to_csv(f"{output_dir}/wikpedia_local_{type}.csv")

nltk.download('punkt')

parser = argparse.ArgumentParser(description = 'Generate train/test csvs for wikipedia.')
parser.add_argument("--seed", help="Random seed", default=42)
parser.add_argument("--n_perms", help="Number of permuted documents per document", default=3)
parser.add_argument("--output_dir", help="Output dir", default="wikipedia_data/wikipedia_dataset/")
args = parser.parse_args()

random.seed(args.seed)
n_docs = 1500
n_train, n_test = 1000, 500 
n_sent = 40 # min/max number of sentences in doc
para_size = 5 # Size of paragraph for local jumbling

os.makedirs(args.output_dir, exist_ok=True)

# Load wikipedia dataset, subsample, split into train/test
wiki_data = load_dataset("wikipedia", "20220301.en", split="train").shuffle(seed=args.seed)
# Filter out documents < n_sent sentences
wiki_data = wiki_data.filter(lambda doc: len(sent_tokenize(doc['text'])) >= n_sent)
# Truncate documents > n_sent sentences
wiki_data = wiki_data.map(truncate_doc, fn_kwargs={'n_sent': n_sent})

# Extract subset, split into train/test
wiki_data_subset = wiki_data.select(range(n_docs))
wiki_data_subset_split = wiki_data_subset.train_test_split(test_size=n_test, seed=args.seed)
wiki_train, wiki_test = wiki_data_subset_split['train'], wiki_data_subset_split['test']

print("Global jumbling")
global_jumbling("train", args.n_perms, wiki_train, args.output_dir)
global_jumbling("test", args.n_perms, wiki_test, args.output_dir)
print("Local jumbling")
local_jumbling("train", args.n_perms, wiki_train, args.output_dir, para_size)
local_jumbling("test", args.n_perms, wiki_test, args.output_dir, para_size)
