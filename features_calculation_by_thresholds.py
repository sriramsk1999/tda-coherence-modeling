import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaModel, RobertaTokenizerFast, RobertaForSequenceClassification, RobertaTokenizer
import sys
from stats_count import *
from grab_weights import grab_attention_weights
import os
import warnings
from math import ceil
from multiprocessing import Pool
import argparse

warnings.filterwarnings('ignore')

def get_token_length(batch_texts, tokenizer=None, MAX_LEN=None, is_hat=False):
    print("Get token length")
    if is_hat:
        inputs = tokenizer(batch_texts,
                               add_special_tokens=True,
                               max_length=MAX_LEN,                # Max length to truncate/pad
                               padding='max_length',              # Pad sentence to max length)
                               truncation=True
                              )
        inputs = np.array(inputs['input_ids'])
    else:
        inputs = tokenizer.batch_encode_plus(batch_texts,
           return_tensors='pt',
           add_special_tokens=True,
           max_length=MAX_LEN,             # Max length to truncate/pad
           pad_to_max_length=True,         # Pad sentence to max length
           truncation=True
        )
        inputs = inputs['input_ids'].numpy()
    n_tokens = []
    indexes = np.argwhere(inputs == tokenizer.pad_token_id)
    
    for i in range(inputs.shape[0]):
        ids = indexes[(indexes == i)[:, 0]]
        if not len(ids):
            n_tokens.append(MAX_LEN)
        else:
            n_tokens.append(ids[0, 1])
    return n_tokens

def function_for_v(list_of_v_degrees_of_graph):
    return sum(map(lambda x: np.sqrt(x*x), list_of_v_degrees_of_graph))

def split_matricies_and_lengths(adj_matricies, ntokens_array, num_of_workers):
    splitted_adj_matricies = np.array_split(adj_matricies, num_of_workers)
    splitted_ntokens = np.array_split(ntokens_array, num_of_workers)
    assert all([len(m)==len(n) for m, n in zip(splitted_adj_matricies, splitted_ntokens)]), "Split is not valid!"
    return zip(splitted_adj_matricies, splitted_ntokens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'File 1')

    parser.add_argument("--cuda", help="Cuda Device", required=True)
    parser.add_argument("--data_name", help="Data Name", required=True)
    parser.add_argument("--IO_dir", help="I/O dir", required=True)
    parser.add_argument("--chunk_start_idx", help="Chunk start index", type=int, required=True)
    parser.add_argument("--chunk_size", help="Chunk size", type=int, default=10000)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=10)
    parser.add_argument("--dump_size", help="Dump size", type=int, default=100)

    args = parser.parse_args()
    print(args)

    cuda_device = args.cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

    np.random.seed(42) # For reproducibility.
    max_tokens_amount  = 256 # The number of tokens to which the tokenized text is truncated / padded.
    stats_cap          = 500 # Max value that the feature can take. Is NOT applicable to Betty numbers.

    layers_of_interest = [i for i in range(12)]  # Layers for which attention matrices and features on them are
                                                # calculated. For calculating features on all layers, leave it be
                                                # [i for i in range(12)].
    stats_name = "s_e_v_c_b0b1" # The set of topological features that will be count (see explanation below)

    thresholds_array = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75] # The set of thresholds
    thrs = len(thresholds_array)                           # ("t" in the paper)

    # model_path = tokenizer_path = "bert-base-cased"
    model_path = tokenizer_path = "roberta-base"

    # You can use either standard or fine-tuned BERT. If you want to use fine-tuned BERT to your current task, save the
    # model and the tokenizer with the commands tokenizer.save_pretrained(output_dir);
    # bert_classifier.save_pretrained(output_dir) into the same directory and insert the path to it here.


    # ### Explanation of stats_name parameter
    #
    # Currently, we implemented calculation of the following graphs features:
    # * "s"    - amount of strongly connected components
    # * "w"    - amount of weakly connected components
    # * "e"    - amount of edges
    # * "v"    - average vertex degree
    # * "c"    - amount of (directed) simple cycles
    # * "b0b1" - Betti numbers
    #
    # The variable stats_name contains a string with the names of the features, which you want to calculate. The format of the string is the following:
    #
    # "stat_name + "_" + stat_name + "_" + stat_name + ..."
    #
    # **For example**:
    #
    # `stats_name == "s_w"` means that the number of strongly and weakly connected components will be calculated
    #
    # `stats_name == "b0b1"` means that only the Betti numbers will be calculated
    #
    # `stats_name == "b0b1_c"` means that Betti numbers and the number of simple cycles will be calculated
    #
    # e.t.c.

    subset = args.data_name           # .csv file with the texts, for which we count topological features
    input_dir = args.IO_dir  # Name of the directory with .csv file
    output_dir = args.IO_dir # Name of the directory with calculations results

    prefix = output_dir + subset

    r_file = output_dir + 'attentions/' + subset  + "_all_heads_" + str(len(layers_of_interest)) + "_layers_MAX_LEN_" +              str(max_tokens_amount) + "_" + model_path.split("/")[-1]
    # Name of the file for attention matrices weights

    stats_file = output_dir + 'new_features/' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers_" + stats_name              + "_lists_array_" + str(thrs) + "_thrs_MAX_LEN_" + str(max_tokens_amount) +              "_" + model_path.split("/")[-1] + '.npy'
    # Name of the file for topological features array

    chunk_start_idx = args.chunk_start_idx
    chunk_size = args.chunk_size

    # .csv file must contain the column with the name **sentence** with the texts. It can also contain the column **labels**, which will be needed for testing. Any other arbitrary columns will be ignored.

    batch_size = args.batch_size # batch size
    DUMP_SIZE = args.dump_size # number of batches to be dumped

    num_of_workers = 10
    pool = Pool(num_of_workers)

    try:
        data = pd.read_csv(input_dir + subset + ".csv").reset_index(drop=True)
    except:
        #data = pd.read_csv(input_dir + subset + ".tsv", delimiter="\t")
        data = pd.read_csv(input_dir + subset + ".tsv", delimiter="\t", header=None)
        data.columns = ["0", "labels", "2", "sentence"]

    data = data.iloc[chunk_start_idx:min(chunk_start_idx+chunk_size, len(data))] # extract chunk of data

    sentences = data['sentence']
    print("Average amount of words in example:",       np.mean(list(map(len, map(lambda x: re.sub('\w', ' ', x).split(" "), data['sentence'])))))
    print("Max. amount of words in example:",       np.max(list(map(len, map(lambda x: re.sub('\w', ' ', x).split(" "), data['sentence'])))))
    print("Min. amount of words in example:",       np.min(list(map(len, map(lambda x: re.sub('\w', ' ', x).split(" "), data['sentence'])))))

    MAX_LEN = max_tokens_amount
    # tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
    data['tokenizer_length'] = get_token_length(data['sentence'].values)
    ntokens_array = data['tokenizer_length'].values

    # ## Attention extraction
    # Loading **BERT** and tokenizers using **transformers** library.

    number_of_batches = ceil(len(data['sentence']) / batch_size)
    batched_sentences = np.array_split(data['sentence'].values, number_of_batches)
    number_of_files = ceil(number_of_batches / DUMP_SIZE)
    adj_matricies = []
    adj_filenames = []
    assert number_of_batches == len(batched_sentences) # sanity check

    device='cuda'
    # model = BertForSequenceClassification.from_pretrained(model_path, output_attentions=True)
    # tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
    model = RobertaForSequenceClassification.from_pretrained(model_path, output_attentions=True)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
    tokenizer.do_lower_case = False
    model = model.to(device)
    MAX_LEN = max_tokens_amount

    for i in tqdm(range(number_of_batches), desc="Weights calc"):
        attention_w = grab_attention_weights(model, tokenizer, batched_sentences[i], max_tokens_amount, device)
        # sample X layer X head X n_token X n_token
        print("Appending")
        adj_matricies.append(attention_w)
        if (i+1) % DUMP_SIZE == 0: # dumping
            print(f'Saving: shape {adj_matricies[0].shape}')
            adj_matricies = np.concatenate(adj_matricies, axis=1)
            print("Concatenated")
            adj_matricies = np.swapaxes(adj_matricies, axis1=0, axis2=1) # sample X layer X head X n_token X n_token
            filename = r_file + "_part" + str(ceil(i/DUMP_SIZE)) + "of" + str(number_of_files) + '.npy'
            print(f"Saving weights to : {filename}")
            adj_filenames.append(filename)
            np.save(filename, adj_matricies)
            print("Saved !!")
            adj_matricies = []

    if len(adj_matricies):
        filename = r_file + "_part" + str(ceil(i/DUMP_SIZE)) + "of" + str(number_of_files) + '.npy'
        print(f'Saving: shape {adj_matricies[0].shape}')
        adj_matricies = np.concatenate(adj_matricies, axis=1)
        print("Concatenated")
        adj_matricies = np.swapaxes(adj_matricies, axis1=0, axis2=1) # sample X layer X head X n_token X n_token
        print(f"Saving weights to : {filename}")
        np.save(filename, adj_matricies)

    print("Results saved.")

    # ## Calculating topological features

    stats_tuple_lists_array = []
    for i, filename in enumerate(tqdm(adj_filenames, desc='Calculate features from attention maps')):
        adj_matricies = np.load(filename, allow_pickle=True)
        ntokens = ntokens_array[i*batch_size*DUMP_SIZE : (i+1)*batch_size*DUMP_SIZE]
        splitted = split_matricies_and_lengths(adj_matricies, ntokens, num_of_workers)
        args = [(m, thresholds_array, ntokens, stats_name.split("_"), stats_cap) for m, ntokens in splitted]
        stats_tuple_lists_array_part = pool.starmap(

            count_top_stats, args
        )
        stats_tuple_lists_array.append(np.concatenate([_ for _ in stats_tuple_lists_array_part], axis=3))

    stats_tuple_lists_array = np.concatenate(stats_tuple_lists_array, axis=3)
    np.save(stats_file, stats_tuple_lists_array)

    # ##### Checking the size of features matrices:
    #
    # Layers amount **Х** Heads amount **Х** Features amount **X** Examples amount **Х** Thresholds amount
    #
    # **For example**:
    #
    # `stats_name == "s_w"` => `Features amount == 2`
    #
    # `stats_name == "b0b1"` => `Features amount == 2`
    #
    # `stats_name == "b0b1_c"` => `Features amount == 3`
    #
    # e.t.c.
    #
    # `thresholds_array == [0.025, 0.05, 0.1, 0.25, 0.5, 0.75]` => `Thresholds amount == 6`

