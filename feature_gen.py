import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, RobertaModel, \
        RobertaTokenizerFast, RobertaForSequenceClassification, RobertaTokenizer, AutoTokenizer
import transformers
import sys
import os
import warnings
from math import ceil
from multiprocessing import Pool, Queue, Process
import argparse
from collections import defaultdict
import json
import time
import itertools

from stats_count import *
import ripser_count
from grab_weights import grab_attention_weights, text_preprocessing, grab_hat_attention_weights

from features_calculation_by_thresholds import get_token_length, function_for_v, split_matricies_and_lengths
from features_calculation_ripser_and_templates import attention_to_self, attention_to_next_token, attention_to_prev_token, \
    attention_to_beginning, attention_to_ids, count_template_features, calculate_features_t, get_list_of_ids, reformat_barcodes, \
    subprocess_wrap, get_only_barcodes, format_barcodes, save_barcodes, unite_barcodes, matrix_distance

from modelling_hat import HATModelForSequentialSentenceClassification


parser = argparse.ArgumentParser(description = 'End to end TDA feature generation (parallelized)')

parser.add_argument("--cuda", help="Cuda Device", required=True)
parser.add_argument("--data_name", help="Data Name", required=True)
parser.add_argument("--input_dir", help="input dir", required=True)
parser.add_argument("--output_dir", help="output dir", required=True)
parser.add_argument("--batch_size", help="Batch size", type=int, default=10)
parser.add_argument("--no-hat", help="Disable using HAT model.", dest='hat', action='store_false', default=True)

args = parser.parse_args()
print(args)

cuda_device = args.cuda
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

transformers.logging.set_verbosity_error()
np.random.seed(42) # For reproducibility.
thresholds_array = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75] # The set of thresholds
thrs = len(thresholds_array)                           # ("t" in the paper)

stats_cap          = 500 # Max value that the feature can take. Is NOT applicable to Betty numbers.
stats_name = "s_e_v_c_b0b1" # The set of topological features that will be count (see explanation below)

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

if args.hat:
    max_tokens_amount  = 4096 # The number of tokens to which the tokenized text is truncated / padded.
    n_layers = 4 # Only 4 cross segment encoder blocks
    model_path = tokenizer_path = "kiddothe2b/hierarchical-transformer-base-4096"
else:
    max_tokens_amount  = 256 # The number of tokens to which the tokenized text is truncated / padded.
    n_layers = 12
    model_path = tokenizer_path = "roberta-base"

layers_of_interest = [i for i in range(n_layers)]  # Layers for which attention matrices and features on them are
                                             # calculated. For calculating features on all layers, leave it be
                                             # [i for i in range(12)].
subset = args.data_name  # .csv file with the texts, for which we count topological features
input_dir = args.input_dir  # Name of the directory with .csv file
output_dir = args.output_dir # Name of the directory with calculations results
os.makedirs(output_dir, exist_ok=True)

# .csv file must contain the column with the name **sentence** with the texts. It can also contain the column **labels**, which will be needed for testing. Any other arbitrary columns will be ignored.

batch_size = args.batch_size

data = pd.read_csv(input_dir + subset + ".csv").reset_index(drop=True)
data['sentence'] = data['doc']

sentences = data['sentence']
print("Average amount of words in example:",       np.mean(list(map(len, map(lambda x: re.sub('\w', ' ', x).split(" "), data['sentence'])))))
print("Max. amount of words in example:",       np.max(list(map(len, map(lambda x: re.sub('\w', ' ', x).split(" "), data['sentence'])))))
print("Min. amount of words in example:",       np.min(list(map(len, map(lambda x: re.sub('\w', ' ', x).split(" "), data['sentence'])))))

MAX_LEN = max_tokens_amount
if args.hat:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, do_lower_casse=False)
else:
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
data['tokenizer_length'] = get_token_length(data['sentence'].values, tokenizer, MAX_LEN, args.hat)
ntokens_array = data['tokenizer_length'].values

number_of_batches = ceil(len(data['sentence']) / batch_size)
batched_sentences = np.array_split(data['sentence'].values, number_of_batches)
adj_matricies = []
assert number_of_batches == len(batched_sentences) # sanity check
Q = Queue()

def get_attention_matrices(model_path, tokenizer, batch, MAX_LEN, is_hat):
    if is_hat:
        model = HATModelForSequentialSentenceClassification.from_pretrained(model_path, trust_remote_code=True, output_attentions=True)
    else:
        model = RobertaForSequenceClassification.from_pretrained(model_path, output_attentions=True)
    model = model.to('cuda')

    minibatch_size = 25
    n_minibatches = ceil(len(batch) / minibatch_size)
    minibatch = np.array_split(batch, n_minibatches)
    adj_matricies = []
    for i in range(n_minibatches):
        if is_hat:
            attention_w = grab_hat_attention_weights(model, tokenizer, minibatch[i], MAX_LEN, 'cuda')
        else:
            attention_w = grab_attention_weights(model, tokenizer, minibatch[i], MAX_LEN, 'cuda')
        adj_matricies.append(attention_w)
    # sample X layer X head X n_token X n_token
    adj_matricies = np.concatenate(adj_matricies, axis=1)
    adj_matricies = np.swapaxes(adj_matricies,axis1=0,axis2=1) # sample X layer X head X n_token X n_token
    Q.put(adj_matricies)

idx = 0
for i in tqdm(range(number_of_batches), desc="Feature Calculation Loop"):
    # Name of the file for topological features array
    stats_file = f"{output_dir}/{subset}_all_heads_{len(layers_of_interest)}_layers_{stats_name}_lists_array_{thrs}_thrs_MAX_LEN_{MAX_LEN}_{model_path.split('/')[-1]}_{i+1}_of_{number_of_batches}.npy"
    # Name of the file for ripser features array
    ripser_file = f"{output_dir}/{subset}_all_heads_{len(layers_of_interest)}_layers_MAX_LEN_{MAX_LEN}_{model_path.split('/')[-1]}_ripser_{i+1}_of_{number_of_batches}.npy"
    # Name of the file for template features array
    template_file = f"{output_dir}/{subset}_all_heads_{len(layers_of_interest)}_layers_MAX_LEN_{MAX_LEN}_{model_path.split('/')[-1]}_template_{i+1}_of_{number_of_batches}.npy"

    if os.path.exists(template_file): continue # Already generated, skipping

    t1 = time.time()
    # Refactor to run as a separate process so that memory is freed for ripserplusplus
    attention_grab = Process(target=get_attention_matrices, args=(model_path, tokenizer, batched_sentences[i], MAX_LEN, args.hat))
    attention_grab.start()
    adj_matricies = Q.get()
    attention_grab.join()
    print(f"Grabbed attentions. Time taken: {time.time() - t1}s")
    t1 = time.time()

    curr_batch_size = len(batched_sentences[i])
    num_of_workers = curr_batch_size
    pool = Pool(num_of_workers)

    stats_tuple_lists_array = []
    ntokens = ntokens_array[idx: idx+curr_batch_size]
    splitted = split_matricies_and_lengths(adj_matricies, ntokens, num_of_workers)
    arguments = [(m, thresholds_array, ntokens, stats_name.split("_"), stats_cap) for m, ntokens in splitted]
    stats_tuple_lists_array_part = pool.starmap(
        count_top_stats, arguments
    )
    stats_tuple_lists_array.append(np.concatenate([_ for _ in stats_tuple_lists_array_part], axis=3))

    stats_tuple_lists_array = np.concatenate(stats_tuple_lists_array, axis=3)
    np.save(stats_file, stats_tuple_lists_array)
    print(f"Calculated topological features. Time taken: {time.time() - t1}s")
    t1 = time.time()

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

    # Format: "h{dim}\_{type}\_{args}"
    #
    # Dimension: 0, 1, etc.; homology dimension
    #
    # Types:
    #
    #     1. s: sum of lengths; example: "h1_s".
    #     2. m: mean of lengths; example: "h1_m"
    #     3. v: variance of lengths; example "h1_v"
    #     4. n: number of barcodes with time of birth/death more/less then threshold.
    #         4.1. b/d: birth or death
    #         4.2. m/l: more or less than threshold
    #         4.2. t: threshold value
    #        example: "h0_n_d_m_t0.5", "h1_n_b_l_t0.75"
    #     5. t: time of birth/death of the longest barcode (not incl. inf).
    #         3.1. b/d: birth of death
    #         example: "h0_t_d", "h1_t_b"
    #     6. nb: number of barcodes in dim
    #        example: h0_nb
    #     7. e: entropy; example: "h1_e"

    dim = 1
    lower_bound = 1e-3
    ## Calculating and saving barcodes

    barcodes = defaultdict(list)
    splitted = split_matricies_and_lengths(adj_matricies, ntokens, num_of_workers)
    arguments = [(m, ntokens, dim, lower_bound) for m, ntokens in splitted]
    barcodes_all_parts = pool.starmap(
        get_only_barcodes, arguments
    )
    for barcode_part in barcodes_all_parts:
        barcodes = unite_barcodes(barcodes, barcode_part)

    ripser_feature_names=[
        'h0_s',
        'h0_m',
        'h0_v',
        'h0_e',
        'h0_t_b',
        'h0_t_d',
        'h0_nb',
        'h0_q',
        'h0_n_d_m_t0.75',
        'h0_n_d_m_t0.5',
        'h0_n_d_l_t0.25',
        'h1_t_b',
        'h1_n_b_m_t0.25',
        'h1_n_b_l_t0.95',
        'h1_n_b_l_t0.70',
        'h1_s',
        'h1_m',
        'h1_v',
        'h1_e',
        'h1_t_b',
        'h1_t_d',
        'h1_nb',
        'h1_q'
    ]

    features_array = []
    features_part = []
    n_heads = 12
    for layer in range(n_layers):
        features_layer = []
        for head in range(n_heads):
            barcode = reformat_barcodes(format_barcodes(barcodes[(layer, head)]))
            features = ripser_count.count_ripser_features(barcode, ripser_feature_names)
            features_layer.append(features)
        features_part.append(features_layer)
    features_array.append(np.asarray(features_part))

    features = np.concatenate(features_array, axis=2)
    np.save(ripser_file, features)
    print(f"Calculated ripser features. Time taken: {time.time() - t1}s")
    t1 = time.time()

    if args.hat: # HAT does not support 'comma' and 'dot', these require token level attention map but HAT provides segment level maps
        feature_list = ['self', 'beginning', 'prev', 'next']
    else:
        feature_list = ['self', 'beginning', 'prev', 'next', 'comma', 'dot']
    features_array = []

    sentences = data['sentence'].values[idx:idx+curr_batch_size]
    splitted_indexes = np.array_split(np.arange(curr_batch_size), num_of_workers)
    splitted_list_of_ids = [
        get_list_of_ids(sentences[indx], tokenizer, MAX_LEN, args.hat)
        for indx in splitted_indexes
    ]
    splitted_adj_matricies = [adj_matricies[indx] for indx in splitted_indexes]

    arguments = [(m, feature_list, list_of_ids) for m, list_of_ids in zip(splitted_adj_matricies, splitted_list_of_ids)]

    features_array_part = pool.starmap(
        calculate_features_t, arguments
    )
    features_array.append(np.concatenate([_ for _ in features_array_part], axis=3))

    features_array = np.concatenate(features_array, axis=3)
    np.save(template_file, features_array)
    print(f"Calculated template features. Time taken: {time.time() - t1}s")

    # Free up pool resources to make space for model to grab attentions
    pool.close()

