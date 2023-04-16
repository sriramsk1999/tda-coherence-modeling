from collections import defaultdict
import itertools
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import RobertaModel, RobertaTokenizerFast, RobertaForSequenceClassification, RobertaTokenizer
from math import ceil
from stats_count import *
from grab_weights import  text_preprocessing
import warnings
import ripser_count
from multiprocessing import Process, Queue
import json
import itertools
from collections import defaultdict
import os
from multiprocessing import Pool
import sys
import argparse
import yappi

warnings.filterwarnings('ignore')
np.random.seed(42) # For reproducibility.

def attention_to_self(matricies):
    """
    Calculates the distance between input matricies and identity matrix, 
    which representes the attention to the same token.
    """
    _, n, m = matricies.shape
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.eye(n)
    return matrix_distance(matricies, template_matrix)

def attention_to_next_token(matricies):
    """
    Calculates the distance between input and E=(i, i+1) matrix, 
    which representes the attention to the next token.
    """
    _, n, m = matricies.shape
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.triu(np.tri(n, k=1, dtype=matricies.dtype), k=1)
    return matrix_distance(matricies, template_matrix)

def attention_to_prev_token(matricies):
    """
    Calculates the distance between input and E=(i+1, i) matrix, 
    which representes the attention to the previous token.
    """
    _, n, m = matricies.shape
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.triu(np.tri(n, k=-1, dtype=matricies.dtype), k=-1)
    return matrix_distance(matricies, template_matrix)

def attention_to_beginning(matricies):
    """
    Calculates the distance between input and E=(i+1, i) matrix, 
    which representes the attention to [CLS] token (beginning).
    """
    _, n, m = matricies.shape
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.zeros((n, n))
    template_matrix[:, 0] = 1.0
    return matrix_distance(matricies, template_matrix)

def attention_to_ids(matricies, list_of_ids, token_id):
    """
    Calculates the distance between input and ids matrix, 
    which representes the attention to some particular tokens,
    which ids are in the `list_of_ids` (commas, periods, separators).
    """
    batch_size, n, m = matricies.shape
    EPS = 1e-7
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
#     assert len(list_of_ids) == batch_size, f"List of ids length doesn't match the dimension of the matrix"
    template_matrix = np.zeros_like(matricies)
    ids = np.argwhere(list_of_ids == token_id)
    if len(ids):
        batch_ids, row_ids = zip(*ids)
        template_matrix[np.array(batch_ids), :, np.array(row_ids)] = 1.0
        template_matrix /= (np.sum(template_matrix, axis=-1, keepdims=True) + EPS)
    return matrix_distance(matricies, template_matrix, broadcast=False)


def count_template_features(matricies, feature_list=['self', 'beginning', 'prev', 'next', 'comma', 'dot'], ids=None):
    features = []
    comma_id = 1010
    dot_id = 1012
    for feature in feature_list:
        if feature == 'self':
            features.append(attention_to_self(matricies))
        elif feature == 'beginning':
            features.append(attention_to_beginning(matricies))
        elif feature == 'prev':
            features.append(attention_to_prev_token(matricies))
        elif feature == 'next':
            features.append(attention_to_next_token(matricies))
        elif feature == 'comma':
            features.append(attention_to_ids(matricies, ids, comma_id))
        elif feature == 'dot':
            features.append(attention_to_ids(matricies, ids, dot_id))
    return np.array(features)

def calculate_features_t(adj_matricies, template_features, ids=None):
    """Calculate template features for adj_matricies"""
    features = []
    for layer in range(adj_matricies.shape[1]):
        features.append([])
        for head in range(adj_matricies.shape[2]):
            matricies = adj_matricies[:, layer, head, :, :]
            lh_features = count_template_features(matricies, template_features, ids) # samples X n_features
            features[-1].append(lh_features)
    return np.asarray(features) # layer X head X n_features X samples


def get_list_of_ids(sentences, tokenizer, MAX_LEN=None, model_type='roberta'):
    if model_type == 'hat' or model_type == 'longformer':
        inputs = tokenizer([text_preprocessing(s) for s in sentences],
                               add_special_tokens=True,
                               max_length=MAX_LEN,                # Max length to truncate/pad
                               padding='max_length',              # Pad sentence to max length)
                               truncation=True
                              )
    elif model_type == 'roberta':
        inputs = tokenizer.batch_encode_plus([text_preprocessing(s) for s in sentences],
                                           add_special_tokens=True,
                                           max_length=MAX_LEN,             # Max length to truncate/pad
                                           pad_to_max_length=True,         # Pad sentence to max length)
                                           truncation=True
                                          )
    return np.array(inputs['input_ids'])

def reformat_barcodes(barcodes):
    """Return barcodes to their original format"""
    formatted_barcodes = []
    for barcode in barcodes:
        formatted_barcode = {}
        for dim in barcode:
            formatted_barcode[int(dim)] = np.asarray(
                [(b, d) for b,d in barcode[dim]], dtype=[('birth', '<f4'), ('death', '<f4')]
            )
        formatted_barcodes.append(formatted_barcode)
    return formatted_barcodes

def subprocess_wrap(queue, function, args):
    queue.put(function(*args))
    print("Putted in Queue")
    queue.close()
    exit()

def get_only_barcodes(adj_matricies, ntokens_array, dim, lower_bound):
    """Get barcodes from adj matricies for each layer, head"""
    barcodes = {}
    layers, heads = range(adj_matricies.shape[1]), range(adj_matricies.shape[2])
    for (layer, head) in itertools.product(layers, heads):
        matricies = adj_matricies[:, layer, head, :, :]
        barcodes[(layer, head)] = ripser_count.get_barcodes(matricies, ntokens_array, dim, lower_bound, (layer, head))
    return barcodes

def format_barcodes(barcodes):
    """Reformat barcodes to json-compatible format"""
    return [{d: b[d].tolist() for d in b} for b in barcodes]

def save_barcodes(barcodes, filename):
    """Save barcodes to file"""
    formatted_barcodes = defaultdict(dict)
    for layer, head in barcodes:
        formatted_barcodes[layer][head] = format_barcodes(barcodes[(layer, head)])
    json.dump(formatted_barcodes, open(filename, 'w'))
    
def unite_barcodes(barcodes, barcodes_part):
    """Unite 2 barcodes"""
    for (layer, head) in barcodes_part:
        barcodes[(layer, head)].extend(barcodes_part[(layer, head)])
    return barcodes

def split_matricies_and_lengths(adj_matricies, ntokens, number_of_splits):
    splitted_ids = np.array_split(np.arange(ntokens.shape[0]), number_of_splits) 
    splitted = [(adj_matricies[ids], ntokens[ids]) for ids in splitted_ids]
    return splitted

def get_token_length(batch_texts):
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

def matrix_distance(matricies, template, broadcast=True):
    """
    Calculates the distance between the list of matricies and the template matrix.
    Args:
    
    -- matricies: np.array of shape (n_matricies, dim, dim)
    -- template: np.array of shape (dim, dim) if broadcast else (n_matricies, dim, dim)
    
    Returns:
    -- diff: np.array of shape (n_matricies, )
    """
    diff = np.linalg.norm(matricies-template, ord='fro', axis=(1, 2))
    div = np.linalg.norm(matricies, ord='fro', axis=(1, 2))**2
    if broadcast:
        div += np.linalg.norm(template, ord='fro')**2
    else:
        div += np.linalg.norm(template, ord='fro', axis=(1, 2))**2
    return diff/np.sqrt(div)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Calculating barcodes and template features')

    parser.add_argument("--cuda", help="Cuda Device", required=True)
    parser.add_argument("--data_name", help="Data Name", required=True)
    parser.add_argument("--IO_dir", help="I/O dir", required=True)
    parser.add_argument("--chunk_start_idx", help="Chunk start index", type=int, required=True)
    parser.add_argument("--chunk_size", help="Chunk size", type=int, default=10000)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=10)
    parser.add_argument("--dump_size", help="Dump size", type=int, default=100)

    args = parser.parse_args()
    print(args)

    num_of_workers = 20
    pool = Pool(num_of_workers)
    max_tokens_amount  = 256 # The number of tokens to which the tokenized text is truncated / padded.

    layers_of_interest = [i for i in range(12)]  # Layers for which attention matrices and features on them are
                                                # calculated. For calculating features on all layers, leave it be
                                                # [i for i in range(12)].

    # model_path = tokenizer_path = "bert-base-cased"
    model_path = tokenizer_path = "roberta-base"

    cuda_device = args.cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

    # ## Filenames
    subset = args.data_name          # .csv file with the texts, for which we count topological features
    input_dir = args.IO_dir  # Name of the directory with .csv file
    output_dir = args.IO_dir # Name of the directory with calculations results

    prefix = output_dir + subset

    r_file     = output_dir + 'attentions/' + subset  + "_all_heads_" + str(len(layers_of_interest)) + "_layers_MAX_LEN_" +              str(max_tokens_amount) + "_" + model_path.split("/")[-1]
    # Name of the file for attention matrices weights

    barcodes_file = output_dir + 'barcodes/' + subset  + "_all_heads_" + str(len(layers_of_interest)) + "_layers_MAX_LEN_" +              str(max_tokens_amount) + "_" + model_path.split("/")[-1]
    # Name of the file for barcodes information

    # .csv file must contain the column with the name **sentence** with the texts. It can also contain the column **labels**, which will be needed for testing. Any other arbitrary columns will be ignored.

    chunk_start_idx = args.chunk_start_idx
    chunk_size = args.chunk_size
    batch_size = args.batch_size # batch size
    DUMP_SIZE = args.dump_size # number of batches to be dumped

    print("Reading CSV")
    try:
        data = pd.read_csv(input_dir + subset + ".csv").reset_index(drop=True)
    except:
        #data = pd.read_csv(input_dir + subset + ".tsv", delimiter="\t")
        data = pd.read_csv(input_dir + subset + ".tsv", delimiter="\t", header=None)
        data.columns = ["0", "labels", "2", "sentence"]
    print("CSV Read")

    data = data.iloc[chunk_start_idx:min(chunk_start_idx+chunk_size, len(data))]

    MAX_LEN = max_tokens_amount
    # tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
    tokenizer.do_lower_case = False

    data['tokenizer_length'] = get_token_length(data['sentence'].values)
    ntokens_array = data['tokenizer_length'].values
    number_of_batches = ceil(len(data['sentence']) / batch_size)

    # ## Calculating Ripser features

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

    adj_filenames = [
        output_dir + 'attentions/' + filename
        for filename in os.listdir(output_dir + 'attentions/') if r_file in (output_dir + 'attentions/' + filename)
    ]
    # sorted by part number
    adj_filenames = sorted(adj_filenames, key = lambda x: int(x.split('_')[-1].split('of')[0][4:].strip()))

    dim = 1
    lower_bound = 1e-3

    ## Calculating and saving barcodes

    print("Starting calc_barcodes")
    # yappi.start()
    queue = Queue()
    number_of_splits = 2
    for i, filename in enumerate(tqdm(adj_filenames, desc='Calculating barcodes')):
        barcodes = defaultdict(list)
        print("Loading np filename")
        adj_matricies = np.load(filename, allow_pickle=True) # samples X
        print(f"Matricies loaded from: {filename}")
        ntokens = ntokens_array[i*batch_size*DUMP_SIZE : (i+1)*batch_size*DUMP_SIZE]
        print("Splitting matrices and lengths")
        splitted = split_matricies_and_lengths(adj_matricies, ntokens, number_of_splits)
        print(f"Splitted !")
        if __name__ == "__main__":
            for matricies, ntokens in tqdm(splitted, leave=False):
                print("-- Split --")
                p = Process(
                    target=subprocess_wrap,
                    args=(
                        queue,
                        get_only_barcodes,
                        (matricies, ntokens, dim, lower_bound)
                    )
                )
                p.start()
                print("Started")
                barcodes_part = queue.get() # block until putted and get barcodes from the queue
                # print("Got")
                print("Features got.")
                p.join() # release resources
                print("The process is joined.")
                p.close() # releasing resources of ripser
                print("The proccess is closed.")

                barcodes = unite_barcodes(barcodes, barcodes_part)
        part = filename.split('_')[-1].split('.')[0]
        save_barcodes(barcodes, barcodes_file + '_' + part + '.json')
    print("\n------------------STATS FOR BARCODE CALCULATION LOOP------------------\n")
    print("\n-> FUNC STATS\n")
    # yappi.get_func_stats().print_all()
    print("\n------------------------------------\n")
    print("-> THREAD STATS\n")
    # yappi.get_thread_stats().print_all()
    print("\n###########################################################################\\n")
    ## Calculating features of saved barcodes

    # ripser_feature_names=[
    #     'h0_s',
    #     'h0_e',
    #     'h0_t_d',
    #     'h0_n_d_m_t0.75',
    #     'h0_n_d_m_t0.5',
    #     'h0_n_d_l_t0.25',
    #     'h1_t_b',
    #     'h1_n_b_m_t0.25',
    #     'h1_n_b_l_t0.95',
    #     'h1_n_b_l_t0.70',
    #     'h1_s',
    #     'h1_e',
    #     'h1_v',
    #     'h1_nb'
    # ]
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
        # 'h1_t_b',
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
    adj_filenames = [
        output_dir + 'barcodes/' + filename
        for filename in os.listdir(output_dir + 'barcodes/') if r_file.split('/')[-1] == filename.split('_part')[0]
    ]
    adj_filenames = sorted(adj_filenames, key = lambda x: int(x.split('_')[-1].split('of')[0][4:].strip()))

    features_array = []

    # yappi.start()
    for filename in tqdm(adj_filenames, desc='Calculating ripser++ features'):
        barcodes = json.load(open(filename))
        print(f"Barcodes loaded from: {filename}", flush=True)
        features_part = []
        for layer in barcodes:
            features_layer = []
            for head in barcodes[layer]:
                ref_barcodes = reformat_barcodes(barcodes[layer][head])
                features = ripser_count.count_ripser_features(ref_barcodes, ripser_feature_names)
                features_layer.append(features)
            features_part.append(features_layer)
        features_array.append(np.asarray(features_part))

    # ripser_file = output_dir + 'features/' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers"+ "_MAX_LEN_" + str(max_tokens_amount) +              "_" + model_path.split("/")[-1] + "_ripser" + '.npy'
    ripser_file = output_dir + 'new_features/' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers"+ "_MAX_LEN_" + str(max_tokens_amount) +              "_" + model_path.split("/")[-1] + "_ripser" + '.npy'

    features = np.concatenate(features_array, axis=2)
    np.save(ripser_file, features)

    print("\n------------------STATS FOR RIPSER CALCULATION LOOP + SAVING------------------\n")
    print("\n-> FUNC STATS\n")
    # yappi.get_func_stats().print_all()
    print("\n------------------------------------\n")
    print("-> THREAD STATS\n")
    # yappi.get_thread_stats().print_all()
    print("\n###########################################################################\n\n")

    # ## Calculating template features

    attention_dir = input_dir + 'attentions/'
    attention_name = subset + '_all_heads_12_layers_MAX_LEN_256_roberta-base'


    texts_name = input_dir + subset + '.csv'
    MAX_LEN = max_tokens_amount
    feature_list = ['self', 'beginning', 'prev', 'next', 'comma', 'dot']

    adj_filenames = [
        attention_dir + filename
        for filename in os.listdir(attention_dir)
        if attention_name == filename.split("_part")[0]
    ]
    # sorted by part number
    adj_filenames = sorted(adj_filenames, key = lambda x: int(x.split('_')[-1].split('of')[0][4:].strip()))
    features_array = []

    # yappi.start()
    for i, filename in tqdm(list(enumerate(adj_filenames)), desc='Features calc'):
        adj_matricies = np.load(filename, allow_pickle=True)
        batch_size = adj_matricies.shape[0]
        sentences = data['sentence'].values[i*batch_size:(i+1)*batch_size]
        splitted_indexes = np.array_split(np.arange(batch_size), num_of_workers)
        splitted_list_of_ids = [
            get_list_of_ids(sentences[indx], tokenizer)
            for indx in tqdm(splitted_indexes, desc=f"Calculating token ids on iter {i} from {len(adj_filenames)}")
        ]
        splitted_adj_matricies = [adj_matricies[indx] for indx in splitted_indexes]

        args = [(m, feature_list, list_of_ids) for m, list_of_ids in zip(splitted_adj_matricies, splitted_list_of_ids)]

        features_array_part = pool.starmap(
            calculate_features_t, args
        )
        features_array.append(np.concatenate([_ for _ in features_array_part], axis=3))

    features_array = np.concatenate(features_array, axis=3)
    # np.save(input_dir + "features/" + attention_name + "_template.npy", features_array)
    np.save(input_dir + "new_features/" + attention_name + "_template.npy", features_array)


    print("\n------------------STATS FOR FEATURE CALCULATION LOOP + SAVING------------------\n")
    print("\n-> FUNC STATS\n")
    # yappi.get_func_stats().print_all()
    print("\n------------------------------------\n")
    print("-> THREAD STATS\n")
    # yappi.get_thread_stats().print_all()
    print("\n###########################################################################\n\n")

