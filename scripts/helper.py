import pickle
import numpy as np
from custom_types import Dataset, Sequence
from configuration import Configuration
from typing import Dict
import os
import csv
from typing import Dict

def get_config_filepath(config_filepath: str) -> str:
    path = os.path.realpath(os.path.dirname(__file__))
    return f'{path}/../{config_filepath}'

def get_json_config():
    return Configuration.load_json()

def generate_classifier_data(ds: Dict[str, Sequence]):
    embedding_dim = None
    i = 0
    while not embedding_dim:
        if list(ds.values())[i].embedding is not None:
            embedding_dim = np.array(list(ds.values())[i].embedding).shape[1]
        i += 1
    cnt = 0
    for val in ds.values():
        if val.embedding is not None:
            cnt += val.embedding.shape[0]
    X = np.ndarray(shape=(cnt, embedding_dim), dtype=float)
    y = []
    i = 0
    for id, val in ds.items():
        if val.embedding is None:
            continue
        seq_len = val.embedding.shape[0]
        assert val.embedding.shape[0] == len(val.sequence), f"Embedding length doesn't match the sequence length: {val.sequence}"
        X[i:i+seq_len] = val.embedding
        i += seq_len
        #y_aux = [0]*len(val.seq)
        y_aux = [0]*seq_len
        for ix in val.annotations:
            y_aux[ix-1] = 1
        y += y_aux
    return X, y

def generate_test_train_data() -> Dataset:
    conf = get_json_config()
    data_directory = get_config_filepath(conf.data_directory)

    train_set = pickle.load(open(f'{data_directory}/sequences_TRAIN.pickle', 'rb'))
    test_set = pickle.load(open(f'{data_directory}/sequences_TEST.pickle', 'rb'))
    
    ttd = Dataset()
    ttd.X_train, ttd.y_train = generate_classifier_data(train_set)
    ttd.X_test, ttd.y_test = generate_classifier_data(test_set)
    
    ttd.X_train = ttd.X_train.astype("float32")
    ttd.X_test = ttd.X_test.astype("float32")
    ttd.y_train = np.array(ttd.y_train).astype("float32")
    ttd.y_test = np.array(ttd.y_test).astype("float32")
        
    return ttd

def get_class_weights(y_train) -> Dict[int, float]:
    
    pos = sum(y_train)
    neg = len(y_train) - pos 

    weight_for_neg = (1 / neg) * ((pos+neg) / 2.0) # weight for 0
    weight_for_pos = (1 / pos) * ((pos+neg) / 2.0)

    return {0: weight_for_neg, 1: weight_for_pos}


def process_dataset():
    conf = get_json_config()

    embedding_directory = get_config_filepath(conf.embeddings_directory)
    train_annotations_path = get_config_filepath(conf.train_annotations_path)
    test_annotations_path = get_config_filepath(conf.test_annotations_path)
    sequences_pickle_directory = get_config_filepath(conf.data_directory)

    for annotations_path, suffix in [(train_annotations_path, 'TRAIN'), (test_annotations_path, 'TEST')]:
        ds: Dict[str, Sequence] = {}
        with open(annotations_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                id = row[0].lower() + row[1].upper()
                annotations = row[2]
                sequence = row[3]
                embedding = np.load(f'{embedding_directory}/{id}.npy')

                ds[id] = Sequence(id, sequence, embedding)

                # check data consistency
                assert ds[id].sequence == sequence

                ds[id].add_annotations(annotations)

        f = open(f'{sequences_pickle_directory}/sequences_{suffix}.pickle', 'wb')
        pickle.dump(ds, f)
        f.close()
