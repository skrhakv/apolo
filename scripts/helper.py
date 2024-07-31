import pickle
import numpy as np
from custom_types import Dataset, Sequence
from configuration import Configuration
from typing import Dict
import os
import csv
from typing import Dict
import numpy as np
import tensorflow as tf
from sklearn import metrics
from typing import Union, Dict
import requests
import zipfile
import io


def evaluate_predictions(predictions, ys) -> Dict[str, Union[np.ndarray, float]]:
    p = tf.argmax(predictions, 1).numpy()
    return {
        'CM': metrics.confusion_matrix(ys, p),
        'MCC': metrics.matthews_corrcoef(ys, p),
        'AUC': metrics.roc_auc_score(ys, p),
        'ACC': metrics.accuracy_score(ys, p)
    }


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
        # assert val.embedding.shape[0] == len(
        #     val.sequence), f"Embedding length doesn't match the sequence length: {val.sequence}"
        X[i:i + seq_len] = val.embedding
        i += seq_len
        # y_aux = [0]*len(val.seq)
        y_aux = [0] * seq_len
        for ix in val.annotations:
            if ix - 1 >= len(y_aux):
                print(id, ix)
            y_aux[ix - 1] = 1
        y += y_aux
        assert val.embedding.shape[0] == len(y_aux)
        #     val.sequence), f"Embedding length doesn't match the sequence length: {val.sequence}"

    return X, y


def generate_test_train_data(validation_folds=False) -> Dataset:
    conf = get_json_config()
    sequences_pickle_directory = get_config_filepath(conf.data_directory)
    train_annotations_paths = [get_config_filepath(
        i) for i in conf.train_annotations_path]

    if not validation_folds:
        train_set = {}
        print('load train set ...')
        for i in range(len(train_annotations_paths)):
            print(f'\t process fold {i}')
            train_set = {**train_set, **pickle.load(
                open(f'{sequences_pickle_directory}/sequences_TRAIN_FOLD_{i}.pickle', 'rb'))}
        print('load test set ...')

        test_set = pickle.load(
            open(f'{sequences_pickle_directory}/sequences_TEST.pickle', 'rb'))
        
        ttd = Dataset()
        print('processing train set ...')
        ttd.X_train, ttd.y_train = generate_classifier_data(train_set)
        print('processing test set ...')
        ttd.X_test, ttd.y_test = generate_classifier_data(test_set)

    else:
        print('load train set ...')
        ttd = Dataset()

        for i in range(len(train_annotations_paths)):
            print(f'\t load fold {i} ...')
            train_set = pickle.load(
                open(f'{sequences_pickle_directory}/sequences_TRAIN_FOLD_{i}.pickle', 'rb'))
            tmp_ttd = Dataset()
            tmp_ttd.X_train, tmp_ttd.y_train = generate_classifier_data(train_set)
            tmp_ttd.groups = [i] * len(tmp_ttd.y_train) 
            if len(ttd.groups) != 0:
                ttd = ttd.append_train(tmp_ttd)
            else: ttd = tmp_ttd

    print('generate dataset ...')

    ttd.X_train = ttd.X_train.astype("float32")
    if ttd.X_test is not None:
        ttd.X_test = ttd.X_test.astype("float32")
    ttd.y_train = np.array(ttd.y_train).astype("float32")
    if ttd.X_test is not None:
        ttd.y_test = np.array(ttd.y_test).astype("float32")

    return ttd


def get_class_weights(y_train) -> Dict[int, float]:

    pos = sum(y_train)
    neg = len(y_train) - pos

    weight_for_neg = (1 / neg) * ((pos + neg) / 2.0)  # weight for 0
    weight_for_pos = (1 / pos) * ((pos + neg) / 2.0)

    return {0: weight_for_neg, 1: weight_for_pos}


def compute_embeddings_remotely(fasta_file_location, embedding_directory, server_url, embedder):
    files = {'fasta': open(fasta_file_location, 'rb')}

    request = requests.get(f"{server_url}?embedder={embedder}", files=files)

    z = zipfile.ZipFile(io.BytesIO(request.content))
    z.extractall(embedding_directory)


def process_dataset():
    conf = get_json_config()

    embedding_directory = get_config_filepath(conf.embeddings_directory)
    train_annotations_paths = [get_config_filepath(
        i) for i in conf.train_annotations_path]
    test_annotations_path = get_config_filepath(conf.test_annotations_path)
    sequences_pickle_directory = get_config_filepath(conf.data_directory)
    fasta_file_location = get_config_filepath(
        conf.remote_embedding_computation.fasta_file_location)

    if not conf.remote_embedding_computation.are_embeddings_precomputed:
        compute_embeddings_remotely(fasta_file_location, embedding_directory,
                                    conf.remote_embedding_computation.server_url, conf.remote_embedding_computation.embedder)

    # process test subset
    ds: Dict[str, Sequence] = {}
    with open(test_annotations_path, 'r') as csvfile:
        print(f'Processing {test_annotations_path} ... ')

        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            for chain_id in row[1].split('-'):
                id = row[0].lower() + chain_id
                annotations = row[3]
                if annotations == '':
                    continue

                annotations = [i.split('_')[1] for i in annotations.split(' ') if i.split('_')[0] == chain_id]
                sequence = row[4]
                embedding = np.load(f'{embedding_directory}/{id}.npy')

                ds[id] = Sequence(id, sequence, embedding)

                # check data consistency
                # assert ds[id].sequence == sequence
                ds[id].add_annotations(' '.join(annotations))

    with open(f'{sequences_pickle_directory}/sequences_TEST.pickle', 'wb') as f:
        pickle.dump(ds, f)

    # process each train fold subset
    for i, train_annotations_path in enumerate(train_annotations_paths):
        ds: Dict[str, Sequence] = {}
        print(f'Processing {train_annotations_path} ... ')
        with open(train_annotations_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                for chain_id in row[1].split('-'):
                    id = row[0].lower() + chain_id
                    annotations = row[3]

                    annotations = [i.split('_')[1] for i in annotations.split(' ') if i.split('_')[0] == chain_id]
                    
                    if len(annotations) < 1:
                        continue
                    
                    sequence = row[4]
                    embedding = np.load(f'{embedding_directory}/{id}.npy')

                    ds[id] = Sequence(id, sequence, embedding)
                    # check data consistency
                    # assert ds[id].sequence == sequence
                    ds[id].add_annotations(' '.join(annotations))

        with open(f'{sequences_pickle_directory}/sequences_TRAIN_FOLD_{i}.pickle', 'wb') as f:
            pickle.dump(ds, f)
