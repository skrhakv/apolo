from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
import tensorflow as tf
from helper import evaluate_predictions, get_config_filepath, get_json_config
from custom_types import Results, Sequence, Protein
from typing import Dict, List
import pickle 
import csv
import statistics
import os

def make_predictions(ds: Dict[str, Sequence]) -> Dict[str, Results]:
    conf = get_json_config()
    model_directory = get_config_filepath(conf.model_directory)
    project_name = conf.project_name
    best_trained_model: tf.keras.Model = keras.models.load_model(f'{model_directory}/models/{project_name}/best_trained', custom_objects={'MatthewsCorrelationCoefficient': tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)})

    results : Dict[str, Results] = {}
    for protein_code, sequence in ds.items():
        
        seq_len = sequence.embedding.shape[0]
        assert sequence.embedding.shape[0] == len(sequence.sequence), "Embedding length doesn't match the sequence length"
        X = np.ndarray(shape=(sequence.embedding.shape[0], sequence.embedding.shape[1]), dtype=float)
        X[0:seq_len] = sequence.embedding

        y = [0]*seq_len
        for ix in sequence.annotations:
            y[ix-1] = 1

        X = X.astype("float32")
        y = np.array(y).astype("float32")

        predictions = best_trained_model.predict(X)

        results[protein_code] = Results(y, predictions, evaluate_predictions(predictions, y))
    return results

def process_set(ds, results: Dict[str, Results]) -> List[Protein]:
    avg_auc = []
    threshold_counter = 0
    proteins = []
    for protein_code, sequence in ds.items():
        protein = Protein(protein_code, sequence, results[protein_code].predictions, results[protein_code].actual_values)
        proteins.append(protein)
        avg_auc.append(protein.auc)
        if protein.satisfies_treshold(0.33):
            threshold_counter += 1
    return proteins

def save_to_csv(result_list: List[Protein], path):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)

        if hasattr(result_list[0], 'auc'):
            field = ["code", "length", "binding_residues", "FPR", "TPR", "ACC", "MCC", "F1", "AUC"]
        else:
            field = ["code", "length", "binding_residues", "FPR", "TPR", "ACC", "MCC", "F1"]
    
        writer.writerow(field)
        
        counter = 0
        avg_len = []
        avg_binding_res = []
        avg_fpr = []
        avg_tpr = []
        avg_acc = []
        avg_mcc = []
        avg_f1 = []
        avg_auc = []
        for protein in result_list:
            counter += 1
            avg_len.append(len(protein.sequence.sequence))
            avg_binding_res.append(sum(protein.actual_values))
            avg_fpr.append(protein.get_FPR())
            avg_tpr.append(protein.get_TPR())
            avg_acc.append(protein.accuracy)
            avg_mcc.append(protein.mcc)
            avg_f1.append(protein.f1)

            if hasattr(result_list[0], 'auc'):
                avg_auc.append(protein.auc)
                writer.writerow([protein.id, len(protein.sequence.sequence), sum(protein.actual_values), protein.get_FPR(), protein.get_TPR(), protein.accuracy, protein.mcc, protein.f1, protein.auc])
            else:
                writer.writerow([protein.id, len(protein.sequence.sequence), sum(protein.actual_values), protein.get_FPR(), protein.get_TPR(), protein.accuracy, protein.mcc, protein.f1])
        if hasattr(result_list[0], 'auc'):
            writer.writerow(["average", sum(avg_len) / counter, sum(avg_binding_res) / counter, sum(avg_fpr) / counter, sum(avg_tpr) / counter, sum(avg_acc) / counter, sum(avg_mcc) / counter, sum(avg_f1) / counter, sum(avg_auc) / counter ])
            writer.writerow(["standard deviation", statistics.stdev(avg_len), statistics.stdev(avg_binding_res), statistics.stdev(avg_fpr),statistics.stdev(avg_tpr),statistics.stdev(avg_acc),statistics.stdev(avg_mcc),statistics.stdev(avg_f1),statistics.stdev(avg_auc) ])
        else:
            writer.writerow(["average", sum(avg_len) / counter, sum(avg_binding_res) / counter, sum(avg_fpr) / counter, sum(avg_tpr) / counter, sum(avg_acc) / counter, sum(avg_mcc) / counter, sum(avg_f1) / counter ])
            writer.writerow(["standard deviation", statistics.stdev(avg_len), statistics.stdev(avg_binding_res), statistics.stdev(avg_fpr),statistics.stdev(avg_tpr),statistics.stdev(avg_acc),statistics.stdev(avg_mcc),statistics.stdev(avg_f1) ])

def create_statistics():
    conf = get_json_config()
    statistics_directory = get_config_filepath(conf.statistics_directory)
    data_directory = get_config_filepath(conf.data_directory)
    project_name = conf.project_name

    train_set = pickle.load(open(f'{data_directory}/sequences_TRAIN.pickle', 'rb'))
    test_set = pickle.load(open(f'{data_directory}/sequences_TEST.pickle', 'rb'))
    sets_combined = {**train_set, **test_set}

    results = make_predictions(sets_combined)
    
    train_proteins = process_set(train_set, results)
    test_proteins = process_set(test_set, results)

    os.makedirs(statistics_directory, exist_ok = True)
    save_to_csv(train_proteins, f'{statistics_directory}/train-{project_name}.csv')
    save_to_csv(test_proteins, f'{statistics_directory}/test-{project_name}.csv')
    save_to_csv(train_proteins + test_proteins, f'{statistics_directory}/overall-{project_name}.csv')
    

