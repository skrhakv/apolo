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
        # assert sequence.embedding.shape[0] == len(sequence.sequence), "Embedding length doesn't match the sequence length"
        X = np.ndarray(shape=(sequence.embedding.shape[0], sequence.embedding.shape[1]), dtype=float)
        X[0:seq_len] = sequence.embedding

        y = [0]*seq_len
        for ix in sequence.annotations:
            y[ix - 1] = 1

        X = X.astype("float32")
        y = np.array(y).astype("float32")

        predictions = best_trained_model.predict(X)

        results[protein_code] = Results(y, predictions, evaluate_predictions(predictions, y), protein_code)
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

not_in_pocketminer = set(['3a7gA' ,'1k3aA' ,'3wrfA' ,'5n0xB' ,'6cwwB' ,'8bhuAAA' ,'7av9AAA' ,'6eyrA' ,'1f13B' ,'2e1cA' ,'3vc7B' ,'3ih2A' ,'2yzgA' ,'1eooA' ,'3futA' ,'7l8qA' ,'3fixB' ,'5cxgD' ,'1h13A' ,'5yj2C' ,'5o2nA' ,'3zniM' ,'3m5vB' ,'4tl1B' ,'2zosB' ,'4x1cF' ,'2nvpA' ,'3us5A' ,'2cxyA' ,'6pczB' ,'1vk4A' ,'4bg8A' ,'5oa5B' ,'4bktC' ,'3bjpA' ,'1t5hX' ,'3q7nA' ,'3spbC' ,'1rtuA' ,'1x2gC' ,'3i7cA'])
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
            if protein.id in not_in_pocketminer:
                continue
            counter += 1
            avg_len.append(protein.sequence.embedding.shape[0])
            avg_binding_res.append(sum(protein.actual_values))
            avg_fpr.append(protein.get_FPR())
            avg_tpr.append(protein.get_TPR())
            avg_acc.append(protein.accuracy)
            avg_mcc.append(protein.mcc)
            avg_f1.append(protein.f1)

            if hasattr(result_list[0], 'auc'):
                avg_auc.append(protein.auc)
                writer.writerow([protein.id, protein.sequence.embedding.shape[0], sum(protein.actual_values), protein.get_FPR(), protein.get_TPR(), protein.accuracy, protein.mcc, protein.f1, protein.auc])
            else:
                writer.writerow([protein.id, protein.sequence.embedding.shape[0], sum(protein.actual_values), protein.get_FPR(), protein.get_TPR(), protein.accuracy, protein.mcc, protein.f1])
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
    train_annotations_paths = [get_config_filepath(
            i) for i in conf.train_annotations_path]
    
    train_set = {}
    for i in range(len(train_annotations_paths)):
        train_set = {**train_set, **pickle.load(
            open(f'{data_directory}/sequences_TRAIN_FOLD_{i}.pickle', 'rb'))}
    test_set = pickle.load(open(f'{data_directory}/sequences_TEST.pickle', 'rb'))
    sets_combined = {**train_set, **test_set}

    results = make_predictions(sets_combined)
    
    train_proteins = process_set(train_set, results)
    test_proteins = process_set(test_set, results)

    os.makedirs(statistics_directory, exist_ok = True)
    save_to_csv(train_proteins, f'{statistics_directory}/train-{project_name}.csv')
    save_to_csv(test_proteins, f'{statistics_directory}/test-{project_name}.csv')
    save_to_csv(train_proteins + test_proteins, f'{statistics_directory}/overall-{project_name}.csv')
    

