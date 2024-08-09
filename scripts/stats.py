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
    predictions_output_directory = get_config_filepath(conf.predictions_output_directory)

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

        results[protein_code] = Results(y, predictions, evaluate_predictions(predictions, y), protein_code, predictions_output_directory)
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

# also missing because some bug in the uniprot mapping: 7ndl
not_in_pocketminer = ['7ndl', '4x19', '5igh', '1se8', '5yj2', '1fd4', '3bjp', '5aon', '4fkm', '2phz', '3mwg', '3t8b', '1fe6', '4dnc', '2dfp', '7nbc', '5acv', '5wm9', '3la7', '3bk9', '2czd', '3ve9', '5ujp', '7np0', '1tmi', '3hrm', '4bg8', '1h13', '2nt1', '2xdo', '3b1o', '4jax', '5h8k', '3kjr', '8gxj', '2idj', '2vqz', '3uyi', '5m7r', '4z0y', '2zcg', '5n49', '8hc1', '2vyr', '3lnz', '1xxo', '4nzv', '8h49', '6syh', '1x2g', '1g1m', '7c48', '3pfp', '5dy9', '7qzr', '4p32', '1k47', '2huw', '5gmc', '1r3m', '3x0x']

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

        overall_predictions = None
        overall_actual_values = None

        for protein in result_list:
            if protein.id[:4] in not_in_pocketminer:
                continue
            if protein.id[:4] == '7ndl':
                continue
            
            if overall_predictions is None:
                overall_predictions = protein.predictions_for_auc
                overall_actual_values = protein.actual_values
            else:
                overall_predictions = np.concatenate((overall_predictions, protein.predictions_for_auc), axis=0)
                overall_actual_values = np.concatenate((overall_actual_values, protein.actual_values), axis=0)
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

        overall = Protein('overall', 'X'*overall_predictions.shape[0], overall_predictions, overall_actual_values, overall=True)
        if hasattr(result_list[0], 'auc'):
            writer.writerow(["protein average", sum(avg_len) / counter, sum(avg_binding_res) / counter, sum(avg_fpr) / counter, sum(avg_tpr) / counter, sum(avg_acc) / counter, sum(avg_mcc) / counter, sum(avg_f1) / counter, sum(avg_auc) / counter ])
            writer.writerow(["protein standard deviation", statistics.stdev(avg_len), statistics.stdev(avg_binding_res), statistics.stdev(avg_fpr),statistics.stdev(avg_tpr),statistics.stdev(avg_acc),statistics.stdev(avg_mcc),statistics.stdev(avg_f1),statistics.stdev(avg_auc) ])
            writer.writerow(["overall", overall_predictions.shape[0], sum(overall.actual_values), overall.get_FPR(), overall.get_TPR(),overall.accuracy, overall.mcc, overall.f1, overall.auc ])
        else:
            writer.writerow(["protein average", sum(avg_len) / counter, sum(avg_binding_res) / counter, sum(avg_fpr) / counter, sum(avg_tpr) / counter, sum(avg_acc) / counter, sum(avg_mcc) / counter, sum(avg_f1) / counter ])
            writer.writerow(["protein standard deviation", statistics.stdev(avg_len), statistics.stdev(avg_binding_res), statistics.stdev(avg_fpr),statistics.stdev(avg_tpr),statistics.stdev(avg_acc),statistics.stdev(avg_mcc),statistics.stdev(avg_f1) ])
            writer.writerow(["overall", overall_predictions.shape[0], sum(overall.actual_values), overall.get_FPR(), overall.get_TPR(),overall.accuracy, overall.mcc, overall.f1 ])

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
    

