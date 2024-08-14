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

def process_set(ds, results: Dict[str, Results], threshold: float) -> List[Protein]:
    avg_auc = []
    threshold_counter = 0
    proteins = []
    for protein_code, sequence in ds.items():
        protein = Protein(protein_code, sequence, results[protein_code].predictions, results[protein_code].actual_values, threshold=threshold)
        proteins.append(protein)
        avg_auc.append(protein.auc)
        if protein.satisfies_treshold(0.33):
            threshold_counter += 1
    return proteins

# also missing because some bug in the uniprot mapping: 7ndl
not_in_pocketminer = ['7ndl', '4x19', '5igh', '1se8', '5yj2', '1fd4', '3bjp', '5aon', '4fkm', '2phz', '3mwg', '3t8b', '1fe6', '4dnc', '2dfp', '7nbc', '5acv', '5wm9', '3la7', '3bk9', '2czd', '3ve9', '5ujp', '7np0', '1tmi', '3hrm', '4bg8', '1h13', '2nt1', '2xdo', '3b1o', '4jax', '5h8k', '3kjr', '8gxj', '2idj', '2vqz', '3uyi', '5m7r', '4z0y', '2zcg', '5n49', '8hc1', '2vyr', '3lnz', '1xxo', '8h49', '6syh', '1x2g', '1g1m', '7c48', '3pfp', '5dy9', '7qzr', '4p32', '1k47', '2huw', '5gmc', '1r3m', '3x0x']
is_in_p2rank = ['1arlA', '1bk2A', '1bzjA', '1cwqA', '1dq2A', '1e6kA', '1evyA', '1g59A', '1h13A', '1i7nA', '1k47D', '1ksgB', '1kx9A', '1kxrA', '1lbeB', '1m5wD', '1nd7A', '1p4oB', '1p4vA', '1p9oA', '1pu5C', '1q4kA', '1rjbA', '1rtcA', '1se8A', '1ukaA', '1uteA', '1vsnA', '1x2gC', '1xgdA', '1xjfA', '1xqvA', '1xtcA', '1zm0A', '2akaA', '2d05A', '2dfpA', '2femA', '2fhzB', '2h7sA', '2i3aD', '2i3rA', '2iytA', '2phzA', '2pkfA', '2pwzG', '2qbvA', '2rfjB', '2v6mD', '2vl2C', '2vqzF', '2vyrA', '2w8nA', '2x47A', '2xsaA', '2zj7A', '3a0xA', '3bjpA', '3f4kA', '3flgA', '3fzoA', '3gdgB', '3h8aB', '3i8sB', '3idhA', '3jzgA', '3k01A', '3ly8A', '3mwgB', '3n4uA', '3nx1B', '3pbfA', '3rwvA', '3st6C', '3t8bA', '3tpoA', '3ugkA', '3uyiA', '3v55A', '3vgmA', '3w90A', '3wb9C', '4aemA', '4amvB', '4bg8A', '4cmwB', '4dncB', '4e1yA', '4fkmB', '4gpiC', '4gv9A', '4hyeA', '4ikvA', '4ilgA', '4j4eF', '4jaxF', '4jfrC', '4kmyA', '4mwiA', '4nzvB', '4oqoB', '4p2fA', '4qvkB', '4r0xA', '4rvtB', '4ttpA', '4uc8A', '4uumA', '4zm7A', '4zoeB', '5acvB', '5b0eB', '5cazA', '5e0vA', '5ey7B', '5hijA', '5htoE', '5i3tE', '5ighA', '5kcgB', '5locA', '5o8bA', '5sc2A', '5tc0B', '5tviV', '5uxaA', '5wbmB', '5yhbA', '5yj2C', '5yqpA', '5ysbB', '5z18C', '5zj4D', '6a98C', '6btyB', '6cqeB', '6du4A', '6eqjA', '6f52A', '6fc2C', '6fgjB', '6g6yA', '6heiA', '6isuA', '6jq9B', '6kscA', '6n5jB', '6neiB', '6o4fH', '6tx0B', '6vleA', '6w10A', '7c48A', '7c63A', '7de1A', '7e5qB', '7f2mB', '7f4yB', '7kayA', '7nc8D', '7ndlB', '7nlxA', '7o1iA', '7qoqA', '7qzrD', '7v8kB', '7w19A', '7x0fA', '7x0gB', '7x0iB', '7xgfE', '7yjcA', '8aeqA', '8aqiB', '8b9pA', '8breB', '8h27A', '8i84B', '8iasB', '8j11X', '8onnE', '8u3nA', '8vxuB', '9atcA']
def save_to_csv(result_list: List[Protein], path, threshold=0.5):
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
            # if protein.id[:4] in not_in_pocketminer:
            #     continue
            if protein.id not in is_in_p2rank:
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

        overall = Protein('overall', 'X'*overall_predictions.shape[0], overall_predictions, overall_actual_values, overall=True, threshold=threshold)
        if hasattr(result_list[0], 'auc'):
            writer.writerow(["protein average", sum(avg_len) / counter, sum(avg_binding_res) / counter, sum(avg_fpr) / counter, sum(avg_tpr) / counter, sum(avg_acc) / counter, sum(avg_mcc) / counter, sum(avg_f1) / counter, sum(avg_auc) / counter ])
            writer.writerow(["protein standard deviation", statistics.stdev(avg_len), statistics.stdev(avg_binding_res), statistics.stdev(avg_fpr),statistics.stdev(avg_tpr),statistics.stdev(avg_acc),statistics.stdev(avg_mcc),statistics.stdev(avg_f1),statistics.stdev(avg_auc) ])
            writer.writerow(["overall", overall_predictions.shape[0], sum(overall.actual_values), overall.get_FPR(), overall.get_TPR(),overall.accuracy, overall.mcc, overall.f1, overall.auc ])
            print(', '.join([str(i) for i in ["overall", overall_predictions.shape[0], sum(overall.actual_values), overall.get_FPR(), overall.get_TPR(),overall.accuracy, overall.mcc, overall.f1, overall.auc ]]))
        else:
            writer.writerow(["protein average", sum(avg_len) / counter, sum(avg_binding_res) / counter, sum(avg_fpr) / counter, sum(avg_tpr) / counter, sum(avg_acc) / counter, sum(avg_mcc) / counter, sum(avg_f1) / counter ])
            writer.writerow(["protein standard deviation", statistics.stdev(avg_len), statistics.stdev(avg_binding_res), statistics.stdev(avg_fpr),statistics.stdev(avg_tpr),statistics.stdev(avg_acc),statistics.stdev(avg_mcc),statistics.stdev(avg_f1) ])
            writer.writerow(["overall", overall_predictions.shape[0], sum(overall.actual_values), overall.get_FPR(), overall.get_TPR(),overall.accuracy, overall.mcc, overall.f1 ])

def create_statistics():
    conf = get_json_config()
    statistics_directory = get_config_filepath(conf.statistics_directory)
    data_directory = get_config_filepath(conf.data_directory)
    project_name = conf.project_name
    # train_annotations_paths = [get_config_filepath(
    #         i) for i in conf.train_annotations_path]
    
    # train_set = {}
    # for i in range(len(train_annotations_paths)):
    #     train_set = {**train_set, **pickle.load(
    #         open(f'{data_directory}/sequences_TRAIN_FOLD_{i}.pickle', 'rb'))}
    test_set = pickle.load(open(f'{data_directory}/sequences_TEST.pickle', 'rb'))
    # sets_combined = {**train_set, **test_set}
    sets_combined=test_set
    results = make_predictions(sets_combined)

    print(', '.join(["code", "length", "binding_residues", "FPR", "TPR", "ACC", "MCC", "F1", "AUC"]))
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

        # train_proteins = process_set(train_set, results)
        test_proteins = process_set(test_set, results, threshold=threshold)

        os.makedirs(statistics_directory, exist_ok = True)
        # save_to_csv(train_proteins, f'{statistics_directory}/train-{project_name}.csv')
        save_to_csv(test_proteins, f'{statistics_directory}/test-{project_name}.csv', threshold=threshold)
        # save_to_csv(train_proteins + test_proteins, f'{statistics_directory}/overall-{project_name}.csv')
    

