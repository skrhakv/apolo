import numpy as np
from typing import List
import tensorflow as tf
import numpy as np
from sklearn import metrics  

class Sequence:
    def __init__(self, id, sequence, embedding):
        self.id: str = id
        self.sequence: str = sequence        
        self.embedding: np.ndarray = embedding
        self.annotations: List[int] = []
    
    def add_annotations(self, annotations):
        # annotations: D4 D48 V115 T116 N118 V120 G121 V122
        self.annotations = [int(res[1:]) for res in annotations.split(' ')]
        for annotation in annotations.split(' '):
            aminoacid = annotation[:1]
            label_seq_id = int(annotation[1:])

            assert aminoacid == self.sequence[label_seq_id - 1], f'ID {self.id}: The annotation {annotation} letter does not match the sequence at the specified position! Annotation = {aminoacid}, Sequence[{label_seq_id}] = {self.sequence[label_seq_id - 1]}'
            self.annotations.append(label_seq_id)

class Dataset:
    X_train: np.ndarray = None
    X_test: np.ndarray = None
    y_train: np.ndarray = None
    y_test: np.ndarray = None

class Results:
    def __init__(self, actual_values, predictions, stats):
        self.actual_values = actual_values
        self.predictions = predictions
        self.stats = stats

class Protein:
    def __init__(self, id, sequence, predictions, actual_values, prank=False):
        self.id: str = id
        self.sequence: Sequence = sequence        
        self.actual_values: np.ndarray = actual_values
        if prank:
            self.predictions = predictions
        else:
            self.predictions: np.ndarray = tf.argmax(predictions, 1).numpy()

        self.cf = self.get_conf_matrix()
        if not prank:
            self.auc = self.get_auc(predictions)
            self.predictions_for_auc = predictions
        self.accuracy = self.get_accuracy()
        self.mcc = self.get_mcc()
        self.f1 = self.get_f1()
        self.sanity_check()

    def sanity_check(self):
        assert len(self.sequence.sequence) == len(self.actual_values)
        assert len(self.sequence.sequence) == len(self.predictions)
    
    def get_conf_matrix(self):
        return metrics.confusion_matrix(self.predictions, self.actual_values)

    def get_auc(self,pred):
        return metrics.roc_auc_score(self.actual_values, pred[:,1])
    
    def get_accuracy(self):
        return metrics.accuracy_score(self.predictions, self.actual_values)
    
    def get_mcc(self):
        return metrics.matthews_corrcoef(self.predictions, self.actual_values)
    
    def get_f1(self):
        return metrics.f1_score(self.predictions, self.actual_values)
    
    def print(self):
        for i, aminoacid in enumerate(self.sequence.sequence):
            print(f"{i} {aminoacid}: {self.actual_values[i]} {self.predictions[i]}")

    def satisfies_treshold(self, treshold_percent):
        return (metrics.confusion_matrix(self.predictions, self.actual_values)[1][1] / np.sum(self.actual_values) >= treshold_percent)
    
    def get_TPR(self):
        # FP = self.cf[1][0]
        FN = self.cf[0][1]
        TP = self.cf[1][1]
        # TN = self.cf[0][0]
        # Sensitivity, hit rate, recall, or true positive rate
        return TP/(TP+FN)
    
    def get_FPR(self):
        FP = self.cf[1][0]
        # FN = self.cf[0][1]
        # TP = self.cf[1][1]
        TN = self.cf[0][0]
        # Fall out or false positive rate
        return FP/(FP+TN)
