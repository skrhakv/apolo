import numpy as np
from typing import List
import tensorflow as tf
import numpy as np
from sklearn import metrics  

class Sequence:
    def __init__(self, id, sequence, embedding):
        self.id: str = id
        # self.sequence: str = sequence        
        self.embedding: np.ndarray = embedding
        self.annotations: List[int] = []
        self.noncryptic_annotations: List[int] = []

    def add_annotations(self, annotations, is_noncryptic=False):
        # annotations: D126 Q129 K130 G133 P157 V158 P159 K162 V294 L295

        for annotation in annotations.split(' '):
            # aminoacid = annotation[:1]
            label_seq_id = int(annotation[1:])

            # assert aminoacid == self.sequence[label_seq_id - 1], f'ID {self.id}: The annotation {annotation} letter does not match the sequence at the specified position! Annotation = {aminoacid}, Sequence[{label_seq_id}] = {self.sequence[label_seq_id - 1]}'
            if not is_noncryptic:
                self.annotations.append(label_seq_id)
            else: 
                self.noncryptic_annotations.append(label_seq_id)

class Dataset:
    X_train: np.ndarray = None
    X_test: np.ndarray = None
    y_train: np.ndarray = None
    y_test: np.ndarray = None
    groups: List[int] = []

    def append_train(self, other):
        result = Dataset()
        result.X_train = np.concatenate((self.X_train, other.X_train), axis=0)
        if other.X_test == None and self.X_test == None:
            result.X_test = None
        elif len(other.X_test) > 0 and len(self.X_test) > 0:
            result.X_test = np.concatenate((self.X_test, other.X_test), axis=0)
        elif len(self.X_test) == 0:
            result.X_test = other.X_test
        else:
            result.X_test = self.X_test
        result.y_train = self.y_train + other.y_train
        if other.y_test == None and self.y_test == None:
            result.y_test = None
        else:
            result.y_test = self.y_test + other.y_test
        result.groups = self.groups + other.groups
        return result
        

class Results:
    def __init__(self, actual_values, predictions, stats, protein_code):
        self.actual_values = actual_values
        self.predictions = predictions
        self.stats = stats
        with open(f'../data/predictions/{protein_code}.txt', 'w') as f:
            for actual, pred in zip(self.actual_values, self.predictions):
                f.write(f'{actual} {pred[1]}\n')

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
        assert self.sequence.embedding.shape[0] == len(self.actual_values)
        assert self.sequence.embedding.shape[0] == len(self.predictions)
    
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
