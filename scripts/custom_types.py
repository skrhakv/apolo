import numpy as np
from typing import List

class Sequence:
    def __init__(self, id, sequence, embedding):
        self.id: str = id
        self.sequence: str = sequence        
        self.embedding: np.ndarray = embedding
        self.annotations: List[int] = []
    
    def add_annotations(self, pocket_def):
        # pocket_def: D4 D48 V115 T116 N118 V120 G121 V122
        self.annotations = [int(res[1:]) for res in pocket_def.split(' ')]

class Dataset:
    X_train: np.ndarray = None
    X_test: np.ndarray = None
    y_train: np.ndarray = None
    y_test: np.ndarray = None
