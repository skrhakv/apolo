import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from helper import get_json_config
import tensorflow as tf

class ApoloHyperModel(kt.HyperModel):
    
    def __init__(self, dim_embeddings, class_weight, name=None, tunable=True):
        super().__init__(name, tunable)
        
        self._dim_embeddings = dim_embeddings
        self._class_weight = class_weight
        self.config = get_json_config()
        
    def build(self, hp):        
        
        inputs_embedding = keras.Input(shape=(self._dim_embeddings,))
        x = layers.Dense(units=hp.Int('units', min_value=self.config.hypermodel.neuron_size_min, max_value=self.config.hypermodel.neuron_size_max, step=self.config.hypermodel.neuron_size_step), activation="relu", kernel_regularizer='l2')(inputs_embedding)
        x = layers.Dropout(hp.Float('dropout', self.config.hypermodel.dropout_min, self.config.hypermodel.dropout_max, step=self.config.hypermodel.dropout_step))(x)    
        for i in range(hp.Int('num_layers', min_value=self.config.hypermodel.number_of_layers_min, max_value=self.config.hypermodel.number_of_layers_max, step=self.config.hypermodel.number_of_layers_step)):
            x = layers.Dense(units=hp.Int(f'units_{i}', min_value=self.config.hypermodel.neuron_size_min, max_value=self.config.hypermodel.neuron_size_max, step=self.config.hypermodel.neuron_size_step), activation="relu", kernel_regularizer='l2')(x)    
            x = layers.Dropout(hp.Float(f'dropout_{i}', self.config.hypermodel.dropout_min, self.config.hypermodel.dropout_max, step=self.config.hypermodel.dropout_step))(x)            
        
        model = None
        
        inputs = inputs_embedding
        output_layer = layers.Dense(2, activation="softmax")(x)
            
        model = keras.Model(inputs=inputs, outputs=output_layer)
        auc = tf.keras.metrics.AUC()
        sensitivity_at_specificity = tf.keras.metrics.SensitivityAtSpecificity(specificity=0.9, name='my_sensitivity_at_specificity')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=self.config.hypermodel.learning_rate)),
            loss=keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=[keras.metrics.CategoricalAccuracy(), auc, sensitivity_at_specificity]
            )            
            
        return model
    
    def fit(self, hp, model, *args, **kwargs):    
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", self.config.hypermodel.batch_size),
            class_weight=self._class_weight,
            **kwargs,
        )