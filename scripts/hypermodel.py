import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from helper import get_json_config
import tensorflow as tf

def my_loss(weight):
    def weighted_cross_entropy_with_logits(labels, logits):
        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels, logits, weight
        )
        return loss
    return weighted_cross_entropy_with_logits


class ApoloHyperModel(kt.HyperModel):
    
    def __init__(self, dim_embeddings, class_weight, name=None, tunable=True, additional_features_dim=None, config_filename=None):
        super().__init__(name, tunable)
        
        self._dim_embeddings = dim_embeddings
        self._class_weight = class_weight
        self.config = get_json_config(config_filename)
        self.additional_features_dim = additional_features_dim
        
    def build(self, hp):        
        print(f'Building the model ...')

        inputs_embedding = keras.Input(shape=(self._dim_embeddings,))
        x = layers.Dense(units=hp.Int('units', min_value=self.config.hypermodel.neuron_size_min, max_value=self.config.hypermodel.neuron_size_max, step=self.config.hypermodel.neuron_size_step), activation="relu", kernel_regularizer='l2')(inputs_embedding)
        x = layers.Dropout(hp.Float('dropout', self.config.hypermodel.dropout_min, self.config.hypermodel.dropout_max, step=self.config.hypermodel.dropout_step))(x)    

        for i in range(hp.Int('num_layers', min_value=self.config.hypermodel.number_of_layers_min, max_value=self.config.hypermodel.number_of_layers_max, step=self.config.hypermodel.number_of_layers_step)):
            # i == 1 -> 3rd layer
            # if i == 0:
            #     if self.additional_features_dim is not None:
            #         inputs = keras.Input(shape=(self.additional_features_dim,))
            #         x = layers.concatenate([x,inputs])
            x = layers.Dense(units=hp.Int(f'units_{i}', min_value=self.config.hypermodel.neuron_size_min, max_value=self.config.hypermodel.neuron_size_max, step=self.config.hypermodel.neuron_size_step), activation="relu", kernel_regularizer='l2')(x)    
            x = layers.Dropout(hp.Float(f'dropout_{i}', self.config.hypermodel.dropout_min, self.config.hypermodel.dropout_max, step=self.config.hypermodel.dropout_step))(x)            
        
        model = None

        output_layer = layers.Dense(2, activation="softmax")(x)
            
        if self.additional_features_dim is not None:
            model = keras.Model(inputs=[(inputs_embedding, inputs)], outputs=output_layer)
        else:
            model = keras.Model(inputs=inputs_embedding, outputs=output_layer)

        f1_score = tf.keras.metrics.F1Score(name='f1_score')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=self.config.hypermodel.learning_rate)),
            # loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            loss=my_loss(self._class_weight[1]),
            metrics=[f1_score, tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2, name='mcc'), keras.metrics.CategoricalAccuracy()]
            )            
            
        return model
    
    def fit(self, hp, model, *args, **kwargs):    
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", self.config.hypermodel.batch_size),
            class_weight=self._class_weight,
            **kwargs,
        )