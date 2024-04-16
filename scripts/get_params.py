import tensorflow as tf
import tensorflow_addons as tfa
import keras

best_hp_model: tf.keras.Model = keras.models.load_model(f'../data/results-apo-esm2-oracle/models/cryptic-site-pred-oracle/best_trained')

print(len(best_hp_model.layers))
print(best_hp_model.summary())
print(best_hp_model.optimizer.get_config())