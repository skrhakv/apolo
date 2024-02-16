import tensorflow as tf
import tensorflow_addons as tfa
import keras

best_hp_model: tf.keras.Model = keras.models.load_model(f'../data/results-apo-esm2/models/cryptic-site-pred/best_trained')

print(len(best_hp_model.layers))
best_hp_model.summary()