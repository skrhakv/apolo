import tensorflow as tf
import tensorflow_addons as tfa
import keras

best_hp_model: tf.keras.Model = keras.models.load_model('/home/skrhakv/apolo/data/metadata/results-apo-esm2/models/cryptic-site-pred/best_trained')

print(len(best_hp_model.layers))
print(best_hp_model.summary())
print(best_hp_model.optimizer.get_config())

for i in best_hp_model.layers:
    try:
        print(i.rate)
    except:
        print('no dropout rate')