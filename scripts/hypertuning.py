import keras_tuner as kt
from sklearn.model_selection import GroupKFold
from tensorflow import one_hot
from tensorflow.keras.callbacks import EarlyStopping
from helper import get_json_config, generate_test_train_data, get_class_weights, get_config_filepath
from hypermodel import ApoloHyperModel
from sklearn import metrics
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import sklearn
import shutil, os

def f1_categorical(y, y_pred, **kwargs):
    return metrics.f1_score(y.argmax(1), y_pred.argmax(1), **kwargs)

def create_hypermodel(config_filename):
    print('load config ...')
    conf = get_json_config(config_filename)
    print('load data ...')
    
    append_features_later = conf.append_features_later
    val_dataset = generate_test_train_data(validation_folds=True, config_filename=config_filename)
    cw = get_class_weights(val_dataset.y_train)
    
    dim_embeddings = val_dataset.X_train.shape[1]

    if append_features_later:
        num_of_features = len(conf.features_paths)
        dim_embeddings = dim_embeddings - num_of_features

    project_name = conf.project_name
    tuner_directory = get_config_filepath(conf.tuner_directory)
    model_directory = get_config_filepath(conf.model_directory)
    print('load hypermodel ...')

    tuner = kt.tuners.SklearnTuner(oracle=kt.oracles.BayesianOptimizationOracle(
                                        objective=kt.Objective('score', 'max'),
                                        max_trials=10),
                            hypermodel=ApoloHyperModel(dim_embeddings=dim_embeddings, class_weight=cw, additional_features_dim=(None if not append_features_later else num_of_features), config_filename=config_filename),
                            directory=tuner_directory,    
                            scoring=metrics.make_scorer(f1_categorical),                 
                            project_name=project_name,
                            cv=GroupKFold(n_splits=4)
                            )   
    print('create folds ...')

    if append_features_later:
        X_train = [val_dataset.X_train[..., 0:val_dataset.X_train.shape[1]-num_of_features], val_dataset.X_train[..., val_dataset.X_train.shape[1]-num_of_features:val_dataset.X_train.shape[1]]]
    else:
        X_train = val_dataset.X_train
    # input_val = [val_dataset.X_test] 

    y_train = val_dataset.y_train

    stopper = EarlyStopping(monitor='mcc', min_delta=conf.early_stopping.min_delta, patience=conf.early_stopping.patience, restore_best_weights=True)
    # stop_early_val_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    print('searching for the best hyperparameters ...')

    # find the best set of hyperparameters
    tuner.search(X_train, np.array(one_hot(y_train, 2)), groups=val_dataset.groups)

    # remove the old model if exists
    if os.path.exists(model_directory) and os.path.isdir(model_directory):
        shutil.rmtree(model_directory)
    
    # save the non-fitted model
    not_fit_model = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
    not_fit_model.save(f'{model_directory}/models/{project_name}/not_fitted')

    # Build the model with the best hp in order to find the best # of epochs
    # model = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
    # print('selecting the best hyperparameters ...')
    # print(tuner.get_best_hyperparameters()[0].get_config()["values"] )
    # # Fit again with same validation split to find the best # of epochs
    # history1 = model.fit(X_train, np.array(one_hot(y_train, 2)), class_weight=cw, epochs=conf.early_stopping.max_epochs, validation_data=(input_val, one_hot(y_val, 2)), callbacks=[stopper])
    # best_epoch = history1.history['score'].index(min(history1.history['score']))
    # print('training the best hyperparameters on the best hyperparameters ...')
    # 
    train_test_dataset = generate_test_train_data(config_filename=config_filename)
    if append_features_later:
        input_train_full = [train_test_dataset.X_train[..., 0:train_test_dataset.X_train.shape[1]-num_of_features], train_test_dataset.X_train[..., train_test_dataset.X_train.shape[1]-num_of_features:train_test_dataset.X_train.shape[1]]]
    else:
        input_train_full = [train_test_dataset.X_train]

    # Then, do model fit again with all data and same number of epochs
    best_hp_model = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
    best_hp_model.fit(input_train_full, one_hot(train_test_dataset.y_train, 2), epochs=7, class_weight=cw)#, callbacks=[stopper])
    best_hp_model.save(f'{model_directory}/models/{project_name}/best_trained')
