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

def f1_categorical(y, y_pred, **kwargs):
    return metrics.f1_score(y.argmax(1), y_pred.argmax(1), average='weighted', **kwargs)

def create_hypermodel():
    print('load config ...')
    conf = get_json_config()
    print('load data ...')
    
    val_dataset = generate_test_train_data(validation_folds=True)
    cw = get_class_weights(val_dataset.y_train)
    
    dim_embeddings = val_dataset.X_train.shape[1]
    project_name = conf.project_name
    tuner_directory = get_config_filepath(conf.tuner_directory)
    model_directory = get_config_filepath(conf.model_directory)
    print('load hypermodel ...')

    tuner = kt.tuners.SklearnTuner(oracle=kt.oracles.BayesianOptimizationOracle(
                                        objective=kt.Objective('score', 'max'),
                                        max_trials=10),
                            hypermodel=ApoloHyperModel(dim_embeddings=dim_embeddings, class_weight=cw),
                            directory=tuner_directory,    
                            scoring=metrics.make_scorer(f1_categorical),                 
                            project_name=project_name,
                            cv=GroupKFold(n_splits=4)
                            )   
    print('create folds ...')

    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # train_index, val_index = list(skf.split(val_dataset.X_train, val_dataset.y_train))[0]
    # X_train, X_val = val_dataset.X_train, val_dataset.X_test
    X_train = val_dataset.X_train
    # input_val = [val_dataset.X_test] 

    y_train = val_dataset.y_train

    stopper = EarlyStopping(monitor='mcc', min_delta=conf.early_stopping.min_delta, patience=conf.early_stopping.patience, restore_best_weights=True)
    # stop_early_val_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    print('searching for the best hyperparameters ...')

    # find the best set of hyperparameters
    tuner.search(X_train, np.array(one_hot(y_train, 2)), groups=val_dataset.groups)

    # save the non-fitted model
    not_fit_model = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
    not_fit_model.save(f'{model_directory}/models/{project_name}/not_fitted')

    # Build the model with the best hp in order to find the best # of epochs
    print('selecting the best hyperparameters ...')
    # print(tuner.get_best_hyperparameters()[0].get_config()["values"] )
    # # Fit again with same validation split to find the best # of epochs

    if conf.early_stopping.finetune_epochs or conf.early_stopping.finetune_decision_threshold:
        model = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
        train_mask = [i != 0 for i in val_dataset.groups]
        val_mask = [i == 0 for i in val_dataset.groups]
        X_train = val_dataset.X_train[train_mask]
        y_train = val_dataset.y_train[train_mask]
        X_val = val_dataset.X_train[val_mask]
        y_val = val_dataset.y_train[val_mask]


    if conf.early_stopping.finetune_epochs:
        print('Finetuning number of epochs ...')
        history1 = model.fit(X_train, np.array(one_hot(y_train, 2)), class_weight=cw, epochs=conf.early_stopping.max_epochs, validation_data=(X_val, one_hot(y_val, 2)), callbacks=[stopper])
        best_epoch = history1.history['val_loss'].index(min(history1.history['val_loss']))
        print(f'Best epoch: {best_epoch}')

    if conf.early_stopping.finetune_decision_threshold:
        print('Finetuning threshold ...')

        def to_labels(pos_probs, threshold):
            return (pos_probs >= threshold).astype('int')

        model.fit(X_train, np.array(one_hot(y_train, 2)), class_weight=cw, epochs=conf.hypermodel.epochs)
        predict_val = model.predict(X_val)
        predict_val = predict_val[:, 1]
        thresholds = np.arange(0, 1, 0.05)
        
        scores = [metrics.f1_score(y_val, to_labels(predict_val, t), average='weighted') for t in thresholds]
        ix = np.argmax(scores)
        print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
        exit()

    print('training the best hyperparameters on the best hyperparameters ...')
    # 
    train_test_dataset = generate_test_train_data()
    input_train_full = [train_test_dataset.X_train]

    # Then, do model fit again with all data and same number of epochs
    best_hp_model = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
    best_hp_model.fit(input_train_full, one_hot(train_test_dataset.y_train, 2), epochs=conf.hypermodel.epochs, class_weight=cw)#, callbacks=[stopper])
    best_hp_model.save(f'{model_directory}/models/{project_name}/best_trained')
