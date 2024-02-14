import keras_tuner as kt
from sklearn.model_selection import StratifiedKFold
from tensorflow import one_hot
from tensorflow.keras.callbacks import EarlyStopping
from helper import get_json_config, generate_test_train_data, get_class_weights, get_config_filepath
from hypermodel import ApoloHyperModel

def create_hypermodel():
    print('load config ...')
    conf = get_json_config()
    print('load data ...')

    train_test_dataset = generate_test_train_data()
    cw = get_class_weights(train_test_dataset.y_train)

    dim_embeddings = train_test_dataset.X_train.shape[1]
    project_name = conf.project_name
    tuner_directory = get_config_filepath(conf.tuner_directory)
    model_directory = get_config_filepath(conf.model_directory)
    print('load hypermodel ...')

    tuner = kt.Hyperband(ApoloHyperModel(dim_embeddings=dim_embeddings, class_weight=cw),
                            objective='val_loss',
                            max_epochs=150,
                            factor=3,
                            hyperband_iterations=1,
                            seed=42,
                            directory=tuner_directory,                     
                            project_name=project_name,
                            )   
    print('create folds ...')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_index, val_index = list(skf.split(train_test_dataset.X_train, train_test_dataset.y_train))[0]
    X_train, X_val = train_test_dataset.X_train[train_index], train_test_dataset.X_train[val_index]
    input_train = [X_train]
    input_train_full = [train_test_dataset.X_train]
    input_val = [X_val] 

    y_train, y_val = train_test_dataset.y_train[train_index], train_test_dataset.y_train[val_index]

    stopper = EarlyStopping(monitor='val_loss', min_delta=conf.early_stopping.min_delta, patience=conf.early_stopping.patience, restore_best_weights=True)
    # stop_early_val_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # find the best set of hyperparameters
    tuner.search(input_train, one_hot(y_train, 2), epochs=conf.early_stopping.max_epochs, validation_data=(input_val, tf.one_hot(y_val, 2)), callbacks=[stopper])

    # save the non-fitted model
    not_fit_model = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
    not_fit_model.save(f'{model_directory}/models/{project_name}/not_fitted')

    # Build the model with the best hp in order to find the best # of epochs
    model = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])

    # Fit again with same validation split to find the best # of epochs
    history1 = model.fit(input_train, one_hot(y_train, 2), class_weight=cw, epochs=conf.early_stopping.max_epochs, validation_data=(input_val, one_hot(y_val, 2)), callbacks=[stopper])
    best_epoch = history1.history['my_sensitivity_at_specificity'].index(max(history1.history['my_sensitivity_at_specificity']))
    
    # Then, do model fit again with all data and same number of epochs
    best_hp_model = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
    best_hp_model.fit(input_train_full, one_hot(train_test_dataset.y_train, 2), epochs=best_epoch, class_weight=cw)
    best_hp_model.save(f'{model_directory}/models/{project_name}/best_trained')
