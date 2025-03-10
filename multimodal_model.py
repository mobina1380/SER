
    seed_value = 42
    import os

    os.environ['PYTHONHASHSEED'] = str(seed_value)
    import random

    random.seed(seed_value)
    import numpy as np
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    np.random.seed(seed_value)
    import tensorflow as tf


    tf.random.set_seed(seed_value)

    import sys
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    import keras
    from sklearn.metrics import recall_score
    from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.preprocessing.sequence import pad_sequences


    import os
    import pickle
    import numpy as np
    from keras import Input, Model, layers, Sequential
    from keras.layers import Embedding, LSTM, Dropout, Dense, Bidirectional, Conv1D, MaxPooling1D, Flatten, \
        BatchNormalization, Concatenate, Layer, Reshape, Conv2D, MaxPool2D, SpatialDropout1D
    # from keras.utils.vis_utils import plot_model
    # from training_strategies import cold_start, pre_trained, warm_start, late_fusion, LinearW, attention
    # from plots import plot_confusion_matrix, plot_history

    emo_codes = {"A": 0, "W": 1, "H": 2, "S": 3, "N": 4, "F": 5}
    emo_labels = ["anger", "surprise", "happiness", "sadness", "neutral", "fear"]
    MODEL = 'TEST'
    # Create and save the StratifiedKFold object to the pickle file (do this once)
    seed_value = 42
    outer_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value)
    with open("files/outer_folds2.pickle", "wb") as of:
        pickle.dump(outer_folds, of)

    # Then, load it as needed
    with open("files/outer_folds2.pickle", "rb") as of:
        outer_folds = pickle.load(of)

    # outer_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value)


    def test_model(max_length, embedding_dim, embedding_matrix, units=512):
        # > SPEECH
        input_speech = Input((1, N_FEATURES))
        speech = Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation='relu')(input_speech)
        speech = MaxPooling1D(padding='same')(speech)
        speech = BatchNormalization(axis=-1)(speech)
        speech = Dropout(0.5)(speech)
        speech = Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation='relu')(speech)
        speech = MaxPooling1D(padding='same')(speech)
        speech = BatchNormalization(axis=-1)(speech)
        speech = Dropout(0.5)(speech)
        speech = Flatten()(speech)

        # > TEXT
        num_filters = 512
        filter_sizes = [3, 4, 5]

        input_text = Input(shape=(max_length,), dtype='int32')
        embedding = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], mask_zero=True, trainable=True,
                            input_length=max_length)(input_text)

        embedding = SpatialDropout1D(0.5)(embedding)

        reshape = Reshape((max_length, embedding_dim, 1))(embedding)

        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)

        maxpool_0 = MaxPool2D(pool_size=(max_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(max_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(max_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])

        flatten = Flatten()(concatenated_tensor)

        text = Dropout(0.5)(flatten)

        # > FUSION
        fusion = [speech, text]
        # model_combined = LinearW()(fusion)
        model_combined = Dense(256, activation='relu')(layers.concatenate(fusion))
        model_combined = Dense(128, activation='relu')(model_combined)
        model_combined = Dropout(0.5)(model_combined)
        model_combined = Dense(6, activation='softmax')(model_combined)

        model = Model([input_speech, input_text], model_combined)

        adam = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # plot_model(model, to_file='files/model.png', show_shapes=True)
        # print(str(model.summary()))

        return model


    def fit_model(X_speech, X_text, y, max_length, embedding_dim, embedding_matrix):
        uar_per_fold = []
        acc_per_fold = []
        loss_per_fold = []
        predicted_targets = np.array([])
        actual_targets = np.array([])
        kfold = outer_folds
        fold_no = 1
        for train, test in kfold.split(X_speech, y):
            X_train_speech = X_speech[train]
            X_test_speech = X_speech[test]

            scaler = StandardScaler()
            X_train_speech = scaler.fit_transform(X_train_speech)
            X_test_speech = scaler.transform(X_test_speech)
            X_train_speech = np.expand_dims(X_train_speech, axis=1)
            X_test_speech = np.expand_dims(X_test_speech, axis=1)

            X_train_text = X_text[train]
            X_test_text = X_text[test]

            # > STRATEGIES
            model = test_model(max_length, embedding_dim, embedding_matrix)
            best_weights_file = "files/" + MODEL + ".weights.keras"


            early_stopping_callback = keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                verbose=1,
            )

            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=best_weights_file,
                verbose=1,
                monitor='val_accuracy',
                save_best_only=True
            )

            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


            history = model.fit(
                [X_train_speech, X_train_text], y[train],
                validation_split=0.2,
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping_callback, model_checkpoint_callback,tensorboard_callback],  # اضافه کردن هر دو کال‌بک
                verbose=1
            )

            # best_weights_file = "files/" + MODEL + "_weights.h5"
            # es = EarlyStopping(monitor='val_accuracy', verbose=1, patience=10)
            # mc = ModelCheckpoint(best_weights_file, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)
            # history = model.fit(
            #     [X_train_speech, X_train_text], y[train],
            #     validation_split=0.2,
            #     epochs=100,
            #     batch_size=32,
            #     callbacks=[es, mc],
            #     verbose=1
            # )

            scores = model.evaluate([X_test_speech, X_test_text], y[test], verbose=0)
            y_pred = model.predict([X_test_speech, X_test_text])
            y_pred = np.argmax(y_pred, axis=1)
            predicted_targets = np.append(predicted_targets, y_pred)
            actual_targets = np.append(actual_targets, y[test])
            uar = recall_score(y[test], y_pred, average='macro')
            uar_per_fold.append(uar)
            acc_per_fold.append(scores[1])
            loss_per_fold.append(scores[0])
            fold_no = fold_no + 1

        for i in range(0, len(acc_per_fold)):
            print(f'> fold {i + 1} - uar: {uar_per_fold[i]} - accuracy: {acc_per_fold[i]} - loss: {loss_per_fold[i]}')
        print('____________________ RESULTS ____________________')
        print('Average scores for all folds:')
        print(f'> accuracy: {np.mean(acc_per_fold) * 100} (+- {np.std(acc_per_fold) * 100})')
        print(f'> UAR: {np.mean(uar_per_fold) * 100} (+- {np.std(uar_per_fold) * 100})')
        print(f'> loss: {np.mean(loss_per_fold)}')

        # Uncomment this if you have a plot_confusion_matrix function to visualize the results
        # plot_confusion_matrix(predicted_targets, actual_targets)

    y = np.load('files/emotions.npy')
    X_speech = np.load('files/features.npy')
    N_FEATURES = X_speech.shape[  1]
    X_text = np.load('files/new_padded_docs(10_best).npy')
    embedding_matrix = np.load('files/new_embedding_matrix(10_best).npy')
    embedding_dim = embedding_matrix.shape[1]
    max_length = len(X_text[0])
    vocab_size = len(embedding_matrix)

    N_SAMPLES = X_speech.shape[0]
    perm = np.random.permutation(N_SAMPLES)
    X_speech = X_speech[perm]
    X_text = X_text[perm]
    y = y[perm]
    print(X_text.shape)
    print(X_speech.shape)
    print(y.shape)
    embedding_matrix_shape = embedding_matrix.shape
    expected_shape = (vocab_size, embedding_dim)
    assert embedding_matrix_shape == expected_shape, f"Shape mismatch: {embedding_matrix_shape} != {expected_shape}"
    import numpy as np
    print(embedding_matrix_shape)
    print(expected_shape)

    if __name__ == '__main__':
        fit_model(X_speech, X_text, y, max_length, embedding_dim, embedding_matrix)
