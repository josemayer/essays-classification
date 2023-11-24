import os
import sys
import datetime
import pandas as pd
import tensorflow as tf
import numpy as np
import keras_tuner as kt
import matplotlib as mpl
import matplotlib.pyplot as pl
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LambdaCallback, Callback, ModelCheckpoint
from config.gpu_options import gpu_config

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": "8",
})

def normalize_grades(grades):
    return [int(x / 40) for x in grades]

def join_text(text_array):
    return '\n'.join(text_array)

def read_corpus_and_split(essay_path):
    source_directory = os.path.dirname(os.path.abspath(__file__))
    essay_directory = os.path.join(source_directory, essay_path)
    os.chdir(essay_directory)

    sys.path.append(essay_directory)
    from build_dataset import Corpus
    c = Corpus()

    train, valid, test = c.read_splits()

    train['competence'] = train['competence'].apply(normalize_grades)
    train['essay'] = train['essay'].apply(join_text)
    valid['competence'] = valid['competence'].apply(normalize_grades)
    valid['essay'] = valid['essay'].apply(join_text)
    test['competence'] = test['competence'].apply(normalize_grades)
    test['essay'] = test['essay'].apply(join_text)

    train[['compI', 'compII', 'compIII', 'compIV', 'compV']] = train['competence'].apply(lambda x: pd.Series(x))
    valid[['compI', 'compII', 'compIII', 'compIV', 'compV']] = valid['competence'].apply(lambda x: pd.Series(x))
    test[['compI', 'compII', 'compIII', 'compIV', 'compV']] = test['competence'].apply(lambda x: pd.Series(x))

    os.chdir(source_directory)

    return train, valid, test

def encode_data(dataset, x, y, tokenizer, max_length = 512):
    encodings = tokenizer(dataset[x].to_list(), truncation=True, padding=True, max_length=max_length)
    labels = tf.constant(dataset[y], dtype=tf.float32)

    return encodings, labels

class EssayHyperModel(kt.HyperModel):
    def __init__(self, bert):
      self.bert = bert

    def build(self, hp):
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        embedding = self.bert({'input_ids': input_ids})['pooler_output']

        x = Dense(3000, activation=hp.Choice('activation_l1', values=['selu', 'sigmoid']))(embedding)
        x = Dropout(0.5)(x)

        x = Dense(2000, activation=hp.Choice('activation_l2', values=['selu', 'sigmoid']))(x)
        x = Dropout(0.5)(x)

        x = Dense(2500, activation=hp.Choice('activation_l3', values=['selu', 'sigmoid']))(x)
        x = Dropout(0.5)(x)

        output = Dense(1, activation='linear')(x)

        model = Model(inputs=input_ids, outputs=output)

        optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[2e-3, 2e-5, 2e-7]))
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [2, 3, 4]),
            **kwargs,
        )

def save_best_model(model, name):
    model.save('../models/' + name + '.h5')

def delete_checkpoints():
    source_directory = os.path.dirname(os.path.abspath(__file__))
    checkpoints_directory = os.path.join(source_directory, 'tuner_directory/essay_scoring')
    os.chdir(checkpoints_directory)

    for trial_dir in os.listdir():
        if os.path.isdir(trial_dir):
            os.chdir(trial_dir)
            for checkpoint in os.listdir():
                if checkpoint.endswith('.index') or checkpoint.endswith('.data-00000-of-00001'):
                    os.remove(checkpoint)
            os.chdir('..')

    os.chdir(source_directory)

def generate_plots(history, time, label):
    pl.plot(history.history['loss'])
    pl.plot(history.history['val_loss'])
    pl.title(f"Função de Perda em Treino ({label})")
    pl.ylabel('Perda')
    pl.xlabel('Época')
    pl.legend(['Treinamento', 'Validação'], loc='upper right')
    pl.savefig('logs/plots/' + time + '_loss.pgf')
    pl.clf()

def save_hps_to_log(hps, file_name):
    with open('logs/' + file_name, 'a+') as file:
        file.write("---\n")
        file.write(str(hps))

class DeleteCallback(tf.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        delete_checkpoints()

class LogCallback(tf.keras.callbacks.Callback):
    def __init__(self, logFileName):
        self.logDir = 'logs/'
        self.logFileName = logFileName
        self.best_val = {
            'epoch': 0,
            'val_loss': 9999,
        }

    def on_epoch_end(self, epoch, logs):
        if logs['val_loss'] < self.best_val['val_loss']:
            self.best_val['val_loss'] = logs['val_loss']
            self.best_val['epoch'] = epoch

    def on_train_end(self, logs):
        with open(self.logDir + self.logFileName, 'a+') as logFile:
            logFile.write(str(self.best_val) + '\n')

def main():
    tf.compat.v1.Session(config=gpu_config())

    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    bert = TFBertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

    X_label = 'essay'
    Y_label = 'compI'

    train, valid, test = read_corpus_and_split("datasets/essay")
    train_encodings, train_labels = encode_data(train, X_label, Y_label, tokenizer)
    valid_encodings, valid_labels = encode_data(valid, X_label, Y_label, tokenizer)
    test_encodings, test_labels = encode_data(test, X_label, Y_label, tokenizer)

    hypermodel = EssayHyperModel(bert)

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logFileName = time + '.log'

    tuner = kt.GridSearch(
        hypermodel,
        objective='val_loss',
        executions_per_trial=1,
        directory='tuner_directory',
        project_name='essay_scoring'
    )

    tuner.search(
        np.array(train_encodings['input_ids']),
        train_labels,
        validation_data=(np.array(valid_encodings['input_ids']), valid_labels),
        epochs=6,
        callbacks=[LogCallback(logFileName), DeleteCallback()]
    )

    best_model_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_model_hps)

    checkpoint_path = '../models/c1_reg.h5'
    history = best_model.fit(
        np.array(train_encodings['input_ids']),
        train_labels,
        validation_data=(np.array(valid_encodings['input_ids']), valid_labels),
        epochs=50,
        batch_size=best_model_hps.get('batch_size'),
        callbacks=[
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
            )
        ]
    )

    save_hps_to_log(best_model_hps.values, logFileName)
    generate_plots(history, time)

    evaluation = best_model.evaluate(np.array(test_encodings['input_ids']), test_labels)
    print("Evaluation results:", evaluation)

if __name__ == "__main__":
    main()

