import os
import sys
import pandas as pd
import tensorflow as tf
import numpy as np
import keras_tuner as kt
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

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

    os.chdir('..')

    return train, valid, test

def preprocess_data(train, valid, test):
    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    bert = TFBertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

    max_length = 512
    train_encodings = tokenizer(train['essay'].to_list(), truncation=True, padding=True, max_length=max_length)
    valid_encodings = tokenizer(valid['essay'].to_list(), truncation=True, padding=True, max_length=max_length)

    train_labels = tf.constant(train['compI'], dtype=tf.float32)
    valid_labels = tf.constant(valid['compI'], dtype=tf.float32)

    return train_encodings, train_labels, valid_encodings, valid_labels, bert

class EssayHyperModel(kt.HyperModel):
    def __init__(self, bert):
      self.bert = bert

    def build(self, hp):
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        embedding = self.bert({'input_ids': input_ids})['pooler_output']

        x = Dense(1000, activation=hp.Choice('activation_l1', values=['selu', 'relu', 'sigmoid']))(embedding)
        x = Dropout(0.7)(x)
        x = Dense(2500, activation=hp.Choice('activation_l2', values=['selu', 'relu', 'sigmoid']))(x)
        x = Dropout(0.6)(x)
        x = Dense(3000, activation=hp.Choice('activation_l3', values=['selu', 'relu', 'sigmoid']))(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation='linear')(x)

        model = Model(inputs=input_ids, outputs=output)

        optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[2e-7, 2e-5, 2e-3]))
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [2, 6, 12]),
            **kwargs,
        )

def save_best_model(model, name):
    model.save('models/' + name + '.h')

def main():
    train, valid, test = read_corpus_and_split("datasets/essay")

    train_encodings, train_labels, valid_encodings, valid_labels, bert = preprocess_data(train, valid, test)

    hypermodel = EssayHyperModel(bert)

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
        epochs=10
    )

    best_model = tuner.get_best_models(1)[0]

    max_length = 512
    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    test_encodings = tokenizer(test['essay'].to_list(), truncation=True, padding=True, max_length=max_length)
    test_labels = tf.constant(test['compI'], dtype=tf.float32)

    x_test_input_ids = np.array(test_encodings['input_ids'])
    test_data = x_test_input_ids

    evaluation = best_model.evaluate(test_data, test_labels)
    print("Evaluation results:", evaluation)

    save_best_model(best_model, 'c1_reg')

if __name__ == "__main__":
    main()

