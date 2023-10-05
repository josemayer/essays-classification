import os
import sys
import evaluate
import pandas as pd
import tensorflow as tf
import numpy as np
import keras_tuner as kt
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TrainingArguments, Trainer
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from config.gpu_options import gpu_config

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

    train, valid, test = c.read_splits(single_df = True)

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

def encode_data(data):
    tokenizer = AutoTokenizer.from_pretrained('PORTULAN/albertina-ptbr')
    return tokenizer(data["text"], truncation=True, padding=True)

def save_best_model(model, name):
    model.save('../models/' + name + '.h5')

def compute_metrics(eval_preds):
    metric = evaluate.load("mse")
    logits, labels = eval_preds
    predictions = np.round(logits)
    return metric.compute(predictions=predictions, references=labels)

def main():
    tf.compat.v1.Session(config=gpu_config())

    bert = TFAutoModelForSequenceClassification.from_pretrained('PORTULAN/albertina-ptbr', num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained('PORTULAN/albertina-ptbr')

    X_label = 'essay'
    Y_label = 'compI'

    train, valid, test = read_corpus_and_split("datasets/custom")

    train = train[['essay', 'c1']]
    valid = valid[['essay', 'c1']]
    test = test[['essay', 'c1']]

    train = train.rename(columns={"essay": "text", "c1": "label"})
    valid = valid.rename(columns={"essay": "text", "c1": "label"})
    test = test.rename(columns={"essay": "text", "c1": "label"})

    tokenized_train = train.map(encode_data, batched=True)
    tokenized_valid = valid.map(encode_data, batched=True)
    tokenized_test = test.map(encode_data, batched=True)

    training_args = TrainingArguments(
        output_dir="essay_classification_out",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=bert,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )


if __name__ == "__main__":
    main()

