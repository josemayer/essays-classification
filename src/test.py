import os
import json
import time
import pandas as pd
import numpy as np
import transformers
import tensorflow as tf
from keras.models import load_model
from transformers import BertTokenizer, TFBertModel
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from config.gpu_options import gpu_config

from train import read_corpus_and_split, encode_data

def grade_multiple_essays(model, test_encodings):
    predictions = model.predict(np.array(test_encodings['input_ids']), verbose=0)
    return np.concatenate(predictions)

def calculate_accuracy(arr_true, arr_pred):
    assert(len(arr_true) == len(arr_pred)), "Arrays do not match length"
    equals = 0
    for y_true, y_pred in zip(arr_true, arr_pred):
        if y_true == y_pred:
            equals += 1
    return equals/len(arr_true)

def calculate_discrepancy(arr_true, arr_pred):
    assert(len(arr_true) == len(arr_pred)), "Arrays do not match length"
    discrepants = 0
    for y_true, y_pred in zip(arr_true, arr_pred):
        if abs(y_true - y_pred) > 2:
            discrepants += 1
    return discrepants/len(arr_true)

def retrieve_saved_model(model_path, model_name):
    model = load_model(f"{model_path + model_name}", custom_objects={"TFBertModel": TFBertModel})
    return model

def generate_metrics(metrics):
    qwk, mse, accuracy, discrepancy, conf_matrix, loss = metrics
    model_metrics = {
        "qwk": qwk,
        "mse": mse,
        "accuracy": accuracy,
        "discrepancy": discrepancy,
        "conf_matrix": conf_matrix.tolist(),
        "loss": loss
    }
    return model_metrics

def save_evaluation(evaluation, path="metrics/"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open(path + timestamp + '.json', 'w') as fp:
        json.dump(evaluation, fp)

def main():
    tf.compat.v1.Session(config=gpu_config())

    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    bert = TFBertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

    X_label = 'essay'
    Y_labels = { 'compI': 'c1_reg.h5' }

    current_path = os.getcwd()
    saved_models_path = current_path + '/../models/'

    _, _, test = read_corpus_and_split("datasets/custom")

    evaluation = {}
    for comp in Y_labels:
        file_key = Y_labels[comp]

        test_encodings, test_labels = encode_data(test, X_label, comp, tokenizer)

        model = retrieve_saved_model(saved_models_path, file_key)

        print(f"Model for {comp} ======")
        metrics = model_evaluation(model, test_encodings, test_labels)
        evaluation[comp] = generate_metrics(metrics)
        print("")

    save_evaluation(evaluation, path=(current_path + "../metrics/"))

def model_evaluation(model, test_encodings, test_labels):
    preds = grade_multiple_essays(model, test_encodings)
    preds = np.squeeze(np.round(preds).astype(int))

    qwk = cohen_kappa_score(test_labels, preds, weights="quadratic")
    accuracy = calculate_accuracy(test_labels, preds)
    discrepancy = calculate_discrepancy(test_labels, preds)
    conf_matrix = confusion_matrix(test_labels, preds)

    loss, mse = model.evaluate(np.array(test_encodings['input_ids']), test_labels, verbose=0)

    print(f"Quadratic Weighted Kappa: {qwk}")
    print(f"Mean Squared Error: {mse}")
    print(f"Accuracy: {accuracy * 100}%")
    print(f"Discrepancy: {discrepancy * 100}%")
    print(f"Confusion Matrix: {conf_matrix}")
    return qwk, mse, accuracy, discrepancy, conf_matrix, loss

if __name__ == "__main__":
    main()
