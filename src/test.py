import os
import pandas as pd
import numpy as np
import transformers
import tensorflow as tf
from keras.models import load_model
from transformers import BertTokenizer
from sklearn.metrics import cohen_kappa_score

from train import preprocess_data, tokenize_to_input, grade_multiple_essays, calculate_accuracy, save_best_model

def main():
    source_directory = os.path.dirname(os.path.abspath(__file__))
    essay_directory = os.path.join(source_directory, "essay")
    os.chdir(essay_directory)
    sys.path.append(essay_directory)

    from build_dataset import Corpus

    c = Corpus()
    c.read_corpus().head()

    _, _, test = c.read_splits()

    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

    test_encodings, test_labels, _, _, _ = preprocess_data(None, None, test)

    essay_label = 'essay'
    comps = {
        'compI': 'c1'
    }
    current_dir = os.getcwd()
    models_path = current_dir + '/../models/'

    for comp in comps:
        file_key = comps[comp]

        x_test = test_encodings[essay_label]
        y_test = test_labels[comp]

        model = load_model(f"{models_path + file_key}_reg.h5", custom_objects={"TFBertModel": transformers.TFBertModel})

        model.summary()

        print(f"Model for {comp}")
        model_evaluation(model, x_test, y_test)
        print("")

def model_evaluation(model, x_test, y_test):
    max_length = 512

    preds = grade_multiple_essays(model, x_test)
    preds = np.squeeze(np.round(preds).astype(int))

    qwk = cohen_kappa_score(np.array(y_test), preds, weights="quadratic")
    accuracy = calculate_accuracy(np.array(y_test), preds)

    test_encodings = tokenizer(x_test.to_list(), truncation=True, padding=True, max_length=max_length)
    test_labels = tf.constant(y_test, dtype=tf.float32)

    x_test_input_ids = np.array(test_encodings['input_ids'])
    test_data = x_test_input_ids

    loss, mse = model.evaluate(test_data, test_labels, verbose=0)

    print(f"Quadratic Weighted Kappa: {qwk}")
    print(f"Mean Squared Error: {mse}")
    print(f"Accuracy: {accuracy * 100}%")
    return qwk, mse, accuracy, loss

if __name__ == "__main__":
    main()
