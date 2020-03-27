import tensorflow as tf
import bert
import os
import csv
import random

# tf.test.gpu_device_name()
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print('BERT with Tensorflow 2.0')
print(f'Tf version is: {tf.__version__}')


def create_tokenizer():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    models_folder = os.path.join(current_dir, "models", "multi_cased_L-12_H-768_A-12")
    vocab_file = os.path.join(models_folder, "vocab.txt")

    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    return tokenizer


data_dir = 'data'


def preprocess_data(tokenizer):
    filename = os.path.join(data_dir, "data.csv")
    test_filename = os.path.join(data_dir, "data_test.csv")

    data = []
    data_test = []
    train_set = []
    train_labels = []
    test_set = []
    test_labels = []

    with open(filename, encoding='utf-8') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter=";")
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                data.append(row)
            line_count +=1
    csvFile.close()

    with open(test_filename, encoding='utf-8') as csvFileTest:
        csv_reader_test = csv.reader(csvFileTest, delimiter=";")
        line_count = 0
        for row in csv_reader_test:
            if line_count > 0:
                data_test.append(row)
            line_count +=1
    csvFileTest.close()

    shuffled_set = random.sample(data, len(data))
    training_set = shuffled_set[0:]
    shuffled_set_test = random.sample(data_test, len(data_test))
    testing_set = shuffled_set_test[0:]

    for el in training_set:
        train_set.append(el[1])
        zeros = [0] * classes
        zeros[int(el[0]) - 1] = 1
        train_labels.append(zeros)

    for el in testing_set:
        test_set.append(el[1])
        zeros = [0] * classes
        zeros[int(el[0]) - 1] = 1
        test_labels.append(zeros)

    # defineTokenizerConfig(train_set)

    train_tokens = map(tokenizer.tokenize, train_set)
    train_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], train_tokens)
    train_token_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))

    train_token_ids = map(lambda tids: tids + [0] * (max_seq_length - len(tids)), train_token_ids)
    train_token_ids = np.array(list(train_token_ids))

    test_tokens = map(tokenizer.tokenize, test_set)
    test_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], test_tokens)
    test_token_ids = list(map(tokenizer.convert_tokens_to_ids, test_tokens))

    test_token_ids = map(lambda tids: tids + [0] * (max_seq_length - len(tids)), test_token_ids)
    test_token_ids = np.array(list(test_token_ids))

    train_labels_final = np.array(train_labels)
    test_labels_final = np.array(test_labels)

    return train_token_ids, train_labels_final, test_token_ids, test_labels_final


# create tokenizer using BERT mode vocab
tokenizer = createTokenizer()

# preprocess data (split/shuffle etc)
train_set, train_labels, test_set, test_labels = preprocess_data(tokenizer)

