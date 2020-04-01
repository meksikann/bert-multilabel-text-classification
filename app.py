import tensorflow as tf
import bert
import os
import csv
import random
import numpy as np
# tf.test.gpu_device_name()
from tensorflow.python.client import device_lib

# print(device_lib.list_local_devices())
print('BERT with Tensorflow 2.0')
print(f'Tf version is: {tf.__version__}')

current_dir = os.path.dirname(os.path.realpath(__file__))


def create_tokenizer():

    models_folder = os.path.join(current_dir, "models", "multi_cased_L-12_H-768_A-12")
    vocab_file = os.path.join(models_folder, "vocab.txt")

    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    return tokenizer


data_dir = 'data'
max_seq_length = 48
classes_number = 3


def preprocess_data(tokenizer):
    filename = os.path.join(current_dir, data_dir, "DEtest.csv")
    test_filename = os.path.join(current_dir, data_dir, "data_test.csv")

    data = []
    data_test = []
    train_set = []
    train_labels = []
    test_set = []
    test_labels = []

    # TODO: maybe use Pandas?
    with open(filename, encoding='utf-8') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter=";")
        for row in csv_reader:
            data.append(row)
    csvFile.close()

    # TODO: maybe use Pandas?
    # TODO: make separate validation set
    # with open(test_filename, encoding='utf-8') as csvFileTest:
    #     csv_reader_test = csv.reader(csvFileTest, delimiter=";")
    #     line_count = 0
    #     for row in csv_reader_test:
    #         if line_count > 0:
    #             data_test.append(row)
    #         line_count +=1
    # csvFileTest.close()

    shuffled_set = random.sample(data, len(data))
    training_set = shuffled_set[0:]
    shuffled_set_test = random.sample(data_test, len(data_test))
    testing_set = shuffled_set_test[0:]

    for el in training_set:
        train_set.append(el[2])
        zeros = [0] * classes_number
        class_number = el[1]
        zeros[int(class_number) - 1] = 1  # onehote encode TODO: use existin onhoteencode method
        train_labels.append(zeros)

    for el in testing_set:
        test_set.append(el[1])
        zeros = [0] * classes_number
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


def create_bert_layer():
    global bert_layer

    bert_dir = os.path.join(modelBertDir, "multi_cased_L-12_H-768_A-12")

    bert_params = bert.params_from_pretrained_ckpt(bert_dir)

    bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")

    # with adapter
    bert_layer.apply_adapter_freeze()


def load_bert_checkpoint():
    models_folder = os.path.join(modelBertDir, "multi_cased_L-12_H-768_A-12")
    checkpoint_name = os.path.join(models_folder, "bert_model.ckpt")

    bert.load_stock_weights(bert_layer, checkpoint_name)


#  Add dense layers after my BERT embedded layer with 256 neurons each.


def createModel():
    global model

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='input_ids'),
        bert_layer,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(classes_number, activation=tf.nn.softmax)
    ])

    model.build(input_shape=(None, max_seq_length))

    model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.00001), metrics=['accuracy'])

    print(model.summary())


def fitModel(training_set, training_label, testing_set, testing_label):
    checkpoint_name = os.path.join(modelDir, "bert_faq.ckpt")

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_name,
                                                     save_weights_only=True,
                                                     verbose=1)

    # callback = StopTrainingClassComplete()

    history = model.fit(
        training_set,
        training_label,
        epochs=300,
        validation_data=(testing_set, testing_label),
        verbose=1,
        callbacks=[cp_callback]
    )


# create tokenizer using BERT mode vocab test
tokenizer = create_tokenizer()
# preprocess data (split/shuffle etc)
train_set, train_labels, test_set, test_labels = preprocess_data(tokenizer)
print('X:', train_set)
print('Y:', train_labels)
