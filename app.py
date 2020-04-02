import tensorflow as tf
import bert
import os
import csv
import random
import numpy as np
# tf.test.gpu_device_name()
from tensorflow.python.client import device_lib
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime, os

plt.style.use('classic')
if not tf.__version__.startswith("2."):
    raise Exception('Wrong TF version')

# print(device_lib.list_local_devices())

print('BERT with Tensorflow 2.0')
print(f'Tf version is: {tf.__version__}')

current_dir = os.path.dirname(os.path.realpath(__file__))
models_folder = os.path.join(current_dir, "models", "multi_cased_L-12_H-768_A-12")
EPOCHS = 2
data_dir = 'data'
max_seq_length = 48
classes_number = 3


def create_tokenizer():
    vocab_file = os.path.join(models_folder, "vocab.txt")

    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    return tokenizer


def convert_text_into_tokens(tokenizer, sentences):
    pred_tokens = map(tokenizer.tokenize, sentences)
    pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
    pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

    pred_token_ids = map(lambda tids: tids + [0] * (max_seq_length - len(tids)), pred_token_ids)
    pred_token_ids = np.array(list(pred_token_ids))

    return pred_token_ids


def generate_samples(data_set, classes_num):
    x_set = []
    y_set = []

    for el in data_set:
        x_set.append(el[2])
        zeros = [0] * classes_num
        class_number = el[1]
        zeros[int(class_number) - 1] = 1  # onehote encode TODO: use existin onhoteencode method
        y_set.append(zeros)

    return x_set, y_set


def preprocess_data(tokenizer):
    VAL_PERC = 0.8
    filename = os.path.join(current_dir, data_dir, "DEtest.csv")
    data = []

    # TODO: maybe use Pandas?
    with open(filename, encoding='utf-8') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter=";")
        for row in csv_reader:
            data.append(row)
    csvFile.close()

    # shuffle
    data_set = shuffle(data)

    # split on train test datasets
    split_v = int(VAL_PERC * len(data_set)) + 1

    # split dataset on 80/20
    training_set, testing_set = data_set[:split_v], data_set[split_v:]

    # prepare X and Y for trainset
    x_train_set, y_train_labels = generate_samples(training_set, classes_number)

    # prepare X and Y for test
    x_test_set, y_test_labels = generate_samples(testing_set, classes_number)

    print('------------------------------Train set--------------------------------')
    print(x_train_set)
    print(y_train_labels)
    print('---------------------------------Test set ------------------------------')
    print(x_train_set)
    print(y_train_labels)

    # defineTokenizerConfig(train_set)

    # train_tokens = map(tokenizer.tokenize, train_set)
    # train_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], train_tokens)
    # train_token_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
    #
    # train_token_ids = map(lambda tids: tids + [0] * (max_seq_length - len(tids)), train_token_ids)
    train_token_ids = convert_text_into_tokens(tokenizer, x_train_set)

    # test_tokens = map(tokenizer.tokenize, test_set)
    # test_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], test_tokens)
    # test_token_ids = list(map(tokenizer.convert_tokens_to_ids, test_tokens))
    #
    # test_token_ids = map(lambda tids: tids + [0] * (max_seq_length - len(tids)), test_token_ids)
    test_token_ids = convert_text_into_tokens(tokenizer, x_test_set)

    train_labels_final = np.array(y_train_labels)
    test_labels_final = np.array(y_test_labels)

    return train_token_ids, train_labels_final, test_token_ids, test_labels_final


def create_bert_layer():
    global bert_layer

    bert_params = bert.params_from_pretrained_ckpt(models_folder)

    bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")

    # with adapter
    bert_layer.apply_adapter_freeze()


def load_bert_checkpoint():
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
    checkpoint_name = os.path.join(models_folder, "bert_faq.ckpt")

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_name,
                                                     save_weights_only=True,
                                                     verbose=1)

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    history = model.fit(
        training_set,
        training_label,
        epochs=EPOCHS,
        validation_data=(testing_set, testing_label),
        verbose=1,
        callbacks=[tensorboard_callback]
    )

    return history


# create tokenizer using BERT mode vocab test
tokenizer = create_tokenizer()
# preprocess data (split/shuffle etc)
train_set, train_labels, test_set, test_labels = preprocess_data(tokenizer)
print('------------------------------Tokenized Train set--------------------------------')

print('X:', train_set)
print('Y:', train_labels)

print('Create model')
create_bert_layer()
# load_bert_checkpoint()
createModel()

print('Start model fit ....')

fit_history = fitModel(train_set, train_labels, test_set, test_labels)

plt.title('Accuracy')
plt.plot(fit_history.history['accuracy'], color='blue', label='Train Acc')
plt.plot(fit_history.history['val_accuracy'], color='red', label='Validation Acc')
plt.legend()

_ = plt.figure()

plt.plot(fit_history.history['loss'], color='blue', label='Train Loss')
plt.plot(fit_history.history['val_loss'], color='orange', label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()

# Todo: save the model weights

# model predict

pred_sentences = [
    "Beste Arbeitsumgebung",
    "hoffe, das Management wird das Anreizsystem nicht weiter ändern",
    "netter Chef"
]

pred_token_ids = convert_text_into_tokens(tokenizer, pred_sentences)
res = model.predict(pred_token_ids)

for text, pred in zip(pred_sentences, res):
    print(" text:", text)
    print("  res:", pred)
