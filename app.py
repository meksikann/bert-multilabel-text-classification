import tensorflow as tf
import bert
import os

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
