import tensorflow as tf
import bert
import numpy as np
import os


if not tf.__version__.startswith("2."):
    raise Exception('Wrong TF version!')

current_dir = os.path.dirname(os.path.realpath(__file__))
models_folder = os.path.join(current_dir, "models", "multi_cased_L-12_H-768_A-12")
weights_path = os.path.join(current_dir, "weights", 'bert_model_weights.h5')
max_seq_length = 47
classes = ['MANAGEMENT', 'PAYBENEFITS', 'WORKPLACE']
classes_number = len(classes)
threshold = 0.5


def create_tokenizer():
    vocab_file = os.path.join(models_folder, "vocab.txt")

    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    return tokenizer


def create_bert_layer():
    global bert_layer

    bert_params = bert.params_from_pretrained_ckpt(models_folder)

    bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")

    # with adapter
    bert_layer.apply_adapter_freeze()


def convert_text_into_tokens(tokenizer, sentences):
    pred_tokens = map(tokenizer.tokenize, sentences)
    pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
    pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

    pred_token_ids = map(lambda tids: tids + [0] * (max_seq_length - len(tids)), pred_token_ids)
    pred_token_ids = np.array(list(pred_token_ids))

    return pred_token_ids


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


    print(model.summary())


print('create tokenizer')
tokenizer = create_tokenizer()

print('create BERT layer')
create_bert_layer()

print('create model')
createModel()

print('load model weights')
model.load_weights(weights_path)

print('compile model')
model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.00001), metrics=['accuracy'])

pred_sentences = [
    "Beste Arbeitsumgebung",
    "hoffe, das Management wird das Anreizsystem nicht weiter ändern",
    "netter Chef"
]


def get_predictions():
    print('tokenize text')
    pred_token_ids = convert_text_into_tokens(tokenizer, pred_sentences)
    print('model predict')
    res = model.predict(pred_token_ids)

    print('PREDICTED RESULT:')
    for text, pred in zip(pred_sentences, res):
        print(" Text:", text)
        print(" Res:", pred)

        max_score_index = np.argmax(pred)
        score = pred[max_score_index]

        # filter trough threshold
        if threshold < score:
            pred_class = classes[max_score_index]
            print('Predicted class:', pred_class)
        else:
            print('Prediction is lower than threshold!!')
