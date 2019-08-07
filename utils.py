import json
import tensorflow as tf
from models.baselines import LstmModel, CnnLstmModel, CnnLstmSaModel


def invert_dict(dictionary):
    return {value: key for key, value in dictionary.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = \
            invert_dict(vocab['question_token_to_idx'])
        vocab['program_idx_to_token'] = \
            invert_dict(vocab['program_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])

    '''Sanity check: make sure <NULL>, <START>, and <END> are consistent'''
    assert vocab['question_token_to_idx']['<NULL>'] == 0
    assert vocab['question_token_to_idx']['<START>'] == 1
    assert vocab['question_token_to_idx']['<END>'] == 2
    assert vocab['program_token_to_idx']['<NULL>'] == 0
    assert vocab['program_token_to_idx']['<START>'] == 1
    assert vocab['program_token_to_idx']['<END>'] == 2

    return vocab


def load_cpu(path):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        return saver.restore(sess, path)


def load_baseline(path):
    model_dict = {
        'LSTM': LstmModel,
        'CNN+LSTM': CnnLstmModel,
        'CNN+LSTM+SA': CnnLstmSaModel,
    }

    checkpoint = load_cpu(path)
    baseline_type = checkpoint['baseline_type']
    kwargs = checkpoint['baseline_kwargs']
    state = checkpoint['baseline_state']

    model = model_dict[baseline_type](**kwargs)
    model.load_state_dict(state)

    return model, kwargs
