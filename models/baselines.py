from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Flatten


class StackedAttention:
    def __init__(self, input_dim, hidden_dim):
        self.Wv = Conv2D(input_dim, hidden_dim, kernel_size=1, padding=0)
        self.Wu = Dense(hidden_dim)
        self.Wp = Conv2D(hidden_dim, 1, kernel_size=1, padding=0)
        self.hidden_dim = hidden_dim
        self.attention_maps = None


class LstmEncoder:
    def __init__(self, token_to_idx, wordvec_dim=300, rnn_dim=256,
                 rnn_num_layers=2, rnn_dropout=0):
        self.model = None
        self.token_to_idx = token_to_idx
        self.NULL = token_to_idx['<NULL>']
        self.START = token_to_idx['<START>']
        self.END = token_to_idx['<END>']

        input_1 = Input(shape=(33,))
        x = Embedding(len(token_to_idx), wordvec_dim)(input_1)
        x = LSTM(rnn_dim, recurrent_dropout=rnn_dropout,
                 return_sequences=True)(x)

        for _ in range(rnn_num_layers - 1):
            x = LSTM(rnn_dim, recurrent_dropout=rnn_dropout)(x)
        self.model = Model(inputs=input_1, outputs=x)


class LstmModel:
    def __init__(self, vocab, rnn_wordvec_dim=300, rnn_dim=256,
                 rnn_num_layers=2, rnn_dropout=0, fc_use_batchnorm=False,
                 fc_dropout=0, fc_dims=(1024,)):
        rnn_kwargs = {
            'token_to_idx': vocab['question_token_to_idx'],
            'wordvec_dim': rnn_wordvec_dim,
            'rnn_dim': rnn_dim,
            'rnn_num_layers': rnn_num_layers,
            'rnn_dropout': rnn_dropout,
        }

        self.rnn = LstmEncoder(**rnn_kwargs)

        classifier_kwargs = {
            'rnn_model': self.rnn.model,
            'input_dim': rnn_dim,
            'hidden_dims': fc_dims,
            'output_dim': len(vocab['answer_token_to_idx']),
            'use_batchnorm': fc_use_batchnorm,
            'dropout': fc_dropout,
        }
        self.model = build_mlp(**classifier_kwargs)


class CnnLstmModel:
    def __init__(self, vocab, rnn_wordvec_dim=300, rnn_dim=256,
                 rnn_num_layers=2, rnn_dropout=0, cnn_feat_dim=(1024,14,14),
                 cnn_res_block_dim=128, cnn_num_res_blocks=0, cnn_proj_dim=512,
                 cnn_pooling='maxpool2', fc_dims=(1024,),
                 fc_use_batchnorm=False, fc_dropout=0):
        rnn_kwargs = {
            'token_to_idx': vocab['question_token_to_idx'],
            'wordvec_dim': rnn_wordvec_dim,
            'rnn_dim': rnn_dim,
            'rnn_num_layers': rnn_num_layers,
            'rnn_dropout': rnn_dropout,
        }
        self.rnn = LstmEncoder(**rnn_kwargs)

        cnn_kwargs = {
            'feat_dim': cnn_feat_dim,
            'res_block_dim': cnn_res_block_dim,
            'num_res_blocks': cnn_num_res_blocks,
            'proj_dim': cnn_proj_dim,
            'pooling': cnn_pooling,
        }
        self.cnn = build_cnn(**cnn_kwargs)

        classifier_kwargs = {
            'rnn_model': self.rnn.model,
            'cnn_model': self.cnn,
            'input_dim': self.cnn.layers[-1].output.shape[1],
            'hidden_dims': fc_dims,
            'output_dim': len(vocab['answer_token_to_idx']),
            'use_batchnorm': fc_use_batchnorm,
            'dropout': fc_dropout,
        }
        self.model = build_mlp(**classifier_kwargs)


class CnnLstmSaModel:
    def __init__(self, vocab, rnn_wordvec_dim=300, rnn_dim=256,
                 rnn_num_layers=2, rnn_dropout=0, cnn_feat_dim=(1024,14,14),
                 stacked_attn_dim=512, num_stacked_attn=2,
                 fc_use_batchnorm=False, fc_dropout=0, fc_dims=(1024,)):
        rnn_kwargs = {
            'token_to_idx': vocab['question_token_to_idx'],
            'wordvec_dim': rnn_wordvec_dim,
            'rnn_dim': rnn_dim,
            'rnn_num_layers': rnn_num_layers,
            'rnn_dropout': rnn_dropout,
        }
        self.rnn = LstmEncoder(**rnn_kwargs)

        C, H, W = cnn_feat_dim

        self.image_proj = nn.Conv2d(C, rnn_dim, kernel_size=1, padding=0)
        self.stacked_attns = []

        for i in range(num_stacked_attn):
            sa = StackedAttention(rnn_dim, stacked_attn_dim)
            self.stacked_attns.append(sa)
            self.add_module('stacked-attn-%d' % i, sa)

        classifier_args = {
            'input_dim': rnn_dim,
            'hidden_dims': fc_dims,
            'output_dim': len(vocab['answer_token_to_idx']),
            'use_batchnorm': fc_use_batchnorm,
            'dropout': fc_dropout,
        }
        self.classifier = build_mlp(**classifier_args)


def build_cnn(feat_dim=(1024, 14, 14), res_block_dim=128, num_res_blocks=0,
              proj_dim=512, pooling='maxpool2'):
    input_2 = Input(shape=(28, 28, 3))
    z = None

    if num_res_blocks > 0:
        z = Conv2D(res_block_dim, kernel_size=3, padding='valid',
                   activation='relu')(input_2)

        # residual block is being added, what is it?
    if proj_dim > 0:
        if z is not None:
            z = Conv2D(proj_dim, kernel_size=1, padding='valid',
                       activation='relu')(z)
        else:
            z = Conv2D(proj_dim, kernel_size=1, padding='valid',
                       activation='relu')(input_2)

    if pooling == 'maxpool2':
        if z is not None:
            z = MaxPooling2D(pool_size=2, strides=2)(z)
        else:
            z = MaxPooling2D(pool_size=2, strides=2)(input_2)

    res_block_dim = res_block_dim // 2
    z = Conv2D(res_block_dim, kernel_size=3, padding='valid',
               activation='relu')(z)
    z = MaxPooling2D(pool_size=2, strides=2)(z)

    if z is not None:
        z = Flatten()(z)
    else:
        z = Flatten()(input_2)

    if z is not None:
        z = Dense(256, activation='relu')(z)

    model = Model(inputs=input_2, outputs=z)
    return model


def build_mlp(input_dim, hidden_dims, output_dim, use_batchnorm=False,
              dropout=0, rnn_model=None, cnn_model=None):
    z = None
    if cnn_model is None:
        _input = rnn_model.output
    else:
        _input = concatenate([rnn_model.output, cnn_model.output])

    if dropout > 0:
        z = Dropout(dropout)(_input)

    if use_batchnorm:
        if z is None:
            z = BatchNormalization(input_dim)(_input)
        else:
            z = BatchNormalization(input_dim)(z)

    for dim in hidden_dims:
        if z is None:
            z = Dense(dim, activation='relu')(_input)
        else:
            z = Dense(dim, activation='relu')(z)

        if use_batchnorm:
            z = BatchNormalization(input_dim)(z)

        if dropout > 0:
            z = Dropout(dropout)(z)

    if z is None:
        z = Dense(output_dim)(_input)
    else:
        z = Dense(output_dim)(z)

    if cnn_model is None:
        model = Model(inputs=[rnn_model.input], outputs=z)
    else:
        model = Model(inputs=[rnn_model.input, cnn_model.input], outputs=z)
    return model
