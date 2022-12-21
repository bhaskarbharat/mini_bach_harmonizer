import time
import os
import json
import pathlib
import random
import string
import re
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization


###################################################
# Defining Notes Vocbulary Dictionary
###################################################

notes_to_int_vocab = {'[pad]': 0, '[unk]': 1, '[start]': 2, '[end]': 3}
for i, note in enumerate(range(36, 82)):
  notes_to_int_vocab[note] = i+4

int_to_notes_vocab = {j:i for i,j in notes_to_int_vocab.items()}


##########################
# GLOBAL VARIABLES
##########################

AUTOTUNE = tf.data.experimental.AUTOTUNE
transformer_weights_path = "/persistent/transformer_weights"
coconet_weights_path = "/persistent/coconet_weights"
output_json_path = "/persistent/output_json"

VOCAB_SIZE = len(notes_to_int_vocab)
SEQUENCE_LENGTH = 33
BATCH_SIZE = 1024
DROPOUT_RATE = 0.5

EMBED_DIM = 256
LATENT_DIM = 2048
NUM_HEADS = 8
LEARNING_RATE = 0.001

min_midi_pitch = 36
max_midi_pitch = 81

I = 4 # number of voices
T = 32 # length of samples (32 = two 4/4 measures)
P = max_midi_pitch - min_midi_pitch +1 # number of different pitches
hidden_size = 32


######################################
# Defining Preprocessing Functions
######################################

def notes_to_int(x):
  return [notes_to_int_vocab.get(i) for i in x]

def int_to_notes(x):
  return [int_to_notes_vocab.get(i) for i in x]

def insert_start_to_list(x):
  return ['[start]'] + x

def insert_end_to_list(x):
  return x + ['[end]']

def pad_sequences(x):
  return tf.keras.utils.pad_sequences(x, maxlen=SEQUENCE_LENGTH, padding='post')


###########################################
# Loading Transformer Model
###########################################

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        # print(f"TransformerEncoder call fn - mask = {mask}")
        if mask is not None:
            padding_mask = tf.cast(
                mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "latent_dim": self.latent_dim,
            "num_heads": self.num_heads,
        })
        return config


encoder_inputs = keras.Input(
    shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(SEQUENCE_LENGTH, VOCAB_SIZE, EMBED_DIM)(encoder_inputs)
encoder_outputs = TransformerEncoder(EMBED_DIM, LATENT_DIM, NUM_HEADS)(x)
encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(
    shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(
    shape=(None, EMBED_DIM), name="decoder_state_inputs")
x = PositionalEmbedding(SEQUENCE_LENGTH, VOCAB_SIZE, EMBED_DIM)(decoder_inputs)
x = TransformerDecoder(EMBED_DIM, LATENT_DIM, NUM_HEADS)(x, encoded_seq_inputs)
x = layers.Dropout(DROPOUT_RATE)(x)
decoder_outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(x)
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)


decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = keras.Model(
    [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
)

optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
# transformer.summary()
transformer.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


transformer.load_weights(os.path.join(transformer_weights_path, "keras_transformer_model_weights.ckpt"))




class Translator(tf.Module):
  def __init__(self, tokenizers, transformer):
    self.tokenizers = tokenizers
    self.transformer = transformer

  def input_preprocessing(self, X):
    X = notes_to_int(X)
    X = pad_sequences([X])
    # print(f"X.shape: {X.shape}")
    # print(X)
    return X

  def output_preprocessing(self, Y):
    Y = notes_to_int(Y)
    Y = pad_sequences([Y])
    # Y = tf.convert_to_tensor([Y])
    # print(f"Y.shape: {Y.shape}")
    # print(Y)
    return Y

  def __call__(self, sentence, max_length=SEQUENCE_LENGTH-1):
    # The input sentence is Portuguese, hence adding the `[START]` and `[END]` tokens.
    decoded_sentence = ["[start]"]
    tokenized_input_sentence = self.input_preprocessing(sentence)
    for i in range(max_length):
      tokenized_target_sentence = self.output_preprocessing(decoded_sentence)
      predictions = self.transformer([tokenized_input_sentence, tokenized_target_sentence])

      sampled_token_index = np.argmax(predictions[0, i, :])
      if sampled_token_index < 4:
        idxs_of_sorted_array = np.argsort(predictions, axis=-1)
        counter = -1
        sampled_token_index = idxs_of_sorted_array[counter]
        while sampled_token_index < 4:
          counter -= 1
          sampled_token_index = idxs_of_sorted_array[counter]
        # sampled_token = int_to_notes_vocab[sampled_token_index]

      sampled_token = self.tokenizers[sampled_token_index]
      decoded_sentence.append(sampled_token)

    return decoded_sentence[1:]


def print_translation(sentence, tokens, ground_truth):
  print(f'{"Input:":15s}: {sentence}')
  print(f'{"Prediction":15s}: {tokens}')
  print(f'{"Ground truth":15s}: {ground_truth}')


translator = Translator(int_to_notes_vocab, transformer)




##################################
# Loading Coconet Model
##################################

# harmonize a melody
def harmonize(y, C, model):
    """
    Generate an artificial Bach Chorale starting with y, and keeping the pitches where C==1.
    Here C is an array of shape (4, 32) whose entries are 0 and 1.
    The pitches outside of C are repeatedly resampled to generate new values.
    For example, to harmonize the soprano line, let y be random except y[0] contains the soprano line, let C[1:] be 0 and C[0] be 1.
    """
    # model.eval()
    # with torch.no_grad():
    x = y
    C2 = C.copy()
    num_steps = int(2*I*T)
    alpha_max = .999
    alpha_min = .001
    eta = 3/4
    for i in range(num_steps):
        p = np.maximum(alpha_min, alpha_max - i*(alpha_max-alpha_min)/(eta*num_steps))
        sampled_binaries = np.random.choice(2, size = C.shape, p=[p, 1-p])
        C2 += sampled_binaries
        C2[C==1] = 1
        x_cache = x
        x = model.pred(x, C2)
        x[C2==1] = x_cache[C2==1]
        C2 = C.copy()
    return x


class Unit(tf.keras.Model):
    """
    Two convolution layers each followed by batchnorm and relu, plus a residual connection.
    """

    def __init__(self):
        super(Unit, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(hidden_size, 3, padding='same')
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(hidden_size, 3, padding='same')
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

    def call(self, x, training=None, mask=None):
        # y = x
        y = self.conv1(x)
        y = self.batchnorm1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.batchnorm2(y)
        y = y + x
        y = self.relu2(y)
        return y


class Net(tf.keras.Model):
    """
    A CNN that where you input a starter chorale and a mask and it outputs a prediction for the values
    in the starter chorale away from the mask that are most like the training data.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.initial_conv = tf.keras.layers.Conv2D(
            hidden_size, 3, padding='same')
        self.initial_batchnorm = tf.keras.layers.BatchNormalization()
        self.initial_relu = tf.keras.layers.ReLU()
        self.unit1 = Unit()
        self.unit2 = Unit()
        self.unit3 = Unit()
        self.unit4 = Unit()
        self.unit5 = Unit()
        self.unit6 = Unit()
        self.unit7 = Unit()
        self.unit8 = Unit()
        self.unit9 = Unit()
        self.unit10 = Unit()
        self.unit11 = Unit()
        self.unit12 = Unit()
        self.unit13 = Unit()
        self.unit14 = Unit()
        self.unit15 = Unit()
        self.unit16 = Unit()
        self.affine = tf.keras.layers.Dense(I*T*P, activation=None)

    def forward(self, x, C):
        # x is a tensor of shape (N, I, T, P)
        # C is a tensor of 0s and 1s of shape (N, I, T)
        # returns a tensor of shape (N, I, T, P)

        # get the number of batches
        N = x.shape[0]

        # tile the array C out of a tensor of shape (N, I, T, P)
        tiled_C = tf.reshape(C, shape=(N, I, T, 1))
        tiled_C = tf.repeat(tiled_C, P, axis=3)

        # mask x and combine it with the mask to produce a tensor of shape (N, 2*I, T, P)
        y = tf.keras.layers.Concatenate(axis=1)([tiled_C*x, tiled_C])

        # apply the convolution and relu layers
        y = self.initial_conv(y)
        y = self.initial_batchnorm(y)
        y = self.initial_relu(y)
        y = self.unit1(y)
        y = self.unit2(y)
        y = self.unit3(y)
        y = self.unit4(y)
        y = self.unit5(y)
        y = self.unit6(y)
        y = self.unit7(y)
        y = self.unit8(y)
        y = self.unit9(y)
        y = self.unit10(y)
        y = self.unit11(y)
        y = self.unit12(y)
        y = self.unit13(y)
        y = self.unit14(y)
        y = self.unit15(y)
        y = self.unit16(y)

        # reshape before applying the fully connected layer
        # y = tf.reshape(y, shape=(N, hidden_size*T*P))
        y = tf.reshape(y, shape=(N, -1))
        y = self.affine(y)

        # reshape to (N, I, T, P)
        y = tf.reshape(y, shape=(N, I, T, P))

        return y

    def pred(self, y, C):
        # y is an array of shape (I, T) with integer entries in [0, P)
        # C is an array of shape (I, T) consisting of 0s and 1s
        # the entries of y away from the support of C should be considered 'unknown'

        # x is shape (I, T, P) one-hot representation of y
        compressed = y.reshape(-1)
        x = np.zeros((I*T, P))
        r = np.arange(I*T)
        x[r, compressed] = 1
        x = x.reshape(I, T, P)

        # prep x and C for the plugging into the model
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x = tf.reshape(x, shape=(1, I, T, P))
        C2 = tf.convert_to_tensor(C, dtype=tf.float32)
        C2 = tf.reshape(C2, shape=(1, I, T))

        # plug x and C2 into the model
        # with torch.no_grad():
        out = self.forward(x, C2)
        out = tf.reshape(out, shape=(I, T, P)).numpy()
        out = out.transpose(2, 0, 1)  # shape (P, I, T)
        probs = np.exp(out) / np.exp(out).sum(axis=0)  # shape (P, I, T)
        cum_probs = np.cumsum(probs, axis=0)  # shape (P, I, T)
        u = np.random.rand(I, T)  # shape (I, T)
        return np.argmax(cum_probs > u, axis=0)


coconet = Net()

coconet.load_weights(os.path.join(coconet_weights_path, "my_model"))




############################################################
# Loading & Preprocessing input.json Received From Frontend
############################################################

def load_preprocess_input_from_path(inp):
    # with open(input_path, "rb") as f:
    #     inp = json.load(f)


    def extract_input_notes(x):
        sop = []
        alt = []
        ten = []
        bas = []
        notes_dict_list = x['notes']
        for notes_dict in notes_dict_list:
            note = notes_dict['pitch']
            quantized_start_step = notes_dict['quantizedStartStep']
            instrument = int(notes_dict['instrument'])
            if instrument == 0:
                sop.append((note, quantized_start_step))
            elif instrument == 1:
                alt.append((note, quantized_start_step))
            elif instrument == 2:
                ten.append((note, quantized_start_step))
            else:
                bas.append((note, quantized_start_step))
        return sop, alt, ten, bas
    
    sop, alt, ten, bas = extract_input_notes(inp)
    sop = sorted(sop, key=lambda x: x[1])
    alt = sorted(alt, key=lambda x: x[1])
    ten = sorted(ten, key=lambda x: x[1])
    bas = sorted(bas, key=lambda x: x[1])

    ################################################################
    # Testing Coconet Inference for overlapping inputs & missing
    # initial quantizedStartStep
    final_input_to_model = np.random.randint(P, size=(I, T))
    # final_input_to_model[:, :] = min_midi_pitch
    for n, q in sop:
        final_input_to_model[0, q] = n
    for n, q in alt:
        final_input_to_model[1, q] = n
    for n, q in ten:
        final_input_to_model[2, q] = n
    for n, q in bas:
        final_input_to_model[3, q] = n
    # final_input_to_model = list(final_input_to_model)
    ################################################################

    # final_input_to_model = []
    # for n, q in input_to_model:
    #     final_input_to_model.append(n)

    return final_input_to_model, sop, alt, ten, bas




################################################################
# Formatting the generated harmony in the output.json format
# as this is expected by frontend
#################################################################

def export_output_json(final_arr, output_json, input_json):
    notes_dict_list = []
    notes_dict_keys = ['pitch', 'instrument', 'quantizedStartStep', 'quantizedEndStep']
    instrument_idxs = [3, 2, 1, 0]
    quantized_step = 0
    for i in range(T):
        instrument_notes_from_final_arr = final_arr[:, i]
        for instrument_idx in instrument_idxs:
            notes_dict = {'pitch': int(instrument_notes_from_final_arr[instrument_idx]),
                        'instrument': int(instrument_idx),
                        'quantizedStartStep': int(i),
                        'quantizedEndStep': int(i+1)}
            notes_dict_list.append(notes_dict)

    #######################################################################
    # Testing Coconet Inference for overlapping inputs & missing
    # initial quantizedStartStep
    for ele in input_json['notes']:
        notes_dict_list.append(ele)
    ########################################################################
  
    output_json['notes'] = notes_dict_list
    return output_json




########################################
# Predicting Harmonies
########################################

def make_prediction(sequence):
    
    # Load & preprocess Input
    input_sequence, sop, alt, ten, bas = load_preprocess_input_from_path(sequence)
    ground_truth = [] # We don't know the Ground Truth for the received input

    # # Transformer model is called to complete the incomplete input_sequence
    # # received from frontend
    # decoded_sequence = translator(input_sequence)

    # # Printing the input_sequence, decoded_sequence and ground_truth
    # print_translation(input_sequence, decoded_sequence, ground_truth)

    # Coconet model is called with the decoded_sequence to generate
    # the final harmony
    # y = np.random.randint(P, size=(I, T))

    ##################################################################
    # Testing Coconet Inference for overlapping inputs & missing
    # initial quantizedStartStep
    y = input_sequence - min_midi_pitch
    D = np.zeros((I, T)).astype(int)
    for n, q in sop:
        D[0, q] = 1
    for n, q in alt:
        D[1, q] = 1
    for n, q in ten:
        D[2, q] = 1
    for n, q in bas:
        D[3, q] = 1
    ##################################################################
    
    # y[0] = np.array(decoded_sequence) - min_midi_pitch
    # D0 = np.ones((1, T)).astype(int)
    # D1 = np.zeros((3, T)).astype(int)
    # D = np.concatenate([D0, D1], axis=0)
    arr = harmonize(y, D, coconet)

    harmony_arr = []
    for i in range(arr.shape[0]):
        harmony_arr.append(int_to_notes(arr[i, :] + 4))
    harmony_arr = np.array(harmony_arr)
    print(f"Harmonized Array: {harmony_arr}")

    # Loading output.json format as this is expected by frontend
    with open(os.path.join(output_json_path, "output.json"), "rb") as f:
        out = json.load(f)
    
    # Formatting the generated harmony in the output.json format
    output_json = export_output_json(harmony_arr, out, sequence)
    print(output_json)
    print(type(output_json))

    return output_json