from bs4 import BeautifulSoup as bs
from itertools import chain
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_text as tf_text
import nltk
from nltk import word_tokenize
import joblib
import matplotlib.pyplot as plt
import logging
import os
import shutil
from time import time

# Basic config
nltk.download("punkt")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.basicConfig(
    filename="nmt.log",
    filemode="a",
    format="%(asctime)s - %(message)s",
    level=logging.DEBUG,
)

# Verify if GPU is recognized
gpu_device_name = tf.test.gpu_device_name()
assert gpu_device_name, "GPU not recognized"
logging.info(f"GPU device name: {gpu_device_name}")

"""
Data
"""

# Open XML file and extract segments from all "documents" (TED talks)
def extract_segments(filepath, train=False):
    with open(filepath, mode="r", encoding="utf-8") as f:
        raw = f.read()

    xml = bs(raw, "lxml")
    docs: list = xml.findAll("doc")

    if train:
        texts: list = [doc.findAll(text=True, recursive=False) for doc in docs]
        texts = list(chain(*texts))
        texts = [text.strip() for text in texts if len(text.strip()) > 0]
        segments: list = [text.split("\n") for text in texts]
    else:
        segments = [doc.findAll("seg") for doc in docs]
        segments = list(chain(*segments))
        segments = [seg.findAll(text=True, recursive=False)
                    for seg in segments]

    segments = list(chain(*segments))
    segments = [seg.strip() for seg in segments]

    return segments

DATA_DIR = "data-nmt/"

train_br = extract_segments(DATA_DIR + "train.tags.en-br.br", train=True)
train_en = extract_segments(DATA_DIR + "train.tags.en-br.en", train=True)
dev_br = extract_segments(DATA_DIR + "dIWSLT17.TED.dev2010.en-br.br.xml")
dev_en = extract_segments(DATA_DIR + "dIWSLT17.TED.dev2010.en-br.en.xml")
test_br = extract_segments(DATA_DIR + "IWSLT17.TED.tst2010.en-br.br.xml")
test_en = extract_segments(DATA_DIR + "IWSLT17.TED.tst2010.en-br.en.xml")


def sets_to_df(br, en): return pd.DataFrame({"br": br, "en": en})

nmt_train_set = sets_to_df(train_br, train_en)
nmt_dev_set = sets_to_df(dev_br, dev_en)
nmt_test_set = sets_to_df(test_br, test_en)

for name, df in zip(["train", "dev", "test"], [nmt_train_set, nmt_dev_set, nmt_test_set]):
    logging.info(f"{name} set: {len(df)} segments")

logging.info("Finished collecting data.")
logging.info("Training set sample:")
logging.info(nmt_train_set.sample(5))

exit()

"""
Constants
"""

SOURCE_VOCAB_SIZE = 50_000
TARGET_VOCAB_SIZE = 50_000
MINIBATCH_SIZE = 64
SEQ_LENGTH = 50
MAX_WORD_LENGTH = 35
CHARS = True
UNK_FILTER = 0

MAX_VOCAB_SIZE = 50_000  # TODO: should be 50_000
EMBEDDING_DIM = 500  # TODO: should be 500
UNITS = 256  # TODO: not sure how many
BATCH_SIZE = 8

DEBUG = False

# Special tokens
PAD = "<blank>"
UNK = "[UNK]"
BOS = "[START]"
EOS = "[END]"

"""
Pre-processing
"""

BUFFER_SIZE = len(nmt_train_set)

# WILL DEPEND
input, target = nmt_train_set["br"].values, nmt_train_set["en"].values

# if DEBUG: # Reduce dataset
# input = input[:1_000]
# target = target[:1_000]

train_dataset = tf.data.Dataset.from_tensor_slices(
    (input, target)).shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

# Standardization


def standardize(text):
    # Split accecented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join([BOS, text, EOS], separator=' ')
    return text


input_text_processor = tf.keras.layers.TextVectorization(
    standardize=standardize,
    max_tokens=MAX_VOCAB_SIZE
)

output_text_processor = tf.keras.layers.TextVectorization(
    standardize=standardize,
    max_tokens=MAX_VOCAB_SIZE
)

input_text_processor.adapt(input)
output_text_processor.adapt(target)

print(input_text_processor.get_vocabulary()[:15])
print(output_text_processor.get_vocabulary()[:15])

# example = "Você ainda tá em casa?"
# print(standardize(example).numpy().decode())
# example_tokens = input_text_processor(example)
# print(example_tokens)

"""
ShapeChecker
"""
class ShapeChecker():
    def __init__(self):
        # Keep a cache of every axis-name seen
        self.shapes = {}

    def __call__(self, tensor, names, broadcast=False):
        if not tf.executing_eagerly():
            return

        if isinstance(names, str):
            names = (names,)

        shape = tf.shape(tensor)
        rank = tf.rank(tensor)

        if rank != len(names):
            raise ValueError(f'Rank mismatch:\n'
                f'    found {rank}: {shape.numpy()}\n'
                f'    expected {len(names)}: {names}\n'
            )

        for i, name in enumerate(names):
            if isinstance(name, int):
                old_dim = name
            else:
                old_dim = self.shapes.get(name, None)
            new_dim = shape[i]

            if (broadcast and new_dim == 1):
                continue

            if old_dim is None:
                # If the axis name is new, add its length to the cache.
                self.shapes[name] = new_dim
                continue

            if new_dim != old_dim:
                raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                    f"    found: {new_dim}\n"
                    f"    expected: {old_dim}\n"
                )

"""
Encoder
"""
class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.input_vocab_size = input_vocab_size
        self.embedding_dim = embedding_dim  # !

        # The embedding layer converts tokens to vectors
        self.embedding = tf.keras.layers.Embedding(
            self.input_vocab_size, embedding_dim)

        # The GRU RNN layer processes those vectors sequentially.
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

        self.lstm = tf.keras.layers.LSTM(
            self.enc_units,
            return_sequences=True,
            return_state=True,
        )

    def call(self, tokens, state=None):
        _history = []
        shape_checker = ShapeChecker()
        shape_checker(tokens, ('batch', 's'))

        # 2. The embedding layer looks up the embedding for each token.
        vectors = self.embedding(tokens)
        shape_checker(vectors, ('batch', 's', 'embed_dim'))
        _history.append(("encoder_embedding", vectors))

        """
        # 3. The GRU processes the embedding sequence.
        #    output shape: (batch, s, enc_units)
        #    state shape: (batch, enc_units)
        output, state = self.gru(vectors, initial_state=state)
        _history.append(("encoder_gru_1", output))
        """

        # 3. LSTM layers
        N_LSTM_LAYERS = 2
        for i in range(N_LSTM_LAYERS):
            initial_state = None if i == 0 else [hidden_state, cell_state]
            output, hidden_state, cell_state = self.lstm(
                vectors, initial_state=initial_state)
            shape_checker(output, ('batch', 's', 'enc_units'))
            shape_checker(hidden_state, ('batch', 'enc_units'))
            shape_checker(cell_state, ('batch', 'enc_units'))
            _history.append((f"encoder_lstm_{i+1}", output))

        state = hidden_state  # TODO: should be hidden_state + cell_state?

        # 4. Returns the new sequence and its state.
        return _history, output, state

# Test the encoder

# for example_input_batch, example_target_batch in train_dataset.take(1):
#     print("example_input_batch:", example_input_batch[:3])
#     print("example_target_batch:", example_target_batch[:3])

# example_tokens = input_text_processor(example_input_batch)
# encoder = Encoder(input_text_processor.vocabulary_size(), EMBEDDING_DIM, UNITS)
# history, example_enc_output, example_enc_state = encoder(example_tokens)

# lstm_outputs = [vec for (name, vec) in history if "lstm" in name]
# assert not np.allclose(
#     lstm_outputs[0], lstm_outputs[1]), "LSTM outputs too similar"  # TODO
# assert example_enc_output is not None, "Encoder output was not returned"
# assert example_enc_state is not None, "Encoder state was not returned"

# print("history:")
# for (name, vec) in history:
#     print(f"{name}: {tuple(vec.shape)}")

# print(f'Input batch, shape (batch): {example_input_batch.shape}')
# print(f'Input batch tokens, shape (batch, s): {example_tokens.shape}')
# print(f'Encoder output, shape (batch, s, units): {example_enc_output.shape}')
# print(f'Encoder state, shape (batch, units): {example_enc_state.shape}')

"""
Attention head
"""

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        # For Eqn. (4), the  Bahdanau attention
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)

        self.attention = tf.keras.layers.AdditiveAttention()
        # TODO: check what kind of attention the paper used

    def call(self, query, value, mask):
        shape_checker = ShapeChecker()
        shape_checker(query, ('batch', 't', 'query_units'))
        shape_checker(value, ('batch', 's', 'value_units'))
        shape_checker(mask, ('batch', 's'))

        # From Eqn. (4), `W1@ht`.
        w1_query = self.W1(query)
        shape_checker(w1_query, ('batch', 't', 'attn_units'))

        # From Eqn. (4), `W2@hs`.
        w2_key = self.W2(value)
        shape_checker(w2_key, ('batch', 's', 'attn_units'))

        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        context_vector, attention_weights = self.attention(
            inputs=[w1_query, value, w2_key],
            mask=[query_mask, value_mask],
            return_attention_scores=True,
        )

        shape_checker(context_vector, ('batch', 't', 'value_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))

        return context_vector, attention_weights

# Example query to test the attention layer
# attention_layer = BahdanauAttention(UNITS)
# example_attention_query = tf.random.normal(shape=(len(example_tokens), 2, 10))

# # Attent to the encoded tokens
# context_vector, attention_weights = attention_layer(
#     query=example_attention_query,
#     value=example_enc_output,
#     mask=(example_tokens != 0)
# )

# print(
#     f'Attention result shape: (batch_size, query_seq_length, units): {context_vector.shape}')
# print(
    # f'Attention weights shape: (batch_size, query_seq_length, value_seq_length): {attention_weights.shape}')

"""
Decoder
"""


class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim

        # For Step 1. The embedding layer convets token IDs to vectors
        self.embedding = tf.keras.layers.Embedding(
            self.output_vocab_size, embedding_dim)

        # For Step 2. The RNN keeps track of what's been generated so far.
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

        # For step 3. The RNN output will be the query for the attention layer.
        self.attention = BahdanauAttention(self.dec_units)

        # For step 4. Eqn. (3): converting `ct` to `at`
        self.Wc = tf.keras.layers.Dense(
            dec_units,
            activation=tf.math.tanh,
            use_bias=False
        )

        # For step 5. This fully connected layer produces the logits for each
        # output token.
        self.fc = tf.keras.layers.Dense(self.output_vocab_size)

    def call(self, new_tokens, enc_output, mask, state):
        shape_checker = ShapeChecker()
        shape_checker(new_tokens, ('batch', 't'))
        shape_checker(enc_output, ('batch', 's', 'enc_units'))
        shape_checker(mask, ('batch', 's'))

        if state is not None:
            shape_checker(state, ('batch', 'dec_units'))

        _history = []

        # Step 1. Lookup the embeddings
        vectors = self.embedding(new_tokens)
        shape_checker(vectors, ('batch', 't', 'embedding_dim'))
        _history.append(("decoder_embedding", vectors))

        # Step 2. Process one step with the RNN
        rnn_output, state = self.gru(vectors, initial_state=state)

        shape_checker(rnn_output, ('batch', 't', 'dec_units'))
        shape_checker(state, ('batch', 'dec_units'))
        _history.append(("decoder_gru_1", rnn_output))

        # Step 3. Use the RNN output as the query for the attention over the
        # encoder output.
        context_vector, attention_weights = self.attention(
            query=rnn_output,
            value=enc_output,
            mask=mask
        )
        shape_checker(context_vector, ('batch', 't', 'dec_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))
        # _history.append(("decoder_attention_context", context_vector)) # ?

        # Step 4. Eqn. (3): Join the context_vector and rnn_output
        #     [ct; ht] shape: (batch t, value_units + query_units)
        context_and_rnn_output = tf.concat(
            [context_vector, rnn_output], axis=-1)

        # Step 4. Eqn. (3): `at = tanh(Wc@[ct; ht])`
        attention_vector = self.Wc(context_and_rnn_output)
        shape_checker(attention_vector, ('batch', 't', 'dec_units'))
        _history.append(("decoder_attention_vector", attention_vector))

        # Step 5. Generate logit predictions:
        logits = self.fc(attention_vector)
        shape_checker(logits, ('batch', 't', 'output_vocab_size'))
        _history.append(("decoder_logits", logits))

        return _history, logits, attention_weights, state


# Test decoder on example

# decoder = Decoder(output_text_processor.vocabulary_size(), EMBEDDING_DIM, UNITS)

# example_output_tokens = output_text_processor(example_target_batch)
# start_index = output_text_processor.get_vocabulary().index('[START]')
# first_token = tf.constant([[start_index]] * example_output_tokens.shape[0])

# dec_history, logits, attention_weights, dec_state = decoder(
#     new_tokens=first_token,
#     enc_output=example_enc_output,
#     mask=(example_tokens != 0),
#     state=example_enc_state,
# )

# print(f'logits shape: (batch_size, t, output_vocab_size) {logits.shape}')
# print(f'state shape: (batch_size, dec_units) {dec_state.shape}')


"""
Loss
"""
class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self):
        self.name = 'masked_loss'
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none',
        )

    def __call__(self, y_true, y_pred):
        shape_checker = ShapeChecker()
        shape_checker(y_true, ('batch', 't'))
        shape_checker(y_pred, ('batch', 't', 'logits'))

        # Calculate the loss for each item in the batch.
        loss = self.loss(y_true, y_pred)
        shape_checker(loss, ('batch', 't'))

        # Mask off the losses on padding.
        mask = tf.cast(y_true != 0, tf.float32)
        shape_checker(mask, ('batch', 't'))
        loss *= mask

        # Return the total.
        return tf.reduce_sum(loss)

"""
Training class
"""
class TrainTranslator(tf.keras.Model):
    def __init__(self, embedding_dim, units, input_text_processor, output_text_processor,  use_tf_function=True):
        super().__init__()
        # Build the encoder and decoder
        self.encoder = Encoder(
            input_text_processor.vocabulary_size(), embedding_dim, units)
        self.decoder = Decoder(
            output_text_processor.vocabulary_size(), embedding_dim, units)

        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.use_tf_function = use_tf_function
        self.shape_checker = ShapeChecker()

    def train_step(self, inputs):
        self.shape_checker = ShapeChecker()

        # TODO
        if self.use_tf_function:
            return self._tf_train_step(inputs)
        else:
            return self._train_step(inputs)

# %%
# Receives a batch of input_text, target_text from tf.data.Dataset and converts those raw text inputs into token embeddings and masks


def _preprocess(self, input_text, target_text):
    self.shape_checker(input_text, ('batch',))
    self.shape_checker(target_text, ('batch',))

    # Convert the text to token IDs
    input_tokens = self.input_text_processor(input_text)
    self.shape_checker(input_tokens, ('batch', 's'))
    target_tokens = self.output_text_processor(target_text)
    self.shape_checker(target_tokens, ('batch', 't'))

    # Convert IDs to masks.
    input_mask = input_tokens != 0
    self.shape_checker(input_mask, ('batch', 's'))

    target_mask = target_tokens != 0
    self.shape_checker(target_mask, ('batch', 't'))

    return input_tokens, input_mask, target_tokens, target_mask


TrainTranslator._preprocess = _preprocess

# %%


def _train_step(self, inputs):
    input_text, target_text = inputs

    (input_tokens, input_mask, target_tokens,
     target_mask) = self._preprocess(input_text, target_text)

    max_target_length = tf.shape(target_tokens)[1]

    with tf.GradientTape() as tape:
        # Encode the input
        enc_history, enc_output, enc_state = self.encoder(input_tokens)
        self.shape_checker(enc_output, ('batch', 's', 'enc_units'))
        self.shape_checker(enc_state, ('batch', 'enc_units'))

        # Initialize the decoder's state to the encoder's final state.
        # This only works if the encoder and decoder have the same number of
        # units.
        dec_state = enc_state
        loss = tf.constant(0.0)

        for t in tf.range(max_target_length-1):
            # Pass in two tokens from the target sequence:
            # 1. The current input to the decoder.
            # 2. The target for the decoder's next prediction.
            new_tokens = target_tokens[:, t:t+2]
            step_loss, dec_state = self._loop_step(
                new_tokens, input_mask, enc_output, dec_state)
            loss = loss + step_loss

        # Average the loss over all non padding tokens.
        average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))

    # Apply an optimization step
    variables = self.trainable_variables
    gradients = tape.gradient(average_loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))

    # Return a dict mapping metric names to current value
    return {'batch_loss': average_loss}


TrainTranslator._train_step = _train_step

# %%
# Executes the decoder and calculates the incremental loss and the new decoder state `dec_state`


def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):
    input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

    # Run the decoder one step.
    dec_history, logits, attention_weights, dec_state = self.decoder(
        new_tokens=input_token,
        enc_output=enc_output,
        mask=input_mask,
        state=dec_state,
    )
    self.shape_checker(logits, ('batch', 't1', 'logits'))
    self.shape_checker(attention_weights, ('batch', 't1', 's'))
    self.shape_checker(dec_state, ('batch', 'dec_units'))

    # `self.loss` returns the total for non-padded tokens
    y = target_token
    y_pred = logits
    step_loss = self.loss(y, y_pred)

    return step_loss, dec_state


TrainTranslator._loop_step = _loop_step

# %% [markdown]
# **Training step wrapped in `tf.function` for improved performance when training and not debugging**

# %%


@tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]), tf.TensorSpec(dtype=tf.string, shape=[None])]])
def _tf_train_step(self, inputs):
    return self._train_step(inputs)


TrainTranslator._tf_train_step = _tf_train_step

# %% [markdown]
# **Test the training step**

# %%
# v0_translator = TrainTranslator(
#     EMBEDDING_DIM,
#     UNITS,
#     input_text_processor=input_text_processor,
#     output_text_processor=output_text_processor,
#     use_tf_function=False,
# )

# # Configure the loss and optimizer
# v0_translator.compile(
#     optimizer=tf.optimizers.Adam(),
#     loss=MaskedLoss(),
# )

# %%
# The loss should start near this value
# log_vocab_size = np.log(output_text_processor.vocabulary_size())
# log_vocab_size

# %%
# Test training step on one batch of data

# %%time
# step_results = []
# for n in range(10):
#   res = v0_translator.train_step([example_input_batch, example_target_batch])
#   print(n, res)
#   step_results.append(res)

# start_loss = step_results[0]["batch_loss"].numpy()
# assert abs(log_vocab_size - start_loss) < 1, f"First batch loss should be close to {log_vocab_size}"

# print()

# %%
# Switch to `tf.function` and run one step
# (slower because it needs to trace the function)
# v0_translator.use_tf_function = True
# v0_translator.train_step([example_input_batch, example_target_batch])

# %%
# Run again 10 train steps to check if it's faster now with `tf.function`
# (it should be)

# %%time
# for n in range(10):
#   print(n, v0_translator.train_step([example_input_batch, example_target_batch]))
# print()

# %%
# Test if the model quickly overfits a single batch of input

# losses = []
# print("-" * 100, end=" ")
# print("100%")

# for n in range(100):
#     print(".", end="")
#     logs = translator.train_step([example_input_batch, example_target_batch])
#     losses.append(logs["batch_loss"].numpy())

# %%
# plt.plot(losses)

# %%
# Create new copy of the model that will be trained properly

train_translator = TrainTranslator(
    EMBEDDING_DIM,
    UNITS,
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor
)

train_translator.compile(
    optimizer=tf.optimizers.Adam(),
    loss=MaskedLoss(),
)

# %%
# Callback to collect history of batch losses for visualization


class BatchLogs(tf.keras.callbacks.Callback):
    def __init__(self, key):
        self.key = key
        self.logs = []

    def on_train_batch_end(self, n, logs):
        self.logs.append(logs[self.key])


batch_loss = BatchLogs('batch_loss')

# %% [markdown]
# **Train the model**

# %%
# Train!

train_translator.fit(
    train_dataset,
    epochs=1,  # DEBUG
    callbacks=[batch_loss]
)

# %% [markdown]
# ### Final Translator class

# %%
# Translator class that uses the trained model to do the end-to-end text-to-text translation


class Translator(tf.Module):
    def __init__(self, encoder, decoder, input_text_processor, output_text_processor):
        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor

        self.output_token_string_from_index = (
            tf.keras.layers.StringLookup(
                vocabulary=output_text_processor.get_vocabulary(),
                mask_token='',
                invert=True
            )
        )

        # The output should never generate padding, unknown, or start.
        index_from_string = tf.keras.layers.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(),
            mask_token=''
        )

        token_mask_ids = index_from_string(['', '[UNK]', '[START]']).numpy()

        token_mask = np.zeros(
            [index_from_string.vocabulary_size()], dtype=np.bool)
        token_mask[np.array(token_mask_ids)] = True
        self.token_mask = token_mask

        self.start_token = index_from_string(tf.constant('[START]'))
        self.end_token = index_from_string(tf.constant('[END]'))

# %%
# Convert from token IDs to natural language text


def tokens_to_text(self, result_tokens):
    shape_checker = ShapeChecker()
    shape_checker(result_tokens, ('batch', 't'))
    result_text_tokens = self.output_token_string_from_index(result_tokens)
    shape_checker(result_text_tokens, ('batch', 't'))

    result_text = tf.strings.reduce_join(
        result_text_tokens, axis=1, separator=' ')
    shape_checker(result_text, ('batch'))

    result_text = tf.strings.strip(result_text)
    shape_checker(result_text, ('batch',))

    return result_text


Translator.tokens_to_text = tokens_to_text

# %%
# Create final instance of translator
translator = Translator(
    encoder=train_translator.encoder,
    decoder=train_translator.decoder,
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor,
)

# %%
example_output_tokens = tf.random.uniform(
    shape=[5, 2], minval=0, dtype=tf.int64,
    maxval=output_text_processor.vocabulary_size())
translator.tokens_to_text(example_output_tokens).numpy()

# %%
# This function takes the decoder's logit outputs and samples token IDs from that distribution


def sample(self, logits, temperature):
    shape_checker = ShapeChecker()
    # 't' is usually 1 here.
    shape_checker(logits, ('batch', 't', 'vocab'))
    shape_checker(self.token_mask, ('vocab',))

    token_mask = self.token_mask[tf.newaxis, tf.newaxis, :]
    shape_checker(token_mask, ('batch', 't', 'vocab'), broadcast=True)

    # Set the logits for all masked tokens to -inf, so they are never chosen.
    logits = tf.where(self.token_mask, -np.inf, logits)

    if temperature == 0.0:
        new_tokens = tf.argmax(logits, axis=-1)
    else:
        logits = tf.squeeze(logits, axis=1)
        new_tokens = tf.random.categorical(
            logits/temperature, num_samples=1, seed=1)

    shape_checker(new_tokens, ('batch', 't'))

    return new_tokens


Translator.sample = sample

# %%
# Test run this function on some random inputs
example_logits = tf.random.normal(
    [5, 1, output_text_processor.vocabulary_size()], seed=1)
example_output_tokens = translator.sample(example_logits, temperature=1.0)
example_output_tokens

# %% [markdown]
# **Translation loop**

# %%


def translate_unrolled(self, input_text, *, max_length=50, return_attention=True, temperature=1.0):
    batch_size = tf.shape(input_text)[0]
    input_tokens = self.input_text_processor(input_text)
    enc_history, enc_output, enc_state = self.encoder(input_tokens)

    dec_state = enc_state
    new_tokens = tf.fill([batch_size, 1], self.start_token)

    result_tokens = []
    attention = []
    done = tf.zeros([batch_size, 1], dtype=tf.bool)

    for _ in range(max_length):
        dec_history, logits, attention_weights, dec_state = self.decoder(
            new_tokens=new_tokens,
            enc_output=enc_output,
            mask=(input_tokens != 0),
            state=dec_state,
        )

        attention.append(attention_weights)

        new_tokens = self.sample(logits, temperature)

        # If a sequence produces an `end_token`, set it `done`
        done = done | (new_tokens == self.end_token)
        # Once a sequence is done it only produces 0-padding.
        new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)

        # Collect the generated tokens
        result_tokens.append(new_tokens)

        if tf.executing_eagerly() and tf.reduce_all(done):
            break

    # Convert the list of generates token ids to a list of strings.
    result_tokens = tf.concat(result_tokens, axis=-1)
    result_text = self.tokens_to_text(result_tokens)

    if return_attention:
        attention_stack = tf.concat(attention, axis=1)

    history = enc_history + dec_history
    # to differentiate from other returns
    history = {f"_{key}": vec for (key, vec) in history}
    # ps: history has to be returned as top-level items because this function can only return a dict of tensors

    if return_attention:
        attention_stack = tf.concat(attention, axis=1)
        return {"text": result_text, "attention": attention_stack, **history}
    else:
        return {"text": result_text, **history}


Translator.translate = translate_unrolled

# %%
# Define equivalent `tf.function`


@tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
def tf_translate(self, input_text):
    return self.translate(input_text)


Translator.tf_translate = tf_translate

# %% [markdown]
# **Saving trained translator to file**

# %%
MODEL_NAME = "translator"

tf.saved_model.save(
    translator,
    MODEL_NAME,
    signatures={"serving_default": translator.tf_translate}
)

# %% [markdown]
# **Uploading to Google Drive**

# %%
src = f"/content/{MODEL_NAME}/"
dest = "/gdrive/MyDrive/LCT/Y1/Deep Learning/Final Project/models/"

# Rename previous model directory, if any
if MODEL_NAME in os.listdir(dest):
    os.rename(f"{dest}{MODEL_NAME}", f"{dest}{MODEL_NAME}_{int(time())}")

# Copy the current model to the directory with all files recursively
shutil.copytree(src, dest+MODEL_NAME)


# %%
translator = tf.saved_model.load(dest+MODEL_NAME)

# %% [markdown]
# **Testing**

# %%
input_text = tf.constant([
    "Bom dia!"
])

result = translator.translate(
    input_text=input_text
)

print(result['text'][0].numpy().decode())
print()
print(result.keys())

# %%
# separate history items from results
history = {key[1:]: vec for (key, vec) in result.items() if key[0] == "_"}
print("items in history:")
df_history = pd.DataFrame({"item": history.keys(), "shape": [
                          vec.shape for vec in history.values()]})
df_history

# %% [markdown]
# # Classification

# %% [markdown]
# ### Build annotated POS dataset for classifier

# %%

# %%
langs = {"in": "br", "out": "en"}
target = nmt_train_set[langs["out"]].values  # nmt_train_set["en"]


# %% [markdown]
# ## Classifier

# %%
# shape (batch_size, steps, enc_units)
lstm1_out = dict(history)["encoder_lstm_1"]
last_repr = lstm1_out[:, -1, :]  # TO_ASK: -1 to get from last step?

train_X = last_repr  # shape (batch_size, units)

# random y data to test
n_classes = 10
y = tf.random.uniform(shape=(BATCH_SIZE,), minval=0,
                      dtype=tf.int32, maxval=n_classes)  # [2, 1, 3, 9, 7, 1, ...]

encoded_Y = LabelEncoder().fit_transform(y)
# one-hot encoded vector (batch_size, n_classes)
dummy_Y = tf.keras.utils.to_categorical(encoded_Y)

train_Y = dummy_Y

model = tf.keras.Sequential([
    tf.keras.layers.Dense(EMBEDDING_DIM, input_dim=UNITS, activation="relu"),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(n_classes, activation="softmax"),
])

print(model.summary())

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_X, train_Y, epochs=1)

y_pred = model.predict(train_X)
y_pred_classes = np.argmax(y_pred, axis=1)  # get class with the highest prob?
print(y_pred_classes)  # predicting always the same class for every row
