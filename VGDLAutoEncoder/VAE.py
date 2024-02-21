import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.preprocessing.sequence import pad_sequences
import keras_nlp 

import os
from DescriptionToTokens import get_codes , codes_to_string
import random 
import math 


codes, token_decoder = get_codes("examples/all_games_sp.csv" )
NB_WORDS = token_decoder.next_token 
emmeding_len = 50 


lengths = [len(code) for code in codes]
max_len = round(sum(lengths) / len(lengths)) 

codes = pad_sequences(codes, maxlen=max_len)


random.shuffle(codes)
training = codes[:math.floor(len(codes) * 0.8) ]
test = codes[len(training):]

intermediate_dim = 500


# thanks https://nicgian.github.io/text-generation-vae/  for the code 
# Also https://machinelearningmastery.com/lstm-autoencoders/ 
# also https://keras.io/examples/generative/vae/ 
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    

latent_dim = 50

#emmeding_layer = layers.Embedding(NB_WORDS, emmeding_len)

encoder_inputs = keras.Input(shape=(max_len))
emmeding_layer = layers.Embedding(NB_WORDS, emmeding_len)(encoder_inputs)
x = layers.Bidirectional(layers.LSTM(intermediate_dim, return_sequences=False, recurrent_dropout=0.2), merge_mode='concat')(emmeding_layer)
x = layers.Dense(latent_dim * 2, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

repeated_context = layers.RepeatVector(max_len)
decoder_h = layers.LSTM(intermediate_dim, return_sequences=True, recurrent_dropout=0.2)
decoder_mean = layers.TimeDistributed(layers.Dense(emmeding_len, activation='softmax'))#softmax is applied in the seq2seqloss by tf
h_decoded = decoder_h(repeated_context(z))
x_decoded_mean = decoder_mean(h_decoded)



decoder_input = layers.Input(shape=(latent_dim,))


_h_decoded = decoder_h(repeated_context(decoder_input))
_x_decoded_mean = decoder_mean(_h_decoded)
_x_decoded_mean = layers.Flatten()(_x_decoded_mean) 
_x_decoded_mean = layers.Dense(max_len, activation="relu")(_x_decoded_mean)



#generator = keras.Model(decoder_input, _x_decoded_mean)


decoder = keras.Model(decoder_input,  _x_decoded_mean, name="decoder")
decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            print(reconstruction)
            reconstruction_loss = tf.reduce_mean(keras.losses.mean_absolute_error(data, reconstruction),)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }




def sample(vae):
    latent = []
    for i in range(latent_dim):
        latent.append(random.uniform(0, 1))
    latent = np.array([latent])
    print("latent:" , latent)
    results = vae.decoder.predict(latent, verbose = 0)[0]
    print(results)
    #reconstructed_indexes = np.apply_along_axis(np.argmax, 1, results)
    return codes_to_string(results, token_decoder)

if __name__ == "__main__":
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(training, epochs=50, batch_size=10)
    print(sample(vae))