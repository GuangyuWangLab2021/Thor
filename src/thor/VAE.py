import warnings

warnings.filterwarnings("ignore")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers


class VAE(keras.Model):
    """ Variational Autoencoder (VAE) class.

    Parameters
    ----------
    keras : keras.Model
        The keras model to be used for the VAE.
        
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.set_verbose_level()

    def set_models(
        self,
        encoder,
        decoder,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder

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
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction)
                )
            )
            kl_loss = -0.5 * (
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
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

    def load_models(
        self,
        encoder_model_path=None,
        decoder_model_path=None,
    ) -> None:
        self.encoder = tf.keras.models.load_model(encoder_model_path)
        self.decoder = tf.keras.models.load_model(decoder_model_path)

    def set_verbose_level(self, verbose_level=0):
        self.verbose = verbose_level

    def set_scaler(self, ):
        self.scaler = MinMaxScaler()

    def encode(self, X, update_scaler=True):
        if not hasattr(self, "scaler"):
            self.set_scaler()
        if update_scaler:
            self.scaler = self.scaler.fit(X)
        X_norm = self.scaler.transform(X)
        return self.encoder.predict(X_norm, verbose=self.verbose)[2]

    def decode(self, z):
        X_decoded_norm = self.decoder.predict(z, verbose=self.verbose)
        return self.scaler.inverse_transform(X_decoded_norm)


def sampling(inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def train_vae(
    X,
    latent_dim=50,
    intermediate_dim=1024,
    learning_rate=0.0005,
    patience=50,
    n_epochs=2500,
    batch_size=256,
    test_size=0.1,
    random_state=12345,
    verbose=1,
    save_prefix=None
):
    """
    Train a Variational Autoencoder (VAE) on the given transcriptomic data.

    Parameters:
    ----------
    X : numpy.ndarray
        The input data, expected to be minmax normalized.
    latent_dim : int, optional
        Dimensionality of the latent space, by default 50.
    intermediate_dim : int, optional
        Dimensionality of the intermediate encoding layer, by default 1024.
    learning_rate : float, optional
        Learning rate for the optimizer, by default 0.0005.
    patience : int, optional
        Number of epochs with no improvement after which training will be stopped, by default 50.
    n_epochs : int, optional
        Number of training epochs, by default 2500.
    batch_size : int, optional
        Number of samples per gradient update, by default 256.
    test_size : float, optional
        Fraction of the data to be used as test set, by default 0.1.
    random_state : int, optional
        Random seed for reproducibility, by default 12345.
    verbose : int, optional
        Verbosity mode, 0 or 1, by default 1.
    save_prefix : str, optional
        Prefix for saving encoder and decoder models, by default None.

    Returns:
    -------
    None
        The trained encoder and decoder models are saved as h5 files.
    """

    X_train, X_test = train_test_split(
        X, test_size=test_size, random_state=random_state
    )
    original_dim = (X_train.shape)[1]

    encoder_inputs = keras.Input(shape=(original_dim, ))
    encoded = layers.Dense(intermediate_dim, activation="relu")(encoder_inputs)
    x_mean = layers.Dense(latent_dim)(encoded)
    x_log_var = layers.Dense(latent_dim)(encoded)
    z = layers.Lambda(sampling, name="encoder_output")([x_mean, x_log_var])
    encoder = keras.Model(
        encoder_inputs, [x_mean, x_log_var, z], name="encoder"
    )

    decoded = layers.Dense(intermediate_dim, activation="relu")(z)
    decoded = layers.Dense(original_dim, activation="sigmoid")(decoded)
    NN = keras.Model(encoder_inputs, decoded, name="vae")
    encoded_input = keras.Input(shape=(latent_dim, ))

    decoder_layer = NN.layers[-2](encoded_input)
    decoder_layer = NN.layers[-1](decoder_layer)
    decoder = keras.Model(encoded_input, decoder_layer, name="decoder")

    my_vae = VAE()
    my_vae.set_models(encoder, decoder)
    my_vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    early = EarlyStopping(
        monitor="reconstruction_loss",
        mode="min",
        patience=patience,
        verbose=verbose
    )
    my_vae.fit(
        X_train,
        None,
        shuffle=True,
        epochs=n_epochs,
        batch_size=batch_size,
        callbacks=[early]
    )
    if save_prefix is None:
        encoder.save("VAE_encoder.h5")
        decoder.save("VAE_decoder.h5")
    else:
        dirname = os.path.dirname(save_prefix)

        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)
        encoder.save(f"{save_prefix}_VAE_encoder.h5")
        decoder.save(f"{save_prefix}_VAE_decoder.h5")



class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, input_array, y=None):
        return self
    
    def transform(self, input_array, y=None):
        return input_array

class IdentityGenerator(IdentityTransformer):
    def __init__(self):
        pass

    def encode(self, input_array):
        return self.fit_transform(input_array)

    def decode(self, encoded_array):
        return self.fit_transform(encoded_array)
