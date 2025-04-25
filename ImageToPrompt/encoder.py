import numpy as np
import tensorflow as tf
import keras

def build_cnn_encoder(output_dim=256):
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', pooling='avg')
    base_model.trainable = False  # freeze weights
    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),  # normalize to [0,1]
        base_model,
        tf.keras.layers.Dense(output_dim, activation='relu')
    ])