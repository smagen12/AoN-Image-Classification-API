# app/model.py

import tensorflow as tf

MODEL_PATH = "app/keras_model/animevscartoon_model.keras"

def load_keras_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model
