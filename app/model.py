import tensorflow as tf
import json

MODEL_JSON_PATH = "app/keras_model/AnimevsCartoon-model.json"
MODEL_WEIGHTS_PATH = "app/keras_model/animevscartoon_weights.keras"

def load_keras_model():
    with open(MODEL_JSON_PATH, "r") as json_file:
        model_json = json_file.read()

    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(MODEL_WEIGHTS_PATH)
    
    # Compile model with same config used during training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    return model
