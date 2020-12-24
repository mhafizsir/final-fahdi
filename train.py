import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

NAME = f"Mangrove-{int(time.time())}"

tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = np.array(X / 255.0)
y = np.array(y)


def train_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=X.shape[1:]),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])

    model.fit(X, y, epochs=10, validation_split=0.1)

    model.save("model.h5")


def ConvertH5():
    add_model = tf.keras.models.load_model('model.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(add_model)
    tflite_model = converter.convert()
    open("model.tflite", "wb").write(tflite_model)


if __name__ == '__main__':
    # train_model()
    ConvertH5()