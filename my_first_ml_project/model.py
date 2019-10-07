import numpy as np
import tensorflow as tf


class Model:

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear'),
        ])

        self.model.compile(optimizer='adam',
                           loss='mean_squared_error',
                           metrics=['mean_squared_error'])

    def train(self):
        self.model.fit(self.train_data, epochs=2)

    def predict(self):
        return self.model.predict(self.train_data)
