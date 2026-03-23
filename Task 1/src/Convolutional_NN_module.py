import numpy as np

from MnistClassifierInterface_module import MnistClassifierInterface
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input


class ConvolutionalNN(MnistClassifierInterface):

    def __init__(self):

        """
        Initialize the Convolutional NN model.
        Note: setting input_shape = (28, 28, 1) because of additional channel dimension
        """

        self.model = Sequential([
            Input(shape=(28, 28, 1)),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides= 2),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides= 2),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dense(units=10, activation='softmax')
        ])

        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])



    def train(self, X_train, y_train):
        """
        Train the Convolutional NN model
        Note: using np.expand_dims to reshape (width, height) array to (width, height, channels) array, essential for Conv2D layer
            Args:
                X_train: array-like training data expected shape (n_samples, width, height)
                y_train: array-like training data expected shape (n_samples)
        """

        X_train = np.expand_dims(X_train, axis=3)

        self.model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=2)


    def predict(self, X_test):

        """
        Classifying with the trained Convolutional NN model.
            Note: using np.expand_dims to reshape (width, height) array to (width, height, channels) array, essential for Conv2D layer
            Args:
                X_test: array-like test data expected shape (n_samples, width, height)
            Returns:
                preds: array-like class integer values, expected shape (n_samples)
        """

        X_test = np.expand_dims(X_test, axis=3)

        preds = self.model.predict(X_test)
        preds = np.argmax(preds, axis=1)

        return preds