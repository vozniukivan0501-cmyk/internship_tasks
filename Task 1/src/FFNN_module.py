from MnistClassifierInterface_module import MnistClassifierInterface
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
import numpy as np

class FeedForwardNN(MnistClassifierInterface):

    def __init__(self):

        """
        Initialize FeedForwardNN for 10-class classification.
        Note: using Flatten() to convert 2D image matrices into 1D vectors for the Dense layers
        Note: using softmax to get probability values for each class.
        """

        self.model = Sequential([
            Input(shape=(28, 28)),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax'),
        ])

        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def train(self, X_train, y_train):
        """
        Training FeedForwardNN model.
            Args:
                X_train: array-like training data expected shape (n_samples, width, height)
                y_train: array-like training data expected shape (n_samples)
        """

        self.model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=2)

    def predict(self, X_test):

        """
        Classifying with trained FeedForwardNN model.
            Args:
                X_test: array-like test data expected shape (n_samples, width, height)
            Return:
                preds: array-like class integer values, expected shape (n_samples)
        """

        preds = self.model.predict(X_test)
        preds = np.argmax(preds, axis=1)

        return preds