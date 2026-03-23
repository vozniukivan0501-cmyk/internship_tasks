from Convolutional_NN_module import ConvolutionalNN
from FFNN_module import FeedForwardNN
from RandomForest_module import RandomForest

class MnistClassifier:

    MODELS = {
        'cnn': ConvolutionalNN,
        'nn': FeedForwardNN,
        'rf': RandomForest
    }

    def __init__(self, algorithm: str):

        """
        Initialize MnistClassifier with given algorithm.
        """

        if algorithm not in self.MODELS:
            raise ValueError("Invalid algorithm, choose from: cnn, nn, rf")

        self.algorithm = algorithm
        self.model = self.MODELS[algorithm]()

    def train(self, X_train, y_train):

        """
        Calls training script for selected algorithm.
            Args:
                X_train: array-like training data expected shape (n_samples, width, height)
                y_train: array-like training data expected shape (n_samples)
        """

        self.model.train(X_train, y_train)
        print('Training complete')

    def predict(self, X_test):

        """
        Calls prediction script for selected algorithm.
            Args:
                X_test: array-like test data expected shape (n_samples, width, height)
            Returns:
                preds: array-like class integer values, expected shape (n_samples)
        """

        preds = self.model.predict(X_test)
        print('Prediction complete')

        return preds

